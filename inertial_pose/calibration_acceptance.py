"""标定链验收：解析运动 -> 传感器观测 -> 标定/坐标转换/关节约束，全链路核验。

本模块**只新增验收逻辑**，不修改部署中的标定/求解器。被验收的量（关节角、
标定残差）全部读取真实 ``PoseResult`` 与 ``Calibration`` 的输出；所有期望值
由本模块**独立推导**（不调用被测函数），从而既能暴露坏结果，也不允许伪报。

核心问题：装反轴 vs 陀螺静偏
----------------------------
传感器**装反了轴**（固定的安装错误，传感器框架 S 内的常数偏移）与**陀螺仪
静偏**（运行时角速度零偏，滤波器航向随时间增长）**能产生相同的静态世界
朝向**。只看一帧静止数据，两者以乘积/参数耦合，无法分辨。区分必须依赖
**运动**：

- 装反轴：标定后骨段朝向的误差在整段内**近似常量**；
- 静偏：误差随**时间**（航向积分）或随**转角**增长。

因此验收器只在观测具备**区分力**时允许提交/更新骨架标定：

- **插入转身**：世界航向跨度足够大 —— 世界航向变了而芯片框架内的固定偏移
  不变，固定错误与随航向增长的偏置可分离；
- **延长静置**：时间跨度足够长且残差随时间线性增长 —— 常量错误与增长错误
  可分离。

无区分力的观测（短暂静置）被 ``CalibrationLedger`` 拒绝签核；已签核的标定
在其**决定性传感器**被拔掉时撤销签核。

坐标约定沿用 :mod:`inertial_pose`：``[w,x,y,z]`` 四元数，Z 向上、角色面向
+X、左 +Y；安装链 ``q_W0B = q_W0W * q_WS * conj(q_BS)``。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import quat
from .calibration import Calibration, calibrate
from .sensors import SensorFrame, SensorSequence
from .skeleton import DEFAULT_SENSOR_PLACEMENT, DEFAULT_SKELETON
from .solver import PoseSolver, SolverConfig
from .synthetic import (
    generate_motion,
    make_calibration,
    random_mount_rotations,
)

_Z = np.array([0.0, 0.0, 1.0])
_UNIT_FACTOR = np.pi / 180.0


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _yaw_of_batch(q: np.ndarray) -> np.ndarray:
    R = quat.to_matrix(np.asarray(q, dtype=float))
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def _rot(axis: np.ndarray, angle) -> np.ndarray:
    return quat.from_axis_angle(np.asarray(axis, dtype=float),
                                np.asarray(angle, dtype=float))


def _seg_for_sensor(placement: dict[str, int], sid: int) -> str:
    return next(seg for seg, s in placement.items() if s == sid)


# --------------------------------------------------------------------------- #
# 1) 解析运动：能力参数化（转身 / 屈膝 / 举臂），全链路一致
# --------------------------------------------------------------------------- #
@dataclass
class AnalyticMotion:
    """一段解析动作的全部产物，可直接驱动部署式求解器。"""

    dt: float
    times: np.ndarray
    truth_local_rot: dict[str, np.ndarray]
    truth_root_pos: np.ndarray
    truth_segment_rot: dict[str, np.ndarray]
    sensor_q_WS: dict[int, np.ndarray]
    placement: dict[str, int]
    skeleton: object
    reference_q_WS: dict[int, np.ndarray]
    reference_seg: dict[str, np.ndarray]
    world_ref_yaw: dict[int, float] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.times)


def _analytic_local_rot(skel, times, turn, dip, arm):
    n = len(times)
    local = {name: np.tile(quat.identity(), (n, 1)) for name in skel.order}

    if dip is not None:
        t0, dur, depth = dip
        u = _smoothstep((times - t0) / max(dur, 1e-9))
        bend = depth * np.sin(np.pi * u)
        y_axis = np.broadcast_to(np.array([0.0, 1.0, 0.0]), (n, 3))
        for side in ("l", "r"):
            local[f"{side}_hip"] = quat.from_axis_angle(y_axis, -0.8 * bend)
            local[f"{side}_knee"] = quat.from_axis_angle(y_axis, -1.4 * bend)
            local[f"{side}_ankle"] = quat.from_axis_angle(y_axis, 0.4 * bend)

    if arm:
        z_axis = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (n, 3))
        y_axis = np.broadcast_to(np.array([0.0, 1.0, 0.0]), (n, 3))
        env = arm * np.sin(np.pi * _smoothstep(times / max(times[-1], 1e-9)))
        for side in ("l", "r"):
            sgn = 1.0 if side == "l" else -1.0
            local[f"{side}_shoulder"] = quat.from_axis_angle(z_axis, sgn * env)
            local[f"{side}_elbow"] = quat.from_axis_angle(y_axis, -0.2 * env)

    if turn is not None:
        t0, dur = turn
        u = _smoothstep((times - t0) / max(dur, 1e-9))
        local["pelvis"] = _rot(np.broadcast_to(_Z, (n, 3)), (np.pi / 2.0) * u)
    return local


def _fk_truth(skel, local, n):
    truth_seg = {seg: np.zeros((n, 4)) for seg in skel.segments}
    root = np.zeros((n, 3))
    root[:, 2] = 0.95
    for k in range(n):
        lk = {j: local[j][k] for j in skel.order}
        world_rot, _ = skel.forward_kinematics(lk, root[k])
        for seg in skel.segments:
            truth_seg[seg][k] = world_rot[skel.segment_joint[seg]]
    return truth_seg, root


def _synthesize(skel, placement, truth_seg, times, mounts, yaw0):
    n = len(times)
    sensor_q: dict[int, np.ndarray] = {}
    ref_q: dict[int, np.ndarray] = {}
    ref_seg: dict[str, np.ndarray] = {}
    yaw_out: dict[int, float] = {}
    for seg, sid in placement.items():
        q_BS = quat.normalize(mounts[sid])
        q_W0W = _rot(np.broadcast_to(_Z, (n, 3)), np.full(n, yaw0[sid]))
        sensor_q[sid] = quat.normalize(
            quat.mul(quat.mul(quat.inv(q_W0W), truth_seg[seg]), q_BS))
        q_W0W_ref = _rot(_Z, yaw0[sid])
        ref_q[sid] = quat.normalize(quat.mul(quat.inv(q_W0W_ref), q_BS))
        ref_seg[seg] = truth_seg[seg][0].copy()
        yaw_out[sid] = float(yaw0[sid])
    return sensor_q, ref_q, ref_seg, yaw_out


def generate_analytic_motion(
    *, duration: float = 6.0, dt: float = 0.05,
    turn: tuple[float, float] | None = None,
    dip: tuple[float, float, float] | None = None,
    arm: float = 0.0,
    mount_q_BS: dict[int, np.ndarray] | None = None,
    world_yaw: dict[int, float] | None = None,
    placement: dict[str, int] | None = None,
    skeleton=None,
) -> AnalyticMotion:
    """生成解析动作（可含转身、屈膝、举臂）及传感器观测 ``q_WS``。

    观测严格按 ``q_WS = conj(q_W0W) * q_W0B * q_BS`` 反推；真值骨段朝向只用
    于标定样本与指标，绝不直接驱动角色。
    """
    skel = skeleton or DEFAULT_SKELETON
    placement = placement or DEFAULT_SENSOR_PLACEMENT
    times = np.arange(0.0, duration, dt)
    n = len(times)
    local = _analytic_local_rot(skel, times, turn, dip, arm)
    truth_seg, root = _fk_truth(skel, local, n)
    mounts = mount_q_BS or {sid: quat.identity() for sid in placement.values()}
    yaw0 = world_yaw or {sid: 0.0 for sid in placement.values()}
    sensor_q, ref_q, ref_seg, yaw_out = _synthesize(
        skel, placement, truth_seg, times, mounts, yaw0)
    return AnalyticMotion(
        dt=dt, times=times, truth_local_rot=local, truth_root_pos=root,
        truth_segment_rot=truth_seg, sensor_q_WS=sensor_q, placement=placement,
        skeleton=skel, reference_q_WS=ref_q, reference_seg=ref_seg,
        world_ref_yaw=yaw_out)


def generate_known_axis_motion(
    *, axis_label: str = "z", rate_deg_s: float = 45.0,
    duration: float = 4.0, dt: float = 0.05, offset_deg: float = 5.0,
    sensor_id: int | None = None,
) -> AnalyticMotion:
    """已知**绕轴匀速动作**（可选轴），l_thigh 上带固定安装偏移。

    这是三类误差注入的基准：每帧绕轴角 ``θ_k = rate*(t_k - t_0)`` 已知，
    且 ``q_WS_k = q_W0B_k * E``、``q_W0B_k = R(θ_k) q_W0B_0``，因此可独立
    推导真值角而无需读取任何被测函数。
    """
    skel = DEFAULT_SKELETON
    placement = DEFAULT_SENSOR_PLACEMENT
    sid = sensor_id if sensor_id is not None else placement["l_thigh"]
    axis = {"x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": _Z.copy()}[axis_label]

    times = np.arange(0.0, duration, dt)
    n = len(times)
    # 变速 S 形曲线（非匀速）：纯匀速下时标差只是一条直线的常量偏移，
    # 无从与起始角差区分；变速让时标差在形状上可辨。
    total = np.deg2rad(rate_deg_s) * max(times[-1] - times[0], dt)
    angle = total * _smoothstep((times - times[0]) /
                                max(times[-1] - times[0], 1e-9))
    local = {name: np.tile(quat.identity(), (n, 1)) for name in skel.order}
    local["pelvis"] = _rot(np.broadcast_to(axis, (n, 3)), angle)
    truth_seg, root = _fk_truth(skel, local, n)

    mounts = {s: quat.identity() for s in placement.values()}
    mounts[sid] = _rot(axis, np.deg2rad(offset_deg))
    sensor_q, ref_q, ref_seg, yaw_out = _synthesize(
        skel, placement, truth_seg, times, mounts,
        {s: 0.0 for s in placement.values()})
    return AnalyticMotion(
        dt=dt, times=times, truth_local_rot=local, truth_root_pos=root,
        truth_segment_rot=truth_seg, sensor_q_WS=sensor_q, placement=placement,
        skeleton=skel, reference_q_WS=ref_q, reference_seg=ref_seg,
        world_ref_yaw=yaw_out)


# --------------------------------------------------------------------------- #
# 2) 装反轴 vs 静偏：相同静态朝向的两种解释
# --------------------------------------------------------------------------- #
def static_equivalence_demo(
    *, angle_deg: float = 10.0, axis_label: str = "z", duration: float = 4.0,
    dt: float = 0.05,
) -> dict:
    """演示「装反轴」与「陀螺静偏」在**静止瞬间**产生相同朝向、随运动分离。

    - 装反轴（固定错误）：偏移 ``E`` 恒定 -> 相对首帧的偏差**恒为 0**；
    - 静偏（增长错误）：偏移 ``E(t)=R(axis, α + b·t)`` 随时间增长。

    返回两者首帧朝向、以及各帧相对首帧偏差，供验收断言静态一致、动态可辨。
    """
    axis = {"x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": _Z.copy()}[axis_label]
    times = np.arange(0.0, duration, dt)
    n = len(times)
    alpha = np.deg2rad(angle_deg)
    # 固定错误：常数偏移
    e_fixed = _rot(axis, np.full(n, alpha))
    # 增长错误：从同一 alpha 起线性增长（b 取 alpha/时长的一半）
    rate = alpha * 0.5 / max(times[-1], 1e-9)
    e_growing = _rot(axis, alpha + rate * (times - times[0]))

    d_fixed = np.array([quat.angular_distance(e_fixed[0], e_fixed[k])
                        for k in range(n)])
    d_growing = np.array([quat.angular_distance(e_growing[0], e_growing[k])
                          for k in range(n)])
    return {
        "static_angle_deg": angle_deg,
        "static_identical": bool(np.isclose(
            quat.angular_distance(e_fixed[0], e_growing[0]), 0.0, atol=1e-12)),
        "fixed_variation_deg": float(np.degrees(d_fixed.max())),
        "growing_variation_deg": float(np.degrees(d_growing.max())),
        "bias_rate_deg_s": float(np.degrees(rate)),
    }


# --------------------------------------------------------------------------- #
# 3) 独立残差参照（不调用 calibration/solver）
# --------------------------------------------------------------------------- #
def reference_residual_deg(
    motion,
    *, q_WS: dict[int, np.ndarray] | None = None,
    truth_seg: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """**独立推导**的逐帧标定残差（度），与被测 ``Calibration`` 无关。

    原理：由静止参考帧 ``q_WS_ref`` 与已知参考骨段朝向 ``q_W0B_ref`` 及世界
    航向反解安装 ``q_BS = conj(q_W0B_ref) q_W0W_ref q_WS_ref``，再用
    ``q_pred = q_W0W q_WS conj(q_BS)`` 与真值骨段朝向比较。刻意不调用
    :func:`inertial_pose.calibration.calibrate`，作为期望值的独立来源。

    兼容 :class:`~inertial_pose.synthetic.SyntheticMotion`：其静止参考对应
    rest pose，缺少 ``reference_seg`` 字段时按单位骨段朝向处理。
    """
    q_WS = q_WS if q_WS is not None else motion.sensor_q_WS
    truth = truth_seg if truth_seg is not None else motion.truth_segment_rot
    reference_seg = getattr(motion, "reference_seg", None)
    out: dict[str, np.ndarray] = {}
    for seg, sid in motion.placement.items():
        if sid not in q_WS or seg not in truth:
            continue
        yaw = float(motion.world_ref_yaw.get(sid, 0.0))
        q_W0W = _rot(_Z, yaw)
        q_WS_ref = quat.normalize(motion.reference_q_WS[sid])
        if reference_seg is not None and seg in reference_seg:
            q_W0B_ref = quat.normalize(reference_seg[seg])
        else:
            q_W0B_ref = quat.identity()
        q_BS = quat.normalize(quat.mul(quat.mul(quat.inv(q_W0B_ref), q_W0W),
                                       q_WS_ref))
        q_pred = quat.mul(quat.mul(q_W0W, np.asarray(q_WS[sid], dtype=float)),
                          quat.inv(q_BS))
        out[seg] = np.degrees(quat.angular_distance(
            quat.normalize(q_pred), np.asarray(truth[seg], dtype=float)))
    return out


# --------------------------------------------------------------------------- #
# 4) 区分力判据 + 标定签核台账
# --------------------------------------------------------------------------- #
@dataclass
class DiscriminationEvidence:
    """一组观测能否区分「装反轴（固定）」与「静偏（增长）」的证据。"""

    yaw_swing_deg: float
    hold_seconds: float
    residual_span_deg: float
    residual_linear_deg_s: float
    decisive: bool
    mode: str          # "turn" | "hold" | "insufficient"
    reason: str

    def as_dict(self) -> dict:
        return {
            "yaw_swing_deg": self.yaw_swing_deg,
            "hold_seconds": self.hold_seconds,
            "residual_span_deg": self.residual_span_deg,
            "residual_linear_deg_s": self.residual_linear_deg_s,
            "decisive": self.decisive,
            "mode": self.mode,
            "reason": self.reason,
        }


def evaluate_discrimination(
    motion,
    *, q_WS: dict[int, np.ndarray] | None = None,
    truth_seg: dict[str, np.ndarray] | None = None,
    use_truth: bool = True,
    hold_threshold_s: float = 3.0,
    swing_threshold_deg: float = 20.0,
    min_residual_slope_deg_s: float = 0.5,
) -> DiscriminationEvidence:
    """判定观测是否足以区分两种解释，并给出依据。

    - **转身路径（仅用可观测量）**：观测世界航向跨度 ``≥ swing_threshold_deg``；
    - **静置路径（需真值参照）**：时长 ``≥ hold_threshold_s`` 且逐帧残差随时间
      线性增长 ``≥ min_residual_slope_deg_s``。``use_truth=False`` 时不计算残差，
      静置路径不启用——这对应“只看观测、不偷看真值”的在线判定。

    验收器可用 ``use_truth=True`` 在受控数据上验证静置路径；捕捉视窗使用
    ``use_truth=False``，仅凭可观测的世界航向跨度判定。
    """
    q_WS = q_WS if q_WS is not None else motion.sensor_q_WS
    times = np.asarray(motion.times, dtype=float)
    hold = float(times[-1] - times[0]) if len(times) > 1 else 0.0

    # 世界航向跨度：取骨干传感器航向（缺省用首个 placement 传感器）
    sid0 = next(iter(motion.placement.values()))
    heading = _yaw_of_batch(q_WS[sid0])
    yaw_swing = float(np.degrees(np.ptp(np.unwrap(heading))))

    span = 0.0
    slope = 0.0
    if use_truth:
        truth = truth_seg if truth_seg is not None else motion.truth_segment_rot
        resid = reference_residual_deg(motion, q_WS=q_WS, truth_seg=truth)
        per_frame = np.stack(list(resid.values())).max(axis=0)
        span = float(per_frame.max() - per_frame.min())
        if hold > 1e-6 and len(times) > 2:
            slope = float(np.polyfit(times, per_frame, 1)[0])

    turned = yaw_swing >= swing_threshold_deg
    held = use_truth and hold >= hold_threshold_s and slope >= min_residual_slope_deg_s
    if turned:
        mode = "turn"
        reason = (f"含转身：世界航向跨度 {yaw_swing:.1f}° ≥ {swing_threshold_deg}°，"
                  "固定偏移与随航向增长的偏置可分离")
    elif held:
        mode = "hold"
        reason = (f"延长静置：{hold:.1f}s 内残差以 {slope:.2f}°/s 增长，"
                  "常量错误与增长错误可分离")
    else:
        mode = "insufficient"
        extra = ("" if use_truth else
                 f"，静置 {hold:.1f}s 无真值参照不做静置判定")
        reason = (f"不足以区分：航向跨度 {yaw_swing:.1f}°（需 ≥{swing_threshold_deg}°）"
                  + extra + "；同一静态朝向既可由装反轴也可由静偏解释")
    return DiscriminationEvidence(
        yaw_swing_deg=yaw_swing, hold_seconds=hold, residual_span_deg=span,
        residual_linear_deg_s=slope, decisive=bool(turned or held),
        mode=mode, reason=reason)


@dataclass
class CalibrationCommit:
    """一次标定签核及其决定性证据签名。"""

    segments: tuple[str, ...]
    sensor_ids: tuple[int, ...]
    decisive_sensors: frozenset[int]
    trigger: str
    evidence: DiscriminationEvidence
    signed: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "segments": list(self.segments),
            "sensor_ids": list(self.sensor_ids),
            "decisive_sensors": sorted(self.decisive_sensors),
            "trigger": self.trigger,
            "signed": self.signed,
            "reason": self.reason,
            "evidence": self.evidence.as_dict(),
        }


class CalibrationLedger:
    """标定签核台账：只允许具备区分力的观测更新标定；拔线撤销相关签核。"""

    def __init__(self) -> None:
        self._signed: list[CalibrationCommit] = []
        self._revoked: list[CalibrationCommit] = []

    @property
    def signed(self) -> list[CalibrationCommit]:
        return list(self._signed)

    @property
    def revoked(self) -> list[CalibrationCommit]:
        return list(self._revoked)

    @property
    def active_sensors(self) -> set[int]:
        active: set[int] = set()
        for c in self._signed:
            active.update(c.sensor_ids)
        return active

    def attempt(self, *, segments, sensor_ids, evidence, decisive_sensors,
                trigger) -> CalibrationCommit:
        signed = bool(evidence.decisive)
        commit = CalibrationCommit(
            segments=tuple(segments), sensor_ids=tuple(sensor_ids),
            decisive_sensors=frozenset(decisive_sensors), trigger=trigger,
            evidence=evidence, signed=signed,
            reason=("签核：观测具备区分力——" + evidence.reason) if signed
            else ("拒绝签核：观测无法区分装反轴与静偏——" + evidence.reason))
        if signed:
            self._signed.append(commit)
        return commit

    def revoke(self, sensor_ids) -> list[CalibrationCommit]:
        dropped = set(sensor_ids)
        revoked, kept = [], []
        for c in self._signed:
            (revoked if c.decisive_sensors & dropped else kept).append(c)
        self._signed = kept
        self._revoked.extend(revoked)
        return revoked


# --------------------------------------------------------------------------- #
# 5) 三类注入：单位 / 左右手系 / 时标差
# --------------------------------------------------------------------------- #
@dataclass
class InjectedObservations:
    """已知绕轴动作上注入三类误差后的观测变体（含独立推导的真值角）。"""

    dt: float
    times: np.ndarray
    sensor_id: int
    segment: str
    axis: np.ndarray
    variants: dict[str, np.ndarray]     # kind -> (N,4)
    truth_angle_rad: np.ndarray         # 独立推导（由 rate 与时间）
    labels: dict[str, dict]

    def measured_angle_rad(self, kind: str) -> np.ndarray:
        """由观测变体（以其自身首帧为参照）测得的绕轴有符号角。"""
        return measure_axis_angle(self.variants[kind], self.axis)


def measure_axis_angle(q_batch: np.ndarray, axis_world: np.ndarray,
                       q_ref: np.ndarray | None = None) -> np.ndarray:
    """相对参考帧、绕给定世界轴的逐帧有符号转角（弧度，独立于生成标签）。"""
    q = np.asarray(q_batch, dtype=float)
    q_ref = quat.normalize(q[0] if q_ref is None else q_ref)
    axis_world = np.asarray(axis_world, dtype=float)
    axis_world = axis_world / np.linalg.norm(axis_world)
    out = np.zeros(len(q))
    for k in range(len(q)):
        dq = quat.mul(quat.inv(q_ref), q[k])
        a, ang = quat.to_axis_angle(dq)
        a_world = quat.rotate_vectors(q_ref, a)
        s = float(np.sign(np.dot(a_world, axis_world)))
        out[k] = (s if s != 0.0 else 1.0) * float(ang)
    return out


def inject_three_errors(
    motion: AnalyticMotion, *, axis_label: str = "z", rate_deg_s: float = 45.0,
    time_offset_s: float = 0.2, sensor_id: int | None = None,
) -> InjectedObservations:
    """在已知绕轴动作上注入 单位 / 左右手系 / 时标差 三类误差。

    - ``unit``：绕轴角幅度按 π/180 缩放（度/弧度单位误用）；
    - ``handedness``：绕轴角镜像（把右手系当左手系）；
    - ``time_shift``：观测流整体平移（时标差）。
    """
    sid = sensor_id if sensor_id is not None else motion.placement["l_thigh"]
    seg = _seg_for_sensor(motion.placement, sid)
    axis = {"x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": _Z.copy()}[axis_label]
    times = motion.times
    dt = motion.dt
    n = len(times)
    q_clean = np.asarray(motion.sensor_q_WS[sid], dtype=float)
    # 真值角由干净观测独立测得（干净观测即已知绕轴动作，无注入）
    truth_angle = measure_axis_angle(q_clean, axis)

    variants: dict[str, np.ndarray] = {"clean": q_clean.copy()}
    labels: dict[str, dict] = {
        "clean": {"kind": "clean", "note": "未注入基线"}}

    # 单位：θ -> θ·π/180，即叠加世界系差量旋转 R((π/180-1)θ)
    q_unit = np.empty_like(q_clean)
    for k in range(n):
        dq = _rot(axis, (float(_UNIT_FACTOR) - 1.0) * float(truth_angle[k]))
        q_unit[k] = quat.normalize(quat.mul(dq, q_clean[k]))
    variants["unit"] = q_unit
    labels["unit"] = {"kind": "unit", "factor": float(_UNIT_FACTOR),
                      "note": "角度按单位误用缩放，幅度比约 π/180"}

    # 左右手系：θ -> -θ，叠加 R(-2θ)（绕轴镜像）
    q_hand = np.empty_like(q_clean)
    for k in range(n):
        dq = _rot(axis, -2.0 * float(truth_angle[k]))
        q_hand[k] = quat.normalize(quat.mul(dq, q_clean[k]))
    variants["handedness"] = q_hand
    labels["handedness"] = {"kind": "handedness", "mirror": -1.0,
                            "note": "绕轴角镜像，右手系当左手系"}

    # 时标差：观测流平移（边界夹取；测得角以自身首帧为参照，等价于一段
    # 未记录的前置静止，时移由内部形状而非两端值判定）
    shift = int(round(time_offset_s / dt))
    q_time = np.empty_like(q_clean)
    for k in range(n):
        src = min(max(k - shift, 0), n - 1)
        q_time[k] = q_clean[src]
    variants["time_shift"] = q_time
    labels["time_shift"] = {"kind": "time_shift", "offset_s": float(time_offset_s),
                            "offset_samples": int(shift), "note": "观测时标整体平移"}

    return InjectedObservations(
        dt=dt, times=times, sensor_id=sid, segment=seg, axis=axis,
        variants=variants, truth_angle_rad=truth_angle, labels=labels)


def classify_injected_error(
    truth_angle_rad: np.ndarray, measured_angle_rad: np.ndarray, *,
    dt: float, lag_tolerance_s: float = 0.05, ratio_tolerance: float = 0.25,
) -> str:
    """把测得的绕轴角序列判为 ``clean``/``unit``/``handedness``/``time_shift``。

    判据仅用幅度比、符号一致性与互相关时移，不读取生成时的注入标签。
    """
    truth = np.asarray(truth_angle_rad, dtype=float)
    measured = np.asarray(measured_angle_rad, dtype=float)
    amp_t = float(np.max(np.abs(truth)))
    amp_m = float(np.max(np.abs(measured)))
    if amp_t < 1e-9:
        return "clean"
    ratio = amp_m / amp_t

    # 单位误用：幅度比约为 π/180（远小于 1）
    if abs(ratio - _UNIT_FACTOR) < 0.5 * _UNIT_FACTOR:
        return "unit"

    # 左右手系：显著样本的符号与真值相反
    nz = np.abs(truth) > np.deg2rad(3.0)
    if np.any(nz):
        agree = float(np.mean(np.sign(truth[nz]) == np.sign(measured[nz])))
        if agree < 0.25:
            return "handedness"

    # 时标差：幅度与符号一致，但存在显著时移
    if abs(ratio - 1.0) <= ratio_tolerance:
        corr = float(np.corrcoef(truth, measured)[0, 1]) if len(truth) > 2 else 1.0
        lag = _best_lag_s(truth, measured, dt)
        if corr > 0.6 and abs(lag) >= lag_tolerance_s:
            return "time_shift"
        return "clean"
    return "unknown"


def _best_lag_s(truth: np.ndarray, measured: np.ndarray, dt: float) -> float:
    """measured 相对 truth 的最优时移（秒，正=measured 滞后）。

    以**归一化残差**最小化求时移：``measured[k] ≈ truth[k - lag]``。纯 ramp
    即使在 0 时移下也有很高原始点积，故必须按残差而非点积判定。
    """
    n = len(truth)
    best_lag, best_res = 0.0, np.inf
    for lag in range(-n // 3, n // 3 + 1):
        if lag >= 0:
            j = slice(lag, n)
            i = slice(0, n - lag)
        else:
            j = slice(0, n + lag)
            i = slice(-lag, n)
        if j.stop - j.start < 4:
            continue
        tm = measured[j]
        tt = truth[i]
        denom = float(np.dot(tt, tt)) + 1e-12
        res = float(np.sum((tm - tt) ** 2)) / denom
        if res < best_res:
            best_res, best_lag = res, lag * dt
    return best_lag


# --------------------------------------------------------------------------- #
# 6) 反例与结构性约束
# --------------------------------------------------------------------------- #
def assert_no_truth_into_solver(solver: PoseSolver) -> None:
    """结构性断言：求解器实例不持有任何 ``truth*`` 字段。"""
    offenders = {a for a in vars(solver) if "truth" in a.lower()}
    if offenders:
        raise AssertionError(
            f"求解器持有真值字段 {sorted(offenders)}，真值姿态不得进入求解")
    if getattr(solver, "cal", None) is None:
        raise AssertionError("求解器缺少标定入口")


def perturb_angle(q_ref: np.ndarray, rel: float) -> np.ndarray:
    """把一个**非零**转角按 ``(1+rel)`` 缩放，模拟单位误用造成的角度错误。"""
    axis, angle = quat.to_axis_angle(q_ref)
    return quat.from_axis_angle(axis, angle * (1.0 + float(rel)))

# --------------------------------------------------------------------------- #
# 7) 求解辅助
# --------------------------------------------------------------------------- #
def _sequence(times, dt, q: dict[int, np.ndarray]) -> SensorSequence:
    return SensorSequence(dt=dt, frames=[
        SensorFrame(t=float(t), q_WS={sid: qq[k] for sid, qq in q.items()})
        for k, t in enumerate(times)])


def solve_motion(motion: AnalyticMotion, cal: Calibration, *,
                 q_WS: dict[int, np.ndarray] | None = None,
                 config: SolverConfig | None = None,
                 axis_remap: dict[int, tuple[int, int, int]] | None = None,
                 drop_sensor: int | None = None):
    obs = dict(q_WS if q_WS is not None else motion.sensor_q_WS)
    if drop_sensor is not None:
        obs.pop(drop_sensor, None)
    solver = PoseSolver(cal, skeleton=motion.skeleton, config=config,
                        observed_axis_remap=axis_remap)
    return solver, solver.solve_sequence(_sequence(motion.times, motion.dt, obs))


def _joint_error_deg(motion, results, joint="l_knee") -> np.ndarray:
    return np.degrees(np.array([
        quat.angular_distance(r.local_rot[joint],
                              motion.truth_local_rot[joint][k])
        for k, r in enumerate(results)]))


# --------------------------------------------------------------------------- #
# 8) 验收器（pedantic：未执行/未通过都判失败）
# --------------------------------------------------------------------------- #
@dataclass
class AcceptanceFailure(AssertionError):
    name: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return f"[{self.name}] {self.detail}"


@dataclass
class AcceptanceCase:
    name: str
    ran: bool = False
    passed: bool = False
    detail: str = ""
    metrics: dict = field(default_factory=dict)


class AcceptanceReport:
    def __init__(self) -> None:
        self.cases: list[AcceptanceCase] = []

    def entry(self, name: str) -> AcceptanceCase:
        case = AcceptanceCase(name=name)
        self.cases.append(case)
        return case

    @property
    def passed(self) -> bool:
        return bool(self.cases) and all(c.ran and c.passed for c in self.cases)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "cases": [
            {"name": c.name, "ran": c.ran, "passed": c.passed,
             "detail": c.detail, "metrics": c.metrics} for c in self.cases]}


def _run_case(report: AcceptanceReport, name: str, fn) -> None:
    case = report.entry(name)
    try:
        case.metrics = fn() or {}
        case.ran = True
        case.passed = True
        case.detail = "通过"
    except AcceptanceFailure as exc:
        case.ran, case.passed, case.detail = True, False, str(exc)
    except Exception as exc:  # 代码错误必须显式暴露，绝不静默跳过
        case.ran, case.passed = False, False
        case.detail = f"{type(exc).__name__}: {exc}"


def _require(cond: bool, name: str, detail: str) -> None:
    if not cond:
        raise AcceptanceFailure(name, detail)


def _case_default(report) -> None:
    def fn():
        m = generate_analytic_motion(
            duration=6.0, turn=(2.0, 1.5), dip=(1.0, 2.0, 0.6))
        cal = make_calibration(m)
        _, res = solve_motion(m, cal)
        knee = _joint_error_deg(m, res, "l_knee")
        ref = reference_residual_deg(m)
        max_ref = max(float(np.max(v)) for v in ref.values())
        _require(knee.max() <= 1e-5, "default", f"膝角误差 {knee.max():.3g}° 超限")
        _require(float(np.max(cal.residual_deg)) <= 1e-6, "default", "被测残差非零")
        _require(max_ref <= 1e-6, "default", f"独立参照残差非零 {max_ref:.3g}°")
        return {"knee_max_deg": float(knee.max()),
                "cal_residual_max_deg": float(np.max(cal.residual_deg)),
                "independent_residual_max_deg": max_ref}
    _run_case(report, "default_turn_and_knee_full_chain", fn)


def _case_formal(report) -> None:
    def fn():
        good = generate_analytic_motion(
            duration=5.0, turn=(1.5, 1.2), dip=(0.5, 1.5, 0.5))
        cal = make_calibration(good)
        _, res_good = solve_motion(good, cal)
        knee_good = float(_joint_error_deg(good, res_good, "l_knee").max())

        # 反例：对**单个**骨段传感器（l_shank）叠加固定倾角偏置。共用偏置会在
        # 相对关节角中相消（见 uniform_bias_cancels 断言），必须局部注入才能
        # 让膝角暴露坏结果。
        q_fix = _rot(np.array([1.0, 0.0, 0.0]), np.deg2rad(8.0))
        bad = dict(good.sensor_q_WS)
        sid = good.placement["l_shank"]
        bad[sid] = np.array([quat.normalize(quat.mul(q_fix, q))
                             for q in good.sensor_q_WS[sid]])
        _, res_bad = solve_motion(good, cal, q_WS=bad)
        knee_bad = float(_joint_error_deg(good, res_bad, "l_knee").max())
        ref = reference_residual_deg(good)
        _require(knee_good <= 1e-5, "formal", f"正例膝角 {knee_good:.3g}° 超限")
        _require(knee_bad > knee_good + 1.0, "formal",
                 f"反例未暴露坏结果：{knee_bad:.3g}° vs 正例 {knee_good:.3g}°")
        _require(max(float(np.max(v)) for v in ref.values()) <= 1e-6,
                 "formal", "独立残差参照非零")

        # 同时断言“共用偏置在相对关节角中相消”——这正是必须局部注入的原因。
        q_u = _rot(_Z, np.deg2rad(6.0))
        uniform = {s: np.array([quat.normalize(quat.mul(q_u, q)) for q in qq])
                   for s, qq in good.sensor_q_WS.items()}
        _, res_u = solve_motion(good, cal, q_WS=uniform)
        knee_uniform = float(_joint_error_deg(good, res_u, "l_knee").max())
        _require(knee_uniform <= 1e-5, "formal",
                 f"共用偏置竟未在相对关节角中相消：{knee_uniform:.3g}°")
        return {"knee_good_deg": knee_good, "knee_bad_deg": knee_bad,
                "knee_uniform_bias_deg": knee_uniform}
    _run_case(report, "formal_positive_and_negative_cases", fn)


def _case_full_chain(report) -> None:
    def fn():
        m = generate_analytic_motion(
            duration=5.0, turn=(1.0, 1.5), dip=(2.0, 1.5, 0.7))
        cal = make_calibration(m)
        worst = 0.0
        for sid in cal.sensor_ids:
            seg = cal.segment_for_sensor(sid)
            pred = cal.apply_batch({sid: m.sensor_q_WS[sid]})[seg]
            worst = max(worst, float(np.max(np.degrees(
                quat.angular_distance(pred, m.truth_segment_rot[seg])))))
        _require(worst <= 1e-6, "chain",
                 f"坐标转换未还原骨段朝向 {worst:.3g}°")
        return {"transform_chain_max_deg": worst}
    _run_case(report, "coordinate_transform_chain_recovers_truth", fn)


def _case_discrimination(report) -> None:
    def fn():
        # (1) 短暂静置、带固定偏移：静态相同，且无区分力 -> 拒绝
        m_static = generate_analytic_motion(
            duration=2.0,
            mount_q_BS={sid: _rot(_Z, np.deg2rad(10.0))
                        for sid in DEFAULT_SENSOR_PLACEMENT.values()})
        ev_static = evaluate_discrimination(m_static)
        _require(not ev_static.decisive, "discrimination",
                 "短暂静置不应被判为有区分力")

        # (2) 插入转身：具备区分力
        m_turn = generate_analytic_motion(duration=5.0, turn=(1.5, 1.5))
        ev_turn = evaluate_discrimination(m_turn)
        _require(ev_turn.decisive, "discrimination",
                 "含转身的观测应被判为有区分力")

        # (3) 延长静置 + 倾角漂移（航向不变但残差增长）：静置路径可辨
        m_bias = _static_hold_with_tilt_bias(duration=6.0, bias_deg_s=3.0)
        ev_hold = evaluate_discrimination(m_bias)
        _require(ev_hold.decisive, "discrimination",
                 "长静置 + 增长残差应可辨")
        _require(abs(ev_hold.yaw_swing_deg) < 20.0, "discrimination",
                 "倾角漂移不应带来显著航向变化（应靠静置路径判定）")
        return {"static": ev_static.as_dict(), "turn": ev_turn.as_dict(),
                "long_hold": ev_hold.as_dict()}
    _run_case(report, "discrimination_gate_turn_or_hold", fn)


def _static_hold_with_tilt_bias(*, duration=6.0, dt=0.05, bias_deg_s=3.0):
    """真值静止、但观测绕世界 X 轴随时间漂移（航向不变、残差增长）。"""
    m = generate_analytic_motion(duration=duration, dt=dt)
    bias = np.deg2rad(bias_deg_s) * (m.times - m.times[0])
    q_bias = _rot(np.broadcast_to(np.array([1.0, 0.0, 0.0]), (len(m.times), 3)),
                  bias)
    m.sensor_q_WS = {sid: quat.normalize(quat.mul(q_bias, qq))
                     for sid, qq in m.sensor_q_WS.items()}
    return m


def _case_decisive_revoke(report) -> None:
    def fn():
        m = generate_analytic_motion(
            duration=5.0, turn=(1.5, 1.5), dip=(0.5, 1.5, 0.5))
        cal = make_calibration(m)
        ev = evaluate_discrimination(m)
        _require(ev.decisive, "revoke", "含转身观测应具备区分力")

        decisive_sid = m.placement["l_thigh"]
        ledger = CalibrationLedger()
        commit = ledger.attempt(segments=cal.segments, sensor_ids=cal.sensor_ids,
                                evidence=ev, decisive_sensors={decisive_sid},
                                trigger="turn")
        _require(commit.signed, "revoke", "含转身观测应签核")

        other = next(s for s in cal.sensor_ids if s != decisive_sid)
        ledger.revoke({other})
        _require(bool(ledger.signed), "revoke", "拔掉非决定性传感器不应撤销签核")

        revoked = ledger.revoke({decisive_sid})
        _require(bool(revoked) and not ledger.signed, "revoke",
                 "拔掉决定性传感器必须撤销签核")
        _require(decisive_sid not in ledger.active_sensors, "revoke",
                 "撤销后决定性传感器仍在活动集合")

        # 无区分力观测：拒绝签核
        m_static = generate_analytic_motion(
            duration=2.0,
            mount_q_BS={sid: _rot(_Z, np.deg2rad(8.0))
                        for sid in DEFAULT_SENSOR_PLACEMENT.values()})
        ev_static = evaluate_discrimination(m_static)
        ledger2 = CalibrationLedger()
        commit2 = ledger2.attempt(segments=cal.segments, sensor_ids=cal.sensor_ids,
                                  evidence=ev_static,
                                  decisive_sensors={decisive_sid}, trigger="static")
        _require(not commit2.signed, "revoke", "无区分力观测不得签核")
        return {"revoked_count": len(revoked),
                "signed_after_revoke": len(ledger.signed),
                "static_signed": commit2.signed}
    _run_case(report, "decisive_sensor_revoke_signoff", fn)


def _case_three_errors(report) -> None:
    def fn():
        motion = generate_known_axis_motion(
            axis_label="z", rate_deg_s=45.0, duration=4.0, dt=0.05, offset_deg=5.0)
        inj = inject_three_errors(motion, axis_label="z", rate_deg_s=45.0,
                                  time_offset_s=0.2)
        results = {}
        for kind in ("clean", "unit", "handedness", "time_shift"):
            measured = inj.measured_angle_rad(kind)
            label = classify_injected_error(
                inj.truth_angle_rad, measured, dt=inj.dt)
            results[kind] = label
            _require(label == kind, "three_errors",
                     f"{kind} 被误判为 {label}（应辨出三类误差）")
        return results
    _run_case(report, "detect_unit_handedness_time_shift", fn)


def _case_fail_on_misconfig(report) -> None:
    def fn():
        motion = generate_analytic_motion(
            duration=4.0, turn=(1.0, 1.0), dip=(0.5, 1.5, 0.5))
        cal = make_calibration(motion)
        sid = motion.placement["l_thigh"]

        # (A) 轴交换：正当置换重排观测应使关节角显著劣化；且不把真值带入求解
        solver_ok = PoseSolver(cal, skeleton=motion.skeleton)
        assert_no_truth_into_solver(solver_ok)
        _, base = solve_motion(motion, cal)
        _, swapped = solve_motion(motion, cal, axis_remap={sid: (1, 2, 0)})
        knee_base = float(_joint_error_deg(motion, base, "l_knee").max())
        knee_swap = float(_joint_error_deg(motion, swapped, "l_knee").max())
        _require(knee_base <= 1e-5, "misconfig", f"基线膝角 {knee_base:.3g}° 非零")
        _require(knee_swap > knee_base + 1.0, "misconfig",
                 f"交换轴未使结果明显劣化：{knee_swap:.3g}°")

        # (B) 错误单位/比例：静止参考转角按错误比例缩放 -> 安装估计错误 -> 残差上升。
        # 用带 30° 安装偏移的已知绕轴动作，静止参考角非零，缩放才有效。
        axis_m = generate_known_axis_motion(
            axis_label="z", rate_deg_s=40.0, duration=4.0, offset_deg=30.0)
        ref_wrong = {s: perturb_angle(axis_m.reference_q_WS[s], 0.2)
                     for s in axis_m.reference_q_WS}
        ref_ok = reference_residual_deg(axis_m)
        ref_bad = _reference_with_ref_q(axis_m, ref_wrong)
        ok_res = max(float(np.max(v)) for v in ref_ok.values())
        bad_res = max(float(np.max(v)) for v in ref_bad.values())
        _require(bad_res > ok_res + 3.0, "misconfig",
                 f"错误单位未暴露残差：{bad_res:.3g}° vs {ok_res:.3g}°")
        return {"knee_base_max_deg": knee_base,
                "knee_swap_max_deg": knee_swap,
                "unit_wrong_residual_deg": bad_res,
                "unit_ok_residual_deg": ok_res,
                "solver_has_truth_field": False}
    _run_case(report, "misconfig_axis_swap_and_unit_must_fail", fn)


def _reference_with_ref_q(motion, ref_q_override):
    """用被改写过的静止参考重新独立推导残差（不改动被测对象）。"""
    saved = motion.reference_q_WS
    try:
        motion.reference_q_WS = ref_q_override
        return reference_residual_deg(motion)
    finally:
        motion.reference_q_WS = saved


def _case_unobservable_dof(report) -> None:
    def fn():
        # (1) 未观测关节退回单位先验（显式），即便存在转身/屈膝
        m = generate_analytic_motion(
            duration=4.0, turn=(1.0, 1.0), dip=(0.5, 1.5, 0.5))
        cal = make_calibration(m)
        drop_sid = m.placement["l_upperarm"]
        _, res = solve_motion(m, cal, drop_sensor=drop_sid)
        _require(np.allclose(res[-1].local_rot["l_elbow"], quat.identity(),
                             atol=1e-12), "unobservable",
                 "未观测关节未退回单位先验")

        # (2) 方向数据不唯一确定平移：用无腿动作（举臂，无足锚点），
        #     根高度必须回到固定先验而非随“真值”变化。
        m_arm = generate_analytic_motion(duration=4.0, arm=0.9)
        cal_arm = make_calibration(m_arm)
        _, res_arm = solve_motion(m_arm, cal_arm)
        roots = np.stack([r.root_pos for r in res_arm])
        _require(np.allclose(roots[:, 2], 0.95, atol=1e-9), "unobservable",
                 "无锚点根高度偏离先验（方向数据不得宣称恢复平移）")
        return {"unobserved_joint_identity": True,
                "root_height_prior": float(roots[0, 2])}
    _run_case(report, "unobservable_dof_and_explicit_prior", fn)


def _case_positive_negative(report) -> None:
    def fn():
        good = generate_analytic_motion(
            duration=4.0, turn=(1.0, 1.2), dip=(0.5, 1.2, 0.5))
        cal = make_calibration(good)
        sid = good.placement["l_thigh"]

        _, pos = solve_motion(good, cal)
        pos_knee = float(_joint_error_deg(good, pos, "l_knee").max())

        # 反例 A：轴误配 -> 关节角劣化
        _, swap = solve_motion(good, cal, axis_remap={sid: (1, 2, 0)})
        swap_knee = float(_joint_error_deg(good, swap, "l_knee").max())

        # 反例 B：静态偏置（固定世界航向偏移）-> 标定残差上升
        q_fix = _rot(_Z, np.deg2rad(8.0))
        biased = {s: np.array([quat.normalize(quat.mul(q_fix, q)) for q in qq])
                  for s, qq in good.sensor_q_WS.items()}
        ref_ok = reference_residual_deg(good)
        ref_bias = reference_residual_deg(good, q_WS=biased)
        ok_res = max(float(np.max(v)) for v in ref_ok.values())
        bias_res = max(float(np.max(v)) for v in ref_bias.values())
        cal_bias = calibrate(
            cal.sensor_ids, cal.segments,
            {s: biased[s] for s in cal.sensor_ids},
            {seg: good.truth_segment_rot[seg] for seg in cal.segments},
            reference=good.reference_q_WS, world_ref_yaw=good.world_ref_yaw)
        _require(pos_knee <= 1e-5, "posneg", f"正例膝角 {pos_knee:.3g}° 超限")
        _require(swap_knee > pos_knee + 1.0, "posneg", "轴误配反例未使关节角劣化")
        _require(bias_res > ok_res + 1.0, "posneg", "静态偏置反例未使残差上升")
        _require(float(np.max(cal_bias.residual_deg)) > 1.0, "posneg",
                 "静态偏置未体现在被测标定残差")
        return {"positive_knee_deg": pos_knee, "axis_swap_knee_deg": swap_knee,
                "ok_residual_deg": ok_res, "bias_residual_deg": bias_res,
                "cal_bias_residual_deg": float(np.max(cal_bias.residual_deg))}
    _run_case(report, "axis_swap_and_static_bias_positive_negative", fn)


def _case_replay_foot_and_knee(report) -> None:
    def fn():
        # 同一组读数交给角色回放：读取脚底落点（踝世界位置/锚点）与膝角。
        m = generate_analytic_motion(
            duration=5.0, turn=(1.5, 1.2), dip=(0.5, 2.0, 0.6))
        cal = make_calibration(m)
        _, res = solve_motion(m, cal)
        foot_joints = ("l_ankle", "r_ankle")
        landing = {j: np.stack([r.world_pos[j] for r in res]) for j in foot_joints}
        # 脚底落点必须是有限、连续（无 NaN/瞬移）的真实计算量
        for j in foot_joints:
            _require(np.all(np.isfinite(landing[j])), "replay",
                     f"{j} 落点含非有限值")
            steps = np.linalg.norm(np.diff(landing[j], axis=0), axis=1)
            _require(float(steps.max()) < 1.0, "replay",
                     f"{j} 落点出现瞬移 {steps.max():.3f} m/帧")
        # 膝角从回放读回，且与真值一致（无噪）
        knee_err = _joint_error_deg(m, res, "l_knee")
        _require(knee_err.max() <= 1e-5, "replay",
                 f"回放膝角误差 {knee_err.max():.3g}° 超限")
        knee_range = float(np.ptp(np.degrees([
            2.0 * np.arccos(np.clip(abs(float(r.local_rot["l_knee"][0])), -1, 1))
            for r in res])))
        _require(knee_range > 1.0, "replay", "膝角在整段内无变化（未读到屈膝）")
        return {"foot_landing_finite": True,
                "knee_max_err_deg": float(knee_err.max()),
                "knee_range_deg": knee_range,
                "foot_landing_first": {j: [float(x) for x in landing[j][0]]
                                       for j in foot_joints}}
    _run_case(report, "replay_reads_foot_landing_and_knee", fn)


def run_acceptance() -> AcceptanceReport:
    """运行全部标定链验收子项，返回可审计报告。"""
    report = AcceptanceReport()
    for case_fn in (
        _case_default, _case_formal, _case_full_chain, _case_discrimination,
        _case_decisive_revoke, _case_three_errors, _case_fail_on_misconfig,
        _case_unobservable_dof, _case_positive_negative, _case_replay_foot_and_knee,
    ):
        case_fn(report)
    return report


# --------------------------------------------------------------------------- #
# 9) 旧样例回归：惯性姿态旧样例必须运行到底
# --------------------------------------------------------------------------- #
def run_legacy_inertial_examples(*, duration: float = 4.0) -> dict:
    """运行既有合成动作旧样例（walk/squat/arm_raise），要求无异常跑完。

    使用既有 :mod:`inertial_pose.synthetic` 动作，确认标定链改动未破坏旧行为。
    任何异常都以 ``ran=False`` 记录，绝不吞掉。
    """
    out: dict = {}
    for action in ("walk", "squat", "arm_raise"):
        try:
            rng = np.random.default_rng(0)
            mounts = random_mount_rotations(None, rng)
            m = generate_motion(action=action, duration=duration, dt=0.05,
                                mount_q_BS=mounts,
                                world_yaw={s: 0.25 for s in mounts})
            cal = make_calibration(m)
            solver = PoseSolver(cal, skeleton=m.skeleton)
            res = solver.solve_sequence(
                _sequence(m.times, m.dt, m.sensor_q_WS))
            knee = np.degrees(np.array([
                quat.angular_distance(r.local_rot["l_knee"],
                                      m.truth_local_rot["l_knee"][k])
                for k, r in enumerate(res)]))
            out[action] = {"ran": True, "frames": len(res),
                           "joint_mean_deg": float(np.mean(knee)),
                           "cal_residual_max_deg": float(np.max(cal.residual_deg))}
        except Exception as exc:  # 必须暴露
            out[action] = {"ran": False,
                           "error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------- #
# 10) 执行证据 / 原始记录核对（不伪报）
# --------------------------------------------------------------------------- #
def verify_raw_record(
    motion: AnalyticMotion, cal: Calibration, results: list,
    *, joint: str = "l_knee",
) -> dict:
    """完全由**原始记录**（观测 + 真值 + 参考帧）核验关节角与标定残差。

    - 关节角误差从逐帧 ``PoseResult.local_rot`` 与原始真值反推；
    - 标定残差从原始观测、静止参考与真值骨段朝向独立重算；
    与验收期望（``run_acceptance`` 报告）对得上才 ``consistent=True``。
    """
    joint_err = _joint_error_deg(motion, results, joint)
    resid = reference_residual_deg(motion)
    resid_max = max(float(np.max(v)) for v in resid.values())
    return {
        "joint": joint,
        "joint_mean_deg": float(np.mean(joint_err)),
        "joint_max_deg": float(np.max(joint_err)),
        "calibration_residual_max_deg": resid_max,
        "calibration_engine_max_deg": float(np.max(cal.residual_deg)),
        "consistent": bool(
            abs(resid_max - float(np.max(cal.residual_deg))) < 1e-6),
    }


def evidence_from_disk(run_report: AcceptanceReport, raw_record: dict) -> dict:
    """把验收结果对应到执行证据与原始记录；两者都通过才算整体通过。"""
    execution_ok = run_report.passed and all(c.ran for c in run_report.cases)
    raw_ok = bool(raw_record.get("consistent", False))
    return {
        "execution_passed": execution_ok,
        "cases_ran": sum(1 for c in run_report.cases if c.ran),
        "cases_total": len(run_report.cases),
        "raw_record_consistent": raw_ok,
        "passed": execution_ok and raw_ok,
    }


# --------------------------------------------------------------------------- #
# 11) 捕捉视窗可直接打开的回归样例 + 异常标注（终端仅辅助）
# --------------------------------------------------------------------------- #
REGRESSION_SAMPLES: tuple[dict, ...] = (
    {"action": "walk", "duration": 6.0, "seed": 2, "noise_deg": 0.0,
     "drift_rate": 0.0, "axis_remap": None,
     "note": "基线：零足滑、惯性偏差趋零；切换轴向应劣化并高亮异常"},
    {"action": "arm_raise", "duration": 3.0, "seed": 1, "noise_deg": 1.5,
     "drift_rate": 0.0, "axis_remap": None,
     "note": "含噪：惯性偏差非零但可控；观察偏差条"},
    {"action": "walk", "duration": 8.0, "seed": 2, "noise_deg": 0.0,
     "drift_rate": 0.12, "axis_remap": None,
     "note": "漂移：触发足锁-航向仲裁，冲突事件在时间轴定位"},
    {"action": "squat", "duration": 6.0, "seed": 5, "noise_deg": 0.0,
     "drift_rate": 0.0, "axis_remap": {"7": [1, 2, 0]},
     "note": "轴向切换：l_thigh 正当置换，姿态与惯性偏差应随之改变"},
)


def regression_samples() -> tuple[dict, ...]:
    """捕捉视窗可直接打开的惯性姿态回归样例。"""
    return REGRESSION_SAMPLES


def anomaly_flags(payload: dict) -> list[str]:
    """标注「角色姿态」与「惯性偏差」相关异常（空列表=两项正常）。"""
    if not payload.get("ok", False):
        return [f"链路失败：{payload.get('error', '未知错误')}"]
    flags: list[str] = []
    joint_p95 = [v["p95"] for v in payload.get("joint_error_deg", {}).values()]
    resid_max = payload.get("residual", {}).get("max_deg", [])
    slide = sum(payload.get("foot_slide_total_m", {}).values())
    if joint_p95 and max(joint_p95) > 8.0:
        flags.append(f"角色姿态异常：关节角误差 p95 达 {max(joint_p95):.2f}°")
    if resid_max and max(resid_max) > 8.0:
        flags.append(f"惯性偏差异常：骨段残差最大 {max(resid_max):.2f}°")
    if slide > 1e-3:
        flags.append(f"足锁异常：累计足滑 {slide:.4f} m")
    return flags
