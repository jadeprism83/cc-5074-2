"""已知动作的合成数据生成。

用途（对应交付要求）：

- 用已知动作和**不同安装偏差**生成数据，驱动完整标定+求解链路；
- 真值只在两处合法使用：(1) 标定样本的已知骨段朝向；(2) 测试/指标
  计算。测试中禁止把真值骨骼旋转绕过标定直接传给角色。

传感器观测的合成严格按物理链路反推::

    q_WS = conj(q_W0W) * q_W0B * q_BS

其中 q_BS 是每个传感器独立的安装旋转（安装偏差），q_W0W 是该传感器
导航世界相对场景世界的航向（含可选漂移）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import quat
from .calibration import Calibration, calibrate
from .skeleton import DEFAULT_SKELETON, DEFAULT_SENSOR_PLACEMENT, Skeleton


@dataclass
class SyntheticMotion:
    """一段合成动作的完整产物。"""

    dt: float
    times: np.ndarray
    # 真值：每帧关节局部旋转 {joint: (N,4)}
    truth_local_rot: dict[str, np.ndarray]
    # 真值：根位置 (N,3)
    truth_root_pos: np.ndarray
    # 真值：每帧骨段世界朝向 {segment: (N,4)}
    truth_segment_rot: dict[str, np.ndarray]
    # 传感器观测：{sensor_id: (N,4)} q_WS
    sensor_q_WS: dict[int, np.ndarray]
    # 生成时使用的真值安装旋转 {sensor_id: (4,)}
    true_mount_q_BS: dict[int, np.ndarray]
    # 生成时使用的真值世界航向（弧度）{sensor_id: (N,)}
    true_world_yaw: dict[int, np.ndarray]
    placement: dict[str, int]
    skeleton: Skeleton
    # 静止参考帧：{sensor_id: q_WS_ref}，参考姿态下骨段世界朝向为单位旋转
    reference_q_WS: dict[int, np.ndarray] = field(default_factory=dict)
    # 外部世界参考航向（弧度）{sensor_id: yaw}，对应磁力/外部航向输入
    world_ref_yaw: dict[int, float] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.times)


# --------------------------------------------------------------------------- #
# 动作曲线：返回每帧关节局部旋转与根位置
# --------------------------------------------------------------------------- #
def _leg_ik(thigh_len: float, shank_len: float,
            hip: np.ndarray, ankle: np.ndarray) -> tuple[float, float, float]:
    """矢状面（x-z 平面，骨架铰链绕局部 +y）二连杆逆运动学。

    骨架约定：Z 向上、角色面向 +X、左侧为 +Y；髋/膝/踝铰链绕局部 y，
    正向旋转把 -z 摆向 +x（屈髋向前）。返回**左腿**的 (髋俯仰, 膝弯曲,
    踝俯仰)，右腿由调用方对 y 角整体取反得到镜像。支撑相踝反向补偿
    保持足底水平。
    """
    dx = ankle[0] - hip[0]
    dz = ankle[2] - hip[2]
    dist = float(np.clip(np.hypot(dx, dz), 1.0e-6,
                         thigh_len + shank_len - 1.0e-4))
    # 余弦律：膝内角 -> 弯曲量 bend = pi - 内角（直腿 dist=L1+L2 时为 0），
    # 左腿绕局部 -y 弯曲，故膝局部 y 角取负。
    cos_inner = np.clip(
        (thigh_len**2 + shank_len**2 - dist**2)
        / (2.0 * thigh_len * shank_len),
        -1.0, 1.0)
    knee_bend = float(np.pi - np.arccos(cos_inner))
    # 髋->踝方位角：相对竖直向下 -z，向 +x 为正
    target_ang = float(np.arctan2(dx, -dz))
    cos_gamma = np.clip(
        (thigh_len**2 + dist**2 - shank_len**2) / (2.0 * thigh_len * dist),
        -1.0, 1.0)
    gamma = float(np.arccos(cos_gamma))
    # 数值标定（见 FK 几何）：髋局部 y 角 = gamma - 方位角；膝 = -弯曲量
    hip_pitch = gamma - target_ang
    knee = -knee_bend
    ankle_pitch = -(hip_pitch + knee)
    return hip_pitch, knee, ankle_pitch


def _walk_cycle(skel: Skeleton, times: np.ndarray, speed: float = 0.9):
    """步行：基于落脚约束的步态，支撑脚在世界系零水平滑动。

    前进方向为骨架矢状面 +x（铰链绕局部 y，运动发生在 x-z 平面），
    左右脚的横向分居沿 y，与髋部 rest 偏移同轴，因此平面 IK 可严格满足
    落脚约束。每只脚按占空比交替支撑/摆动；支撑相踝世界位置恒定，再用
    平面 IK 反解髋膝踝，保证生成数据可严格检验足锁。
    """
    n = len(times)
    local = {name: np.tile(quat.identity(), (n, 1)) for name in skel.order}

    thigh_len = abs(skel.joints["l_knee"].rest_offset[2])
    shank_len = abs(skel.joints["l_ankle"].rest_offset[2])
    hip_y = abs(skel.joints["l_hip"].rest_offset[1])
    # 踝世界高度：低于髋足够多，使支撑腿留有真实膝弯曲余量（否则腿接近
    # 完全伸直，平面二连杆无法在髋平移时把踝钉在固定世界点）。
    ground_z = 0.10

    step_period = 0.9
    stride = speed * step_period  # 步幅与根前进严格一致 -> 支撑脚零滑
    root_fwd = speed * (times - times[0])  # 沿 +x
    # 支撑脚被钉在固定世界高度，二连杆平面链无法抵消髋部高度变化，
    # 因此步态根高度保持恒定（垂直起伏不进入可严格检验足锁的合成数据）。
    root_z = np.full_like(times, 0.95)

    # 脚目标：左右脚分居骨盆两侧 y，前进/落脚沿 x。
    # 相位以“周期数”为单位，右脚相对左脚错开半个周期；占空比 0.6 支撑。
    foot = {"l": np.zeros((n, 3)), "r": np.zeros((n, 3))}
    side_y = {"l": hip_y, "r": -hip_y}
    duty = 0.6
    for side, phase_off in (("l", 0.0), ("r", 0.5)):
        for k, t in enumerate(times):
            phase = t / step_period + phase_off
            cyc = phase % 1.0
            cycle_idx = int(np.floor(phase))
            anchor_x = (cycle_idx - phase_off) * stride
            if cyc < duty:
                # 支撑相：踝世界位置在整个支撑段内恒定于本周期落脚点
                foot[side][k] = [anchor_x, 0.0, ground_z]
            else:
                # 摆动相：从当前落脚点迈向下一落脚点，并抬脚；横向 y 由腿
                # rest 偏移经 y 铰链保持，不进入平面 IK 目标
                u = (cyc - duty) / (1.0 - duty)
                x1 = anchor_x + stride
                lift = 0.12 * np.sin(np.pi * u)
                foot[side][k] = [anchor_x + (x1 - anchor_x) * u, 0.0,
                                 ground_z + lift]

    # IK 反解下肢。落脚约束在**世界系**：支撑相踝世界位置恒定。
    # 左右腿的局部框架同构（局部 +y 轴与各段 rest 偏移完全一致），横向分离
    # 只由髋 rest 偏移承担；因此两侧对各自的矢状面 x-z 目标用**同一套**
    # 平面 IK 角度，不做角度取反——镜像完全来自平移偏移而非旋转符号。
    hip_off = {side: np.array(skel.joints[f"{side}_hip"].rest_offset, dtype=float)
               for side in ("l", "r")}
    y_axis = np.broadcast_to(np.array([0.0, 1.0, 0.0]), (n, 3))
    for side in ("l", "r"):
        hips, knees, ankles = [], [], []
        for k in range(n):
            hip_world = np.array([
                root_fwd[k] + hip_off[side][0],
                hip_off[side][1],
                root_z[k] + hip_off[side][2],
            ])
            # 相对髋的腿系目标（髋为原点）。y 铰链不改变横向 y，腿的横向
            # 分离完全由髋 rest 偏移承担，因此平面 IK 只看矢状面 x-z 分量。
            ankle_rel_hip = foot[side][k] - hip_world
            ankle_rel_hip[1] = 0.0
            hp, kn, ap = _leg_ik(thigh_len, shank_len,
                                 np.zeros(3), ankle_rel_hip)
            hips.append(hp)
            knees.append(kn)
            ankles.append(ap)
        local[f"{side}_hip"] = quat.from_axis_angle(y_axis, np.array(hips))
        local[f"{side}_knee"] = quat.from_axis_angle(y_axis, np.array(knees))
        local[f"{side}_ankle"] = quat.from_axis_angle(y_axis, np.array(ankles))

    # 脊柱只保留矢状面俯仰，不引入偏航，保持平面 IK 闭合
    phase = 2.0 * np.pi * (times - times[0]) / step_period
    local["spine"] = quat.from_axis_angle(
        y_axis, 0.04 * np.sin(phase))

    # 手臂绕局部 y 反向摆动（同样在 x-z 平面）；肘弯曲两侧同号
    arm_swing = 0.35 * np.sin(phase)
    local["l_shoulder"] = quat.from_axis_angle(y_axis, arm_swing)
    local["r_shoulder"] = quat.from_axis_angle(y_axis, -arm_swing)
    local["l_elbow"] = quat.from_axis_angle(y_axis, np.full(n, -0.3))
    local["r_elbow"] = quat.from_axis_angle(y_axis, np.full(n, -0.3))

    root_pos = np.zeros((n, 3))
    root_pos[:, 0] = root_fwd
    root_pos[:, 2] = root_z
    return local, root_pos


def _arm_raise_cycle(skel: Skeleton, times: np.ndarray):
    """举臂：主要驱动肩/肘，验证未观测/部分观测链。

    肩在冠状面绕局部 z 外展（左右镜像取反），肘在矢状面绕局部 y 弯曲。
    """
    n = len(times)
    phase = (times * 0.8) % (2.0 * np.pi)
    local = {name: np.tile(quat.identity(), (n, 1)) for name in skel.order}
    elev = 0.9 * (0.5 - 0.5 * np.cos(phase))
    z_axis = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (n, 3))
    y_axis = np.broadcast_to(np.array([0.0, 1.0, 0.0]), (n, 3))
    for side in ("l", "r"):
        # 肩冠状面外展左右镜像（绕 z，符号来自两侧的对称摆动），
        # 肘矢状面弯曲两侧同号（局部 y 框架同构）。
        sgn = 1.0 if side == "l" else -1.0
        local[f"{side}_shoulder"] = quat.from_axis_angle(
            z_axis, sgn * elev)
        local[f"{side}_elbow"] = quat.from_axis_angle(
            y_axis, np.full(n, -0.2) * (1.0 - np.cos(phase)) * 0.5)
    root_pos = np.zeros((n, 3))
    root_pos[:, 2] = 0.95
    return local, root_pos


def _squat_cycle(skel: Skeleton, times: np.ndarray):
    """深蹲：髋膝踝协同弯曲，根平移抵消踝水平位移以保持足底固定。

    真实深蹲双脚全程支撑、足底不滑动；合成器据此反解每帧根位置，
    使生成数据可用于检验足锁（锚点应零水平滑动）。铰链绕局部 y，
    右腿角度为左腿镜像。
    """
    n = len(times)
    phase = (times * 0.7) % (2.0 * np.pi)
    bend = 0.5 * (1.0 - np.cos(phase))
    local = {name: np.tile(quat.identity(), (n, 1)) for name in skel.order}
    y_axis = np.broadcast_to(np.array([0.0, 1.0, 0.0]), (n, 3))
    # 两侧局部 y 框架同构，深蹲髋膝踝角度左右相同，镜像只来自髋 rest 偏移。
    hip_ang = -0.8 * bend
    knee_ang = -1.4 * bend
    ankle_ang = 0.4 * bend
    for side in ("l", "r"):
        local[f"{side}_hip"] = quat.from_axis_angle(y_axis, hip_ang)
        local[f"{side}_knee"] = quat.from_axis_angle(y_axis, knee_ang)
        local[f"{side}_ankle"] = quat.from_axis_angle(y_axis, ankle_ang)

    # 先以零根 FK 求踝偏移，再反解根位置使左踝世界位置恒定
    root_pos = np.zeros((n, 3))
    target_ankle = None
    for k in range(n):
        local_k = {j: local[j][k] for j in skel.order}
        _, tmp_pos = skel.forward_kinematics(local_k, np.zeros(3))
        ankle = tmp_pos[skel.segment_joint["l_foot"]]
        if target_ankle is None:
            target_ankle = ankle.copy()
        root_pos[k] = target_ankle - ankle
    # 抬到标称根高度，保留深蹲引起的相对高度变化
    root_pos[:, 2] += 0.95
    return local, root_pos


_ACTIONS = {
    "walk": _walk_cycle,
    "arm_raise": _arm_raise_cycle,
    "squat": _squat_cycle,
}


# --------------------------------------------------------------------------- #
# 安装偏差与世界航向
# --------------------------------------------------------------------------- #
def random_mount_rotations(placement: dict[str, int] | None, rng: np.random.Generator,
                           max_tilt_deg: float = 20.0) -> dict[int, np.ndarray]:
    """为每个传感器生成随机安装旋转（含航向与小倾角偏差）。"""
    placement = placement or DEFAULT_SENSOR_PLACEMENT
    out = {}
    for sid in placement.values():
        angles = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, size=3))
        out[sid] = quat.normalize(quat.from_euler_xyz(angles))
    return out


def generate_motion(
    action: str = "walk",
    duration: float = 6.0,
    dt: float = 0.05,
    mount_q_BS: dict[int, np.ndarray] | None = None,
    world_yaw: dict[int, float] | None = None,
    yaw_drift_rate: float = 0.0,
    noise_deg: float = 0.0,
    placement: dict[str, int] | None = None,
    skeleton: Skeleton | None = None,
    seed: int = 0,
) -> SyntheticMotion:
    """生成一段合成动作及其传感器观测。

    mount_q_BS      安装旋转；缺省为单位安装（无偏差）
    world_yaw       各传感器初始世界航向偏置（弧度）；缺省为 0
    yaw_drift_rate  世界航向线性漂移（弧度/秒），用于制造足锁冲突
    noise_deg       观测四元数加性方向噪声（度，旋转向量扰动）
    """
    if action not in _ACTIONS:
        raise ValueError(f"未知动作 {action}，可选 {sorted(_ACTIONS)}")
    skel = skeleton or DEFAULT_SKELETON
    placement = placement or DEFAULT_SENSOR_PLACEMENT
    rng = np.random.default_rng(seed)

    times = np.arange(0.0, duration, dt)
    local, root_pos = _ACTIONS[action](skel, times)
    n = len(times)

    # 真值 FK：骨段世界朝向
    truth_seg: dict[str, np.ndarray] = {seg: np.zeros((n, 4)) for seg in skel.segments}
    for k in range(n):
        local_k = {j: local[j][k] for j in skel.order}
        world_rot, _ = skel.forward_kinematics(local_k, root_pos[k])
        for seg in skel.segments:
            truth_seg[seg][k] = world_rot[skel.segment_joint[seg]]

    mounts = mount_q_BS or {sid: quat.identity() for sid in placement.values()}
    yaw0 = world_yaw or {sid: 0.0 for sid in placement.values()}

    sensor_q: dict[int, np.ndarray] = {}
    true_yaw: dict[int, np.ndarray] = {}
    reference_q: dict[int, np.ndarray] = {}
    world_ref: dict[int, float] = {}
    for seg, sid in placement.items():
        q_BS = quat.normalize(mounts[sid])
        yaws = np.full(n, yaw0[sid]) + yaw_drift_rate * (times - times[0])
        true_yaw[sid] = yaws
        q_W0W = quat.from_axis_angle(
            np.broadcast_to(np.array([0.0, 0.0, 1.0]), (n, 3)), yaws)
        # q_WS = conj(q_W0W) * q_W0B * q_BS
        q = quat.mul(quat.mul(quat.inv(q_W0W), truth_seg[seg]), q_BS)
        if noise_deg > 0.0:
            perturb = rng.normal(0.0, np.radians(noise_deg), size=(n, 3))
            q_noise = quat.from_axis_angle(
                perturb, np.linalg.norm(perturb, axis=1))
            q = quat.mul(q, q_noise)
        sensor_q[sid] = quat.normalize(q)

        # 静止参考帧：rest pose 下 q_W0B_ref=I，故 q_WS_ref = conj(q_W0W) * q_BS
        q_W0W_ref = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw0[sid])
        reference_q[sid] = quat.normalize(
            quat.mul(quat.inv(q_W0W_ref), q_BS)
        )
        world_ref[sid] = float(yaw0[sid])

    return SyntheticMotion(
        dt=dt,
        times=times,
        truth_local_rot=local,
        truth_root_pos=root_pos,
        truth_segment_rot=truth_seg,
        sensor_q_WS=sensor_q,
        true_mount_q_BS=mounts,
        true_world_yaw=true_yaw,
        placement=placement,
        skeleton=skel,
        reference_q_WS=reference_q,
        world_ref_yaw=world_ref,
        extra={"action": action},
    )


# --------------------------------------------------------------------------- #
# 标定样本生成（静止参考 + 标定动作）
# --------------------------------------------------------------------------- #
def apply_online_yaw_drift(motion: SyntheticMotion,
                           drift_rate: float) -> dict[int, np.ndarray]:
    """在标定后的观测流上叠加运行期世界航向漂移。

    标定阶段使用无漂移数据得到安装旋转与初始世界参考；运行期 IMU
    航向以 ``drift_rate``（弧度/秒）缓慢旋转。对每个传感器观测左乘
    世界系 Z 旋转，返回新的 q_WS 字典，用于驱动足锁-航向冲突仲裁。
    """
    out = {}
    for sid, q in motion.sensor_q_WS.items():
        n = q.shape[0]
        q_drift = quat.from_axis_angle(
            np.broadcast_to(np.array([0.0, 0.0, 1.0]), (n, 3)),
            drift_rate * (motion.times - motion.times[0]),
        )
        out[sid] = quat.normalize(quat.mul(q_drift, q))
    return out


def make_calibration(motion: SyntheticMotion,
                     sample_count: int = 30) -> Calibration:
    """从静止参考帧 + 动作前若干帧构造显式标定。

    真值骨段朝向只在此处进入标定流程；求解器之后只能见到传感器观测，
    测试不得把真值骨骼旋转绕过本标定直接传给角色。
    """
    k = min(sample_count, motion.num_frames)
    sensor_ids = tuple(motion.placement[s] for s in motion.placement)
    segments = tuple(motion.placement.keys())
    q_WS = {sid: motion.sensor_q_WS[sid][:k] for sid in sensor_ids}
    q_W0B = {seg: motion.truth_segment_rot[seg][:k] for seg in segments}
    return calibrate(
        sensor_ids, segments, q_WS, q_W0B,
        reference=motion.reference_q_WS,
        world_ref_yaw=motion.world_ref_yaw,
    )
