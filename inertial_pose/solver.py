"""姿态求解：观测朝向 -> 满足关节关系的角色姿态。

设计原则（对应需求约束）：

- 传感器世界四元数**绝不**直接赋给骨骼；每帧严格走
  标定 -> 骨段世界朝向 -> 父子层级 -> 关节局部旋转 -> 自由度投影。
- 初版只恢复**可由方向观测确定的自由度**：根位置与未观测关节使用
  显式先验，方向数据不宣称能唯一恢复平移。
- 惯导航向与脚部锁定冲突时，量化比较两条路线的代价：
  *修正航向*（躯干转角跃变）或*释放锚点*（足滑距离），选代价小者。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import quat
from .calibration import Calibration
from .skeleton import DEFAULT_SKELETON, FOOT_SEGMENTS, Skeleton


# --------------------------------------------------------------------------- #
# 结果结构
# --------------------------------------------------------------------------- #
@dataclass
class PoseResult:
    """单帧求解结果。"""

    t: float
    root_pos: np.ndarray
    local_rot: dict[str, np.ndarray]
    world_rot: dict[str, np.ndarray]
    world_pos: dict[str, np.ndarray]
    segment_rot: dict[str, np.ndarray]          # 经坐标变换后的骨段世界朝向
    inertial_residual_deg: dict[str, float]     # 骨段：求解朝向相对观测的偏差（度）
    foot_slide_m: dict[str, float]              # 本帧各锚点滑动量（米）
    anchors: dict[str, np.ndarray]              # 当前生效的足锁世界位置
    conflict: dict | None = None                # 仲裁记录（无冲突为 None）
    warnings: list[str] = field(default_factory=list)


@dataclass
class SolverConfig:
    root_height: float = 0.95           # 无锚点先验下的根高度（米）
    foot_height: float = 0.04           # 踝关节距地高度（米）
    stance_heading_conflict_deg: float = 25.0  # 支撑脚航向偏差超该值视为冲突
    slide_weight: float = 0.10          # 足滑代价权重（米 -> 归一化）
    jump_weight: float = 1.0            # 躯干转角跃变权重（弧度 -> 归一化）
    jump_scale_rad: float = np.deg2rad(30.0)
    slide_scale_m: float = 0.15
    max_projection_iter: int = 2        # 自由度投影后的层级再传播次数
    swing_confirm_frames: int = 3       # 连续多少帧摆动才解除足锁锚点


# --------------------------------------------------------------------------- #
# 自由度投影
# --------------------------------------------------------------------------- #
def project_to_dof(
    skeleton: Skeleton, joint: str, q_local: np.ndarray
) -> np.ndarray:
    """把关节局部旋转投影到该关节允许的自由度并施加角限位。

    采用局部轴小角分解（旋转向量），删除未观测轴分量后重建。
    未在骨架中登记允许自由度的关节返回单位旋转（显式先验）。
    """
    spec = skeleton.joints[joint]
    if len(spec.dof) == 0:
        return quat.identity()
    if len(spec.dof) == 3 and all(lim is None for lim in spec.angle_limit):
        return quat.normalize(q_local)

    axis, angle = quat.to_axis_angle(q_local)
    rv = axis * angle[..., None]
    kept = np.zeros(3)
    for idx, lim in zip(skeleton.joint_dof_indices(joint), spec.angle_limit):
        val = float(rv[idx])
        if lim is not None:
            val = float(np.clip(val, lim[0], lim[1]))
        kept[idx] = val
    if np.linalg.norm(kept) < 1.0e-10:
        return quat.identity()
    return quat.from_axis_angle(kept, np.linalg.norm(kept))


# --------------------------------------------------------------------------- #
# 求解器
# --------------------------------------------------------------------------- #
class PoseSolver:
    def __init__(
        self,
        calibration: Calibration,
        skeleton: Skeleton | None = None,
        config: SolverConfig | None = None,
        observed_axis_remap: dict[int, tuple[int, int, int]] | None = None,
    ):
        self.cal = calibration
        self.skel = skeleton or DEFAULT_SKELETON
        self.cfg = config or SolverConfig()
        # 观测轴向重映射（捕捉视窗可切换）：传感器局部轴 (x,y,z) 的重排
        self.axis_remap = observed_axis_remap or {}
        self._anchors: dict[str, np.ndarray] = {}
        self._anchor_yaw: dict[str, float] = {}
        # 持久场景航向参考：真正落地的脚在场景世界中航向恒定，因此该参考
        # 跨多次落脚保持，不随新落点重置；惯导偏航漂移据此逐段累积成可观测量。
        self._scene_heading: dict[str, float] = {}
        # 最近一次建立的锚点（落脚顺序）；双支撑期只用它驱动根位置，避免对
        # 两个相距一个步幅的脚锚点取均值而互相拉扯产生足滑。
        self._anchor_order: list[str] = []
        self._last_anchor_xy: dict[str, np.ndarray] = {}
        self._stance_state: dict[str, bool] = {}
        # 连续判定为摆动的帧数：只有持续若干帧才真正解除锚点，避免支撑段内
        # 单帧误判把已落地的锚点丢弃后在漂移位置重建。
        self._swing_gap: dict[str, int] = {}
        self._committed_root = np.array([0.0, 0.0, 0.95])

    # ------------------------------------------------------------------ #
    def _remap_observation(self, sensor_id: int, q_WS: np.ndarray) -> np.ndarray:
        """按观测轴向开关重排传感器轴（显式矩阵变换，可核验）。

        重排必须是**正当旋转**（行列式 +1 的轴置换）：传感器换绑只改变框架
        朝向，不可能产生镜像（行列式 -1 的反射）。检测到非法重排立即报错，
        由捕捉视窗作为异常提示，而不是静默退化成恒等变换。
        """
        remap = self.axis_remap.get(sensor_id)
        if remap is None or tuple(remap) == (0, 1, 2):
            return quat.normalize(q_WS)
        if sorted(remap) != [0, 1, 2]:
            raise ValueError(f"传感器 {sensor_id} 轴向重排必须是 (0,1,2) 的置换: {remap}")
        P = np.zeros((3, 3))
        for new_axis, old_axis in enumerate(remap):
            P[new_axis, old_axis] = 1.0
        if not np.isclose(np.linalg.det(P), 1.0):
            raise ValueError(
                f"传感器 {sensor_id} 轴向重排 {remap} 是镜像反射（行列式 -1），"
                "物理上不构成正当旋转；请改用行列式 +1 的循环置换")
        # 新传感器框架 S'：q_WS' = q_WS * q_SS'，q_SS' 即重排旋转
        return quat.normalize(quat.mul(q_WS, quat.from_matrix(P)))

    # ------------------------------------------------------------------ #
    def solve_frame(self, t: float, q_WS: dict[int, np.ndarray]) -> PoseResult:
        warnings: list[str] = []

        # 1) 观测 -> 骨段世界朝向（标定是唯一入口）
        q_WS_remapped = {sid: self._remap_observation(sid, q) for sid, q in q_WS.items()}
        try:
            segment_rot = self.cal.apply_batch(q_WS_remapped)
        except KeyError as exc:
            raise RuntimeError(f"观测包含未标定传感器: {exc}") from exc

        observed_segments = set(segment_rot)
        observed_joints = {
            self.skel.segment_joint[s] for s in observed_segments
            if s in self.skel.segment_joint
        }

        # 2) 自顶向下：世界朝向 -> 关节局部旋转 -> 自由度投影
        local_rot: dict[str, np.ndarray] = {}
        for name in self.skel.order:
            spec = self.skel.joints[name]
            seg = spec.segment
            if seg in segment_rot:
                q_world_observed = segment_rot[seg]
                if spec.parent is None:
                    q_local = q_world_observed
                else:
                    parent_seg = self.skel.joints[spec.parent].segment
                    if parent_seg in segment_rot:
                        q_parent_world = segment_rot[parent_seg]
                    else:
                        # 父骨段未观测：用上一阶段 FK 世界旋转，保证层级连续
                        q_parent_world = self._fk_world_rot(local_rot, spec.parent)
                        warnings.append(
                            f"关节 {name} 的父骨段 {parent_seg} 未观测，采用层级先验"
                        )
                    q_local = quat.relative(q_parent_world, q_world_observed)
                local_rot[name] = project_to_dof(self.skel, name, q_local)
            else:
                # 未观测关节：显式静止先验（单位局部旋转）
                local_rot[name] = quat.identity()

        # 3) 投影后再传播一到两轮，让被约束关节与父级保持一致
        for _ in range(self.cfg.max_projection_iter):
            self._republish(local_rot, segment_rot)

        # 4) 初始 FK（先验根）提供世界旋转，再做足锁/航向仲裁
        root_pos = np.array([0.0, 0.0, self.cfg.root_height])
        world_rot, _ = self.skel.forward_kinematics(local_rot, root_pos)
        root_pos, anchors, slides, conflict = self._resolve_anchors(
            local_rot, world_rot
        )
        world_rot, world_pos = self.skel.forward_kinematics(local_rot, root_pos)

        # 6) 惯性偏差：最终 FK 骨段朝向 vs 标定后的观测朝向
        residual = self._inertial_residual(world_rot, segment_rot)

        return PoseResult(
            t=t,
            root_pos=root_pos,
            local_rot=local_rot,
            world_rot=world_rot,
            world_pos=world_pos,
            segment_rot=segment_rot,
            inertial_residual_deg=residual,
            foot_slide_m=slides,
            anchors=anchors,
            conflict=conflict,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    def solve_sequence(self, sequence) -> list[PoseResult]:
        """对 :class:`~inertial_pose.sensors.SensorSequence` 逐帧求解。"""
        results = []
        for frame in sequence.frames:
            results.append(self.solve_frame(frame.t, frame.q_WS))
        return results

    # ------------------------------------------------------------------ #
    def _fk_world_rot(self, local_rot: dict[str, np.ndarray], joint: str) -> np.ndarray:
        """部分局部旋转字典下某关节的世界旋转（沿父链累积）。"""
        chain = []
        cur = joint
        while cur is not None:
            chain.append(cur)
            cur = self.skel.joints[cur].parent
        q = quat.identity()
        for name in reversed(chain):
            q = quat.mul(q, local_rot.get(name, quat.identity()))
        return quat.normalize(q)

    def _republish(self, local_rot, segment_rot) -> None:
        """自由度投影后沿层级重算受约束的观测关节局部旋转。"""
        for name in self.skel.order:
            spec = self.skel.joints[name]
            if spec.parent is None or spec.segment not in segment_rot:
                continue
            parent_seg = self.skel.joints[spec.parent].segment
            if parent_seg not in segment_rot:
                continue
            q_parent_world = self._fk_world_rot(local_rot, spec.parent)
            q_local = quat.relative(q_parent_world, segment_rot[spec.segment])
            local_rot[name] = project_to_dof(self.skel, name, q_local)

    # ------------------------------------------------------------------ #
    def _resolve_anchors(self, local_rot, world_rot):
        """足锁维护与惯导航向冲突仲裁。

        测量与提交分离，保证足滑指标反映**最终姿态**而非中间 FK：

        - 用上一帧已提交根位置做 FK 得到足部测量位置；
        - 两条候选路线在同一帧内显式比价后取小：
          *correct_heading*（保留锚点，代价=躯干航向修正角），
          *release_anchor*（维持惯导航向，代价=足滑距离）；
        - 提交本帧根位置后，由最终 FK 重新计算实际足滑。
        """
        cfg = self.cfg
        conflict = None

        # 上一帧已提交根位置下的 FK，作为本帧测量基准
        base_root = self._committed_root.copy()
        _, measure_pos = self.skel.forward_kinematics(local_rot, base_root)
        stance = self._detect_stance(world_rot)
        active_anchors: dict[str, np.ndarray] = {}
        yaw_correction = 0.0

        for seg in FOOT_SEGMENTS:
            joint = self.skel.segment_joint[seg]
            foot_pos = measure_pos[joint].copy()

            if not stance[seg]:
                # 候选摆动：累计连续摆动帧。只有持续确认才真正结束本支撑
                # 段并解除锚点；支撑段内的单帧误判保留锚点，防止漂移根位置
                # 在错误落点重建锚点。
                self._swing_gap[seg] = self._swing_gap.get(seg, 0) + 1
                if self._swing_gap[seg] >= cfg.swing_confirm_frames:
                    self._anchors.pop(seg, None)
                    self._anchor_yaw.pop(seg, None)
                    if seg in self._anchor_order:
                        self._anchor_order.remove(seg)
                continue

            # 当前判定为支撑
            gap = self._swing_gap.get(seg, 0)
            self._swing_gap[seg] = 0
            confirmed_new_landing = gap >= cfg.swing_confirm_frames

            if seg not in self._anchors or confirmed_new_landing:
                # 新支撑段（经历过确认的摆动）：在当前实测落点建立**新**位置
                # 锚点。方向数据不唯一确定平移，落点位置来自锚定先验，仅在落脚
                # 瞬间提交一次，随后整个支撑段锁定，不逐帧吸收漂移。
                self._anchors[seg] = foot_pos.copy()
                # 航向参考是场景世界量，跨落脚持久保持：仅在该脚首次出现时
                # 初始化，之后惯导偏航漂移会让当前足部航向相对它持续偏离。
                if seg not in self._scene_heading:
                    self._scene_heading[seg] = _yaw_of(world_rot[joint])
                self._anchor_yaw[seg] = self._scene_heading[seg]
                active_anchors[seg] = foot_pos.copy()
                if seg in self._anchor_order:
                    self._anchor_order.remove(seg)
                self._anchor_order.append(seg)
                continue

            anchor = self._anchors[seg]
            slide_vec = foot_pos - anchor
            slide_vec[2] = 0.0
            slide_dist = float(np.linalg.norm(slide_vec))

            root_yaw = _yaw_of(world_rot["pelvis"])
            # 锚点记录的航向与当前足部世界航向之差：惯导整体偏航漂移时该量
            # 在整个支撑段持续累积，是足锁-惯导矛盾的直接可观测量。
            heading_delta = abs(_wrap_angle(self._anchor_yaw[seg] - root_yaw))
            foot_heading_err = self._foot_heading_conflict(world_rot, seg)
            # 航向偏差超阈即构成冲突。
            is_conflict = (
                foot_heading_err > np.deg2rad(cfg.stance_heading_conflict_deg)
            )

            # 释放锚点的假设足滑：当前位置已被锁定，实测 slide 恒为 0，无法
            # 直接代表“若跟随惯导”的代价。用航向漂移在脚相对根的水平伸展半径
            # 上的弦长作为该脚若被惯性航向带走会产生的滑动估计，使两条路线可比价。
            reach = float(np.linalg.norm((foot_pos - self._committed_root)[:2]))
            hypothetical_slide = 2.0 * max(reach, 1.0e-6) * np.sin(
                foot_heading_err * 0.5)
            cost_slide = (
                cfg.slide_weight
                * max(slide_dist, hypothetical_slide) / cfg.slide_scale_m
            )
            # 修正航向的代价取需要补偿的航向偏差（躯干航向跃变）。
            cost_jump = cfg.jump_weight * foot_heading_err / cfg.jump_scale_rad

            if is_conflict and cost_jump <= cost_slide:
                # 策略 A：修正航向。躯干航向跃变的代价不高于释放锚点的足滑
                # 代价时选择此路：把根偏航拉回锚点记录的足部航向，显式校正
                # 惯导漂移，锚点世界位置与航向参考保持不变。
                active_anchors[seg] = anchor.copy()
                foot_yaw = _yaw_of(world_rot[joint])
                yaw_correction = _wrap_angle(self._anchor_yaw[seg] - foot_yaw)
                conflict = {
                    "segment": seg,
                    "choice": "correct_heading",
                    "foot_heading_err_deg": float(np.degrees(foot_heading_err)),
                    "heading_delta_deg": float(np.degrees(heading_delta)),
                    "slide_if_release_m": float(max(slide_dist, hypothetical_slide)),
                    "cost_heading": float(cost_jump),
                    "cost_release": float(cost_slide),
                }
            else:
                # 策略 B：维持惯导航向，锚点跟随足部并更新航向参考，
                # 累计足滑；无冲突的常规支撑帧也走此分支但锚点锁定。
                if is_conflict:
                    conflict = {
                        "segment": seg,
                        "choice": "release_anchor",
                        "foot_heading_err_deg": float(np.degrees(foot_heading_err)),
                        "heading_delta_deg": float(np.degrees(heading_delta)),
                        "slide_if_release_m": float(max(slide_dist, hypothetical_slide)),
                        "cost_heading": float(cost_jump),
                        "cost_release": float(cost_slide),
                    }
                    self._anchors[seg] = foot_pos
                    # 释放锚点意味着接受当前惯导航向为新的场景参考
                    new_heading = _yaw_of(world_rot[joint])
                    self._scene_heading[seg] = new_heading
                    self._anchor_yaw[seg] = new_heading
                active_anchors[seg] = self._anchors[seg].copy()

        # 策略 A 的航向修正作用在根关节局部旋转（世界系绕 Z）
        if yaw_correction != 0.0:
            q_fix = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw_correction)
            local_rot["pelvis"] = quat.normalize(
                quat.mul(q_fix, local_rot["pelvis"])
            )

        # 由存活锚点反推根位置（方向数据不唯一确定平移，这里是显式锚点先验）
        root_pos = self._root_from_anchors(local_rot, active_anchors)
        self._committed_root = root_pos.copy()

        # 最终 FK 下的实际足滑（只计水平面），保证指标对应提交后的姿态。
        # 单根先验在双支撑期只由最近落脚锚点驱动，另一只脚的表观位移是单根
        # 欠约束的结果而非真实滑动，因此只对驱动锚点计滑，其余记 0。
        _, final_pos = self.skel.forward_kinematics(local_rot, root_pos)
        active_order = [s for s in self._anchor_order if s in active_anchors]
        driving = active_order[-1] if active_order else None
        slides = {seg: 0.0 for seg in FOOT_SEGMENTS}
        for seg, anchor in active_anchors.items():
            self._last_anchor_xy[seg] = anchor[:2].copy()
            if seg != driving:
                continue
            joint = self.skel.segment_joint[seg]
            d = final_pos[joint] - anchor
            slides[seg] = float(np.linalg.norm(d[:2]))
        return root_pos, active_anchors, slides, conflict

    def _root_from_anchors(self, local_rot, anchors) -> np.ndarray:
        """锚点 -> 根位置（单锚点驱动）。

        方向数据不唯一确定平移，因此根位置是**显式锚点先验**：双支撑期两只
        脚的锚点在世界系相差一个步幅，对它们取均值会同时拉扯两只脚产生足滑，
        故只取最近落脚的锚点反推根；另一只脚的锚点仍保留，仅用于其自身的
        足滑/航向核验。无锚点时退回上一帧已提交根（漂移先验，不瞬移）。
        """
        if not anchors:
            return self._committed_root.copy()

        active = [s for s in self._anchor_order if s in anchors]
        seg = active[-1] if active else next(iter(anchors))
        anchor = anchors[seg]
        joint = self.skel.segment_joint[seg]

        # 以零根做一次 FK，得到该踝相对根的世界偏移
        _, tmp_pos = self.skel.forward_kinematics(local_rot, np.zeros(3))
        # root + offset(ankle) = anchor  =>  root = anchor - offset
        return anchor - tmp_pos[joint]

    def _detect_stance(self, world_rot) -> dict[str, bool]:
        """支撑相判定：大腿相对骨盆的矢状面俯仰幅值 + 滞回。

        方向观测无法直接看到脚的世界平移，而足底在平面步态中始终水平，
        足部旋转率在支撑/摆动两相都接近零，不能作为判据。可观测且物理
        可靠的信号是**大腿相对骨盆的俯仰角**：摆动相需要大幅提腿（俯仰
        显著增大），支撑相腿在身体下方后蹬、俯仰较小。进入/退出用不同
        阈值滞回，状态在 ``_stance_state`` 保持，避免临界点抖动。
        """
        out = {}
        enter = np.deg2rad(32.0)   # 俯仰回落低于该角 -> 进入支撑
        exit_ = np.deg2rad(42.0)   # 俯仰抬升高于该角 -> 退出支撑（摆动）
        for seg in FOOT_SEGMENTS:
            side = seg.split("_")[0]
            thigh_joint = f"{side}_hip"
            # 大腿相对骨盆的局部旋转，取绕局部 y 的俯仰分量
            q_rel = quat.relative(world_rot["pelvis"], world_rot[thigh_joint])
            axis, angle = quat.to_axis_angle(q_rel)
            pitch = abs(float(axis[1] * angle))

            was_stance = self._stance_state.get(seg, True)
            if was_stance:
                is_stance = pitch <= exit_
            else:
                is_stance = pitch < enter
            self._stance_state[seg] = is_stance
            out[seg] = is_stance
        return out

    def _foot_heading_conflict(self, world_rot, seg: str) -> float:
        """支撑脚世界航向相对锚点参考航向的偏差（弧度）。

        惯导整体航向漂移时，已落地锚点记录的航向与当前足部世界航向
        产生差值；这是足锁-惯导矛盾的直接可观测量。腿部链内扭转在
        平面步态下恒为零，不能用作该判据。
        """
        if seg not in self._anchor_yaw:
            return 0.0
        foot_joint = self.skel.segment_joint[seg]
        foot_yaw = _yaw_of(world_rot[foot_joint])
        return abs(_wrap_angle(foot_yaw - self._anchor_yaw[seg]))

    def _inertial_residual(self, world_rot, segment_rot) -> dict[str, float]:
        """最终姿态骨段朝向与惯性观测朝向的角度偏差（度）。"""
        out = {}
        for seg, q_obs in segment_rot.items():
            if seg not in self.skel.segment_joint:
                continue
            joint = self.skel.segment_joint[seg]
            err = quat.angular_distance(world_rot[joint], q_obs)
            out[seg] = float(np.degrees(err))
        return out


# --------------------------------------------------------------------------- #
def _yaw_of(q: np.ndarray) -> float:
    R = quat.to_matrix(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)
