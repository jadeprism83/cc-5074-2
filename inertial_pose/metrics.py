"""评估指标：角度误差、标定残差、足滑距离、躯干转角跃变。

所有指标均由实际求解结果与合成真值计算，不使用任何绕过标定的数据。
"""

from __future__ import annotations

import numpy as np

from . import quat
from .solver import PoseResult


def joint_angle_errors(truth_local: dict[str, np.ndarray],
                       results: list[PoseResult]) -> dict[str, np.ndarray]:
    """每个关节逐帧的局部旋转角度误差（度）。"""
    out = {}
    for joint, q_true in truth_local.items():
        errs = np.zeros(len(results))
        for k, res in enumerate(results):
            q_est = res.local_rot[joint]
            errs[k] = np.degrees(quat.angular_distance(q_est, q_true[k]))
        out[joint] = errs
    return out


def root_position_error(truth_pos: np.ndarray,
                        results: list[PoseResult]) -> np.ndarray:
    """根位置逐帧误差（米）。

    注意：方向数据不唯一确定平移。无锚点时该误差反映先验设定，
    不应被解读为方向求解的失败。
    """
    est = np.stack([r.root_pos for r in results])
    return np.linalg.norm(est - truth_pos[: len(est)], axis=1)


def total_foot_slide(results: list[PoseResult]) -> dict[str, float]:
    """各脚累计滑动距离（米）。"""
    totals: dict[str, float] = {}
    for res in results:
        for seg, d in res.foot_slide_m.items():
            totals[seg] = totals.get(seg, 0.0) + d
    return totals


def torso_yaw_jumps(results: list[PoseResult]) -> np.ndarray:
    """逐帧躯干（pelvis）航向跃变（度），首帧为 0。"""
    yaws = np.array([_yaw(r.world_rot["pelvis"]) for r in results])
    jumps = np.zeros(len(yaws))
    jumps[1:] = np.degrees(np.diff(yaws))
    jumps = (jumps + 180.0) % 360.0 - 180.0
    return jumps


def calibration_residual_rms(cal) -> dict[str, float]:
    """标定残差（度），直接取自标定结果的每传感器 RMS。"""
    return {seg: float(r) for seg, r in zip(cal.segments, cal.residual_deg)}


def summarize(truth_local, truth_seg, truth_pos, skeleton, results, cal) -> dict:
    """汇总全部指标，供 CLI/视窗/测试共用。"""
    je = joint_angle_errors(truth_local, results)
    se = {}
    for seg, q_true in truth_seg.items():
        joint = skeleton.segment_joint[seg]
        errs = np.array([
            np.degrees(quat.angular_distance(r.world_rot[joint], q_true[k]))
            for k, r in enumerate(results)
        ])
        se[seg] = errs
    return {
        "joint_mean_deg": {j: float(np.mean(e)) for j, e in je.items()},
        "joint_p95_deg": {j: float(np.percentile(e, 95)) for j, e in je.items()},
        "segment_mean_deg": {s: float(np.mean(e)) for s, e in se.items()},
        "segment_p95_deg": {s: float(np.percentile(e, 95)) for s, e in se.items()},
        "root_pos_rmse_m": float(np.sqrt(np.mean(root_position_error(truth_pos, results) ** 2))),
        "foot_slide_total_m": total_foot_slide(results),
        "torso_yaw_jump_max_deg": float(np.max(np.abs(torso_yaw_jumps(results)))),
        "calibration_residual_deg": calibration_residual_rms(cal),
        "conflicts": [r.conflict for r in results if r.conflict is not None],
    }


def _yaw(q: np.ndarray) -> float:
    R = quat.to_matrix(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))
