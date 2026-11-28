"""命令行辅助入口（**仅辅助** 捕捉视窗）。

图形入口 :mod:`inertial_pose.capture` 才是惯性姿态的主要使用方式；本 CLI
只提供无界面的批处理辅助：标定核验、驱动求解与指标、偏置/漂移扫描。
所有数值链路与视窗共用同一套计算（标定 -> 求解 -> 指标），不存在绕过
标定把真值骨段旋转直接赋给角色的路径。

子命令::

    python3 -m inertial_pose.cli calibrate --action walk
    python3 -m inertial_pose.cli run        --action walk --drift-rate 0.12
    python3 -m inertial_pose.cli sweep      --action walk
    python3 -m inertial_pose.cli bias       --action walk
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import metrics
from .sensors import sequence_from_orientations
from .solver import PoseSolver, SolverConfig
from .synthetic import (
    apply_online_yaw_drift,
    generate_motion,
    make_calibration,
    random_mount_rotations,
)


def _build(args):
    """生成合成动作、随机安装偏差并显式标定，返回 (motion, calibration)。"""
    rng = np.random.default_rng(args.seed)
    mounts = random_mount_rotations(None, rng, args.mount_tilt_deg)
    motion = generate_motion(
        action=args.action,
        duration=args.duration,
        dt=args.dt,
        mount_q_BS=mounts,
        world_yaw={sid: 0.25 for sid in mounts},
        noise_deg=args.noise_deg,
        seed=args.seed,
    )
    return motion, make_calibration(motion)


def _solve(motion, cal, drift_rate, config):
    observed = (
        apply_online_yaw_drift(motion, drift_rate)
        if drift_rate > 0.0 else motion.sensor_q_WS
    )
    solver = PoseSolver(cal, skeleton=motion.skeleton, config=config)
    return solver.solve_sequence(sequence_from_orientations(observed, motion.dt))


# --------------------------------------------------------------------------- #
def cmd_calibrate(args) -> dict:
    motion, cal = _build(args)
    out = {
        "action": args.action,
        "segments": list(cal.segments),
        "world_yaw_deg": [float(np.degrees(y)) for y in cal.world_yaw],
        "residual_rms_deg": [float(r) for r in cal.residual_deg],
        "max_residual_deg": float(np.max(cal.residual_deg)),
    }
    return out


def cmd_run(args) -> dict:
    motion, cal = _build(args)
    config = SolverConfig(
        slide_weight=args.slide_weight, jump_weight=args.jump_weight)
    results = _solve(motion, cal, args.drift_rate, config)
    summary = metrics.summarize(
        motion.truth_local_rot, motion.truth_segment_rot,
        motion.truth_root_pos, motion.skeleton, results, cal)

    joint_mean = float(np.mean(list(summary["joint_mean_deg"].values())))
    payload = {
        "action": args.action,
        "frames": len(results),
        "joint_mean_deg": joint_mean,
        "joint_p95_max_deg": float(max(summary["joint_p95_deg"].values())),
        "segment_p95_max_deg": float(max(summary["segment_p95_deg"].values())),
        "root_pos_rmse_m": summary["root_pos_rmse_m"],
        "foot_slide_total_m": {k: float(v)
                               for k, v in summary["foot_slide_total_m"].items()},
        "torso_yaw_jump_max_deg": summary["torso_yaw_jump_max_deg"],
        "calibration_max_residual_deg": float(np.max(cal.residual_deg)),
        "num_conflicts": len(summary["conflicts"]),
        "conflict_choices": _count_choices(summary["conflicts"]),
    }
    if args.out:
        _write_pose(args.out, motion, cal, results)
        payload["pose_file"] = args.out
    return payload


def cmd_sweep(args) -> dict:
    """航向漂移率 x 仲裁权重扫描：观察修正航向/释放锚点如何随权重翻转。"""
    rows = []
    for drift in (0.0, 0.05, 0.10, 0.15):
        for slide_w, jump_w in ((2.0, 0.2), (0.1, 1.0), (0.01, 5.0)):
            ns = argparse.Namespace(**vars(args))
            motion, cal = _build(ns)
            config = SolverConfig(slide_weight=slide_w, jump_weight=jump_w)
            results = _solve(motion, cal, drift, config)
            summary = metrics.summarize(
                motion.truth_local_rot, motion.truth_segment_rot,
                motion.truth_root_pos, motion.skeleton, results, cal)
            rows.append({
                "drift_rate": drift,
                "slide_weight": slide_w,
                "jump_weight": jump_w,
                "conflicts": len(summary["conflicts"]),
                "choices": _count_choices(summary["conflicts"]),
                "foot_slide_total_m": float(
                    sum(summary["foot_slide_total_m"].values())),
                "torso_yaw_jump_max_deg": summary["torso_yaw_jump_max_deg"],
            })
    return {"action": args.action, "rows": rows}


def cmd_bias(args) -> dict:
    """观测噪声（角度偏置）扫描：噪声增大时误差应单调可控、标定残差上升。"""
    rows = []
    for noise in (0.0, 0.5, 1.0, 2.0, 4.0):
        ns = argparse.Namespace(**vars(args))
        ns.noise_deg = noise
        motion, cal = _build(ns)
        results = _solve(motion, cal, 0.0, SolverConfig())
        summary = metrics.summarize(
            motion.truth_local_rot, motion.truth_segment_rot,
            motion.truth_root_pos, motion.skeleton, results, cal)
        rows.append({
            "noise_deg": noise,
            "joint_mean_deg": float(np.mean(list(summary["joint_mean_deg"].values()))),
            "joint_p95_max_deg": float(max(summary["joint_p95_deg"].values())),
            "calibration_max_residual_deg": float(np.max(cal.residual_deg)),
            "foot_slide_total_m": float(
                sum(summary["foot_slide_total_m"].values())),
        })
    return {"action": args.action, "rows": rows}


def _count_choices(conflicts: list[dict | None]) -> dict[str, int]:
    counts = {"correct_heading": 0, "release_anchor": 0}
    for c in conflicts:
        if c:
            counts[c["choice"]] = counts.get(c["choice"], 0) + 1
    return counts


def _write_pose(path: str, motion, cal, results) -> None:
    """把逐帧求解姿态导出为纯 JSON（位置/局部旋转/惯性偏差/仲裁）。"""
    data = {
        "dt": motion.dt,
        "bones": list(motion.skeleton.order),
        "frames": [
            {
                "t": float(r.t),
                "root_pos": [float(v) for v in r.root_pos],
                "local_rot": {
                    j: [float(v) for v in r.local_rot[j]]
                    for j in motion.skeleton.order
                },
                "world_pos": {
                    j: [float(v) for v in r.world_pos[j]]
                    for j in motion.skeleton.order
                },
                "inertial_residual_deg": {
                    s: float(v) for s, v in r.inertial_residual_deg.items()
                },
                "conflict": r.conflict,
            }
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--action", choices=("walk", "squat", "arm_raise"),
                   default="walk")
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-deg", type=float, default=0.0)
    p.add_argument("--mount-tilt-deg", type=float, default=15.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m inertial_pose.cli",
        description="惯性姿态模块的命令行**辅助**工具；图形入口见 "
                    "`python3 -m inertial_pose.capture`")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="标定并报告安装/世界参考残差")
    _add_common(p_cal)
    p_cal.set_defaults(func=cmd_calibrate)

    p_run = sub.add_parser("run", help="驱动求解并输出指标，可选导出姿态 JSON")
    _add_common(p_run)
    p_run.add_argument("--drift-rate", type=float, default=0.0)
    p_run.add_argument("--slide-weight", type=float,
                       default=SolverConfig.slide_weight)
    p_run.add_argument("--jump-weight", type=float,
                       default=SolverConfig.jump_weight)
    p_run.add_argument("--out", default=None, help="导出逐帧姿态 JSON 的路径")
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser("sweep", help="漂移率 x 仲裁权重扫描")
    _add_common(p_sweep)
    p_sweep.set_defaults(func=cmd_sweep)

    p_bias = sub.add_parser("bias", help="观测噪声偏置扫描")
    _add_common(p_bias)
    p_bias.set_defaults(func=cmd_bias)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
