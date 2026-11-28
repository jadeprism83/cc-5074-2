#!/usr/bin/env python3
"""用已知动作 + 不同安装偏差生成合成数据并核验标定残差。

这是 :mod:`inertial_pose.cli calibrate/run` 的薄封装，便于按多组随机
安装偏置批量复现：真值骨段旋转只进入标定与指标，不绕过标定驱动角色。

用法::

    python3 scripts/generate_synthetic.py --action walk --trials 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from inertial_pose.sensors import sequence_from_orientations
from inertial_pose.solver import PoseSolver
from inertial_pose.synthetic import (
    generate_motion,
    make_calibration,
    random_mount_rotations,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", default="walk",
                    choices=("walk", "squat", "arm_raise"))
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--noise-deg", type=float, default=1.0)
    ap.add_argument("--mount-tilt-deg", type=float, default=20.0)
    args = ap.parse_args()

    rows = []
    for trial in range(args.trials):
        seed = 100 + trial
        rng = np.random.default_rng(seed)
        mounts = random_mount_rotations(None, rng, args.mount_tilt_deg)
        motion = generate_motion(
            action=args.action, duration=args.duration, dt=0.05,
            mount_q_BS=mounts,
            world_yaw={sid: 0.25 for sid in mounts},
            noise_deg=args.noise_deg, seed=seed,
        )
        cal = make_calibration(motion)
        results = PoseSolver(cal, skeleton=motion.skeleton).solve_sequence(
            sequence_from_orientations(motion.sensor_q_WS, motion.dt))
        slide = sum(
            sum(r.foot_slide_m.values()) for r in results)
        rows.append({
            "trial": trial,
            "seed": seed,
            "calibration_max_residual_deg": float(np.max(cal.residual_deg)),
            "foot_slide_total_m": float(slide),
            "num_frames": len(results),
        })

    print(json.dumps({"action": args.action, "rows": rows},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
