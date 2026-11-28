#!/usr/bin/env python3
"""偏置/漂移评估：扫描观测噪声与航向漂移，核验误差边界与仲裁方向。

封装 :mod:`inertial_pose.cli` 的 bias/sweep 逻辑并产出一份合并报告：
噪声增大时关节角误差与标定残差应可控上升、足锁保持；漂移增大且权重
不同的两组配置应分别偏向“修正航向”与“释放锚点”。

用法::

    python3 scripts/run_bias_sweep.py --action walk --out /tmp/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inertial_pose.cli import cmd_bias, cmd_sweep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", default="walk",
                    choices=("walk", "squat", "arm_raise"))
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--noise-deg", type=float, default=0.0)
    ap.add_argument("--mount-tilt-deg", type=float, default=15.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {
        "noise_sweep": cmd_bias(args),
        "drift_weight_sweep": cmd_sweep(args),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"报告已写入 {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
