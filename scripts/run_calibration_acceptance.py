#!/usr/bin/env python3
"""标定链验收回归：**终端仅辅助**，交互式核验请用角色捕捉视窗。

运行：标定链验收子项（转身/屈膝全链路、区分力门、签核撤销、三类误差、
故意误配、不可观自由度、正反例）+ 旧样例回归。所有期望值由
:mod:`inertial_pose.calibration_acceptance` 独立推导。

图形入口（在浏览器中打开，不是终端任务）::

    python3 -m inertial_pose.capture            # http://127.0.0.1:8000/

用法::

    python3 scripts/run_calibration_acceptance.py --out /tmp/accept.json
    python3 scripts/run_calibration_acceptance.py --require-pass
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inertial_pose.calibration_acceptance import (  # noqa: E402
    run_acceptance,
    run_legacy_inertial_examples,
)


def run_in_clean_dir() -> dict:
    """在**净目录**执行一次验收，避免污染工作区或受其残留影响。

    以 ``PYTHONPATH=仓库根`` 在临时目录启动子进程跑验收，返回子进程是否全过。
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory(prefix="calib-accept-") as tmp:
        env = dict(os.environ)
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c",
             "import sys; from inertial_pose.calibration_acceptance import "
             "run_acceptance; sys.exit(0 if run_acceptance().passed else 1)"],
            cwd=tmp, env=env, capture_output=True, text=True)
    return {"returncode": proc.returncode, "clean_dir": True,
            "stdout": proc.stdout[-400:], "stderr": proc.stderr[-400:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="报告 JSON 输出路径")
    ap.add_argument("--require-pass", action="store_true",
                    help="任一子项未通过则非零退出")
    ap.add_argument("--clean-dir", action="store_true",
                    help="在临时净目录子进程中再跑一次验收")
    args = ap.parse_args()

    report = run_acceptance()
    legacy = run_legacy_inertial_examples()
    payload = {
        "acceptance": report.to_dict(),
        "legacy_examples": legacy,
        "passed": report.passed and all(v.get("ran") for v in legacy.values()),
    }
    if args.clean_dir:
        clean = run_in_clean_dir()
        payload["clean_dir_run"] = clean
        payload["passed"] = payload["passed"] and clean["returncode"] == 0
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"验收报告已写入 {args.out}（passed={payload['passed']}）")
    else:
        print(text)
    if args.require_pass and not payload["passed"]:
        print("验收未通过", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
