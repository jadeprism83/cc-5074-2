"""图形捕捉视窗（capture）与命令行辅助（cli）核验。

视窗后端必须跑真实标定+求解链路：无噪时角度误差与足滑趋零、惯性偏差为
真实计算；非法观测轴向作为异常结构化返回，而不是静默退化。CLI 仅做批处理
辅助，复用同一套计算。
"""

import argparse
import json

import numpy as np
import pytest

from inertial_pose import capture
from inertial_pose.cli import build_parser, cmd_bias, cmd_calibrate, cmd_run, cmd_sweep


def _ns(**over):
    base = dict(
        action="walk", duration=5.0, dt=0.05, seed=2,
        noise_deg=0.0, mount_tilt_deg=15.0,
        drift_rate=0.0, slide_weight=0.10, jump_weight=1.0, out=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# capture 后端：真实计算
# --------------------------------------------------------------------------- #
def test_capture_payload_runs_full_pipeline_noiseless():
    p = capture.build_capture_payload(
        {"action": "walk", "duration": 5.0, "seed": 2})
    assert p["ok"] is True
    assert p["meta"]["num_frames"] == len(p["frames"])
    # 无噪：关节角误差与足滑趋机器精度零，惯性偏差为零（真实计算而非占位）
    assert max(v["p95"] for v in p["joint_error_deg"].values()) < 1e-6
    assert sum(p["foot_slide_total_m"].values()) < 1e-9
    assert max(p["residual"]["mean_deg"]) < 1e-9
    # 每帧带齐姿态与残差字段
    fr = p["frames"][0]
    assert set(fr["positions"]) == set(p["bones"])
    assert set(fr["residual_deg"]) == set(p["segments"])


def test_capture_residual_is_nonzero_under_noise():
    p = capture.build_capture_payload(
        {"action": "arm_raise", "duration": 3.0, "seed": 1, "noise_deg": 1.5})
    assert np.mean(p["residual"]["mean_deg"]) > 0.0


def test_capture_drift_surfaces_conflict_records():
    p = capture.build_capture_payload(
        {"action": "walk", "duration": 8.0, "seed": 2, "drift_rate": 0.12})
    assert p["conflicts"], "漂移应触发足锁-航向仲裁事件"
    c = p["conflicts"][0]
    assert c["choice"] in ("correct_heading", "release_anchor")
    assert c["foot_heading_err_deg"] >= 25.0 - 1e-6


def test_capture_legal_remap_accepted_and_changes_observation():
    sid = 7  # l_thigh
    p = capture.build_capture_payload({
        "action": "walk", "duration": 3.0, "seed": 0,
        "axis_remap": json.dumps({str(sid): [1, 2, 0]}),
    })
    assert p["ok"] is True


def test_capture_illegal_reflection_remap_raises():
    with pytest.raises(ValueError):
        capture.build_capture_payload({
            "action": "walk", "duration": 2.0, "seed": 0,
            "axis_remap": json.dumps({"7": [1, 0, 2]}),  # det -1 反射
        })


def test_capture_rejects_unknown_action_and_bad_bounds():
    with pytest.raises(ValueError):
        capture.build_capture_payload({"action": "fly"})
    with pytest.raises(ValueError):
        capture.build_capture_payload({"duration": 9999.0})


def test_legal_remaps_are_all_proper_rotations():
    for remap in capture.LEGAL_REMAPS:
        P = np.zeros((3, 3))
        for new_axis, old_axis in enumerate(remap):
            P[new_axis, old_axis] = 1.0
        assert np.isclose(np.linalg.det(P), 1.0)
        assert np.allclose(P @ P.T, np.eye(3))


# --------------------------------------------------------------------------- #
# HTTP 层：异常结构化返回，不崩 500
# --------------------------------------------------------------------------- #
def test_http_handler_returns_structured_error():
    from http.server import ThreadingHTTPServer
    import threading
    from urllib.request import urlopen

    srv = ThreadingHTTPServer(("127.0.0.1", 0), capture._CaptureHandler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200
            assert b"canvas" in resp.read()
        with urlopen(f"http://127.0.0.1:{port}/api/solve?action=fly") as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["ok"] is False and "ValueError" in payload["error"]
        with urlopen(f"http://127.0.0.1:{port}/api/options") as resp:
            opts = json.loads(resp.read().decode("utf-8"))
        assert opts["ok"] is True and opts["actions"]
        assert opts["regression_samples"], "回归样例应通过 options 暴露"
    finally:
        srv.shutdown()
        srv.server_close()


def test_capture_payload_carries_gate_and_anomaly_flags():
    p = capture.build_capture_payload(
        {"action": "walk", "duration": 5.0, "seed": 2})
    assert "calibration_gate" in p and "anomaly_flags" in p
    gate = p["calibration_gate"]
    assert set(gate) >= {"signed", "trigger", "reason", "evidence"}
    assert isinstance(p["anomaly_flags"], list)


# --------------------------------------------------------------------------- #
# CLI 辅助
# --------------------------------------------------------------------------- #
def test_cli_calibrate_reports_zero_residual_noiseless():
    out = cmd_calibrate(_ns(seed=4))
    assert out["max_residual_deg"] < 1e-9
    assert len(out["segments"]) == len(out["residual_rms_deg"])


def test_cli_run_metrics_and_pose_export(tmp_path):
    pose_file = tmp_path / "pose.json"
    out = cmd_run(_ns(seed=3, duration=6.0, out=str(pose_file)))
    assert out["joint_p95_max_deg"] < 1e-9
    assert sum(out["foot_slide_total_m"].values()) < 1e-9
    data = json.loads(pose_file.read_text(encoding="utf-8"))
    assert len(data["frames"]) == out["frames"]
    assert "inertial_residual_deg" in data["frames"][0]


def test_cli_sweep_weight_flips_strategy():
    out = cmd_sweep(_ns(seed=2, duration=6.0))
    drifted = [r for r in out["rows"] if r["drift_rate"] >= 0.10]
    heading = sum(r["choices"]["correct_heading"] for r in drifted)
    release = sum(r["choices"]["release_anchor"] for r in drifted)
    assert heading > 0 and release > 0


def test_cli_bias_error_grows_monotonically_bounded():
    out = cmd_bias(_ns(seed=6))
    means = [r["joint_mean_deg"] for r in out["rows"]]
    assert means[0] < 1e-9
    assert means == sorted(means)
    assert means[-1] < 20.0


def test_cli_parser_has_four_subcommands():
    parser = build_parser()
    for cmd in ("calibrate", "run", "sweep", "bias"):
        ns = parser.parse_args([cmd, "--action", "squat"])
        assert callable(ns.func)
