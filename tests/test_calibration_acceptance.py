"""标定链验收的 pytest 封装：把验收器与独立参照一同入库。

这些测试既验证**验收器本身能辨别正反例**，也验证被验收的部署链路
（标定 + 求解）在受控数据上确实正确。所有期望值来自
``inertial_pose.calibration_acceptance`` 的独立推导，不读取被测函数实现。
"""

from __future__ import annotations

import json

import numpy as np

from inertial_pose import capture, quat
from inertial_pose.calibration_acceptance import (
    CalibrationLedger,
    evaluate_discrimination,
    evidence_from_disk,
    generate_analytic_motion,
    generate_known_axis_motion,
    inject_three_errors,
    classify_injected_error,
    reference_residual_deg,
    run_acceptance,
    run_legacy_inertial_examples,
    static_equivalence_demo,
    verify_raw_record,
    solve_motion,
)
from inertial_pose.synthetic import make_calibration


# --------------------------------------------------------------------------- #
# 验收器整体：必须全过，且不能有子项“未执行”
# --------------------------------------------------------------------------- #
def test_acceptance_report_all_cases_ran_and_passed():
    report = run_acceptance()
    assert report.passed
    assert all(c.ran for c in report.cases), [
        (c.name, c.detail) for c in report.cases if not c.ran]
    assert len(report.cases) >= 10


def test_acceptance_includes_replay_foot_and_knee():
    report = run_acceptance()
    names = {c.name for c in report.cases}
    assert "replay_reads_foot_landing_and_knee" in names
    case = next(c for c in report.cases
                if c.name == "replay_reads_foot_landing_and_knee")
    assert case.passed
    assert case.metrics["knee_max_err_deg"] < 1e-5
    assert case.metrics["knee_range_deg"] > 1.0


def test_acceptance_clean_dir_runner(tmp_path):
    """在净目录子进程执行验收（脚本的 --clean-dir 路径）。"""
    import importlib.util
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "scripts", "run_calibration_acceptance.py")
    spec = importlib.util.spec_from_file_location("run_accept_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.run_in_clean_dir()
    assert result["clean_dir"] is True
    assert result["returncode"] == 0, result["stderr"]


def test_acceptance_report_serializes_with_metrics():
    rep = run_acceptance().to_dict()
    assert rep["passed"] is True
    for case in rep["cases"]:
        assert case["ran"] and case["passed"]
        assert isinstance(case["metrics"], dict)


# --------------------------------------------------------------------------- #
# 静态歧义：装反轴与静偏首帧朝向相同；固定 vs 增长
# --------------------------------------------------------------------------- #
def test_static_ambiguity_identical_at_first_frame():
    demo = static_equivalence_demo(angle_deg=10.0, axis_label="z", duration=4.0)
    assert demo["static_identical"] is True
    assert demo["fixed_variation_deg"] < 1e-9      # 固定错误：相对首帧不变
    assert demo["growing_variation_deg"] > 1.0     # 增长错误：相对首帧显著变化
    assert demo["bias_rate_deg_s"] > 0.0


# --------------------------------------------------------------------------- #
# 区分力门：短暂静置不足以区分；转身/延长静置足够
# --------------------------------------------------------------------------- #
def test_discrimination_rejects_short_static_hold():
    m = generate_analytic_motion(duration=2.0)
    ev = evaluate_discrimination(m, use_truth=True)
    assert ev.decisive is False
    assert ev.mode == "insufficient"


def test_discrimination_accepts_turn_without_truth():
    m = generate_analytic_motion(duration=5.0, turn=(1.5, 1.5))
    ev = evaluate_discrimination(m, use_truth=False)
    assert ev.decisive is True and ev.mode == "turn"
    assert ev.yaw_swing_deg >= 20.0


def test_discrimination_accepts_extended_hold_with_growing_residual():
    from inertial_pose.calibration_acceptance import _static_hold_with_tilt_bias
    m = _static_hold_with_tilt_bias(duration=6.0, bias_deg_s=3.0)
    ev = evaluate_discrimination(m, use_truth=True)
    assert ev.decisive is True and ev.mode == "hold"
    assert ev.residual_linear_deg_s >= 0.5
    # 该判定不依赖航向变化（倾角漂移）
    assert ev.yaw_swing_deg < 20.0


# --------------------------------------------------------------------------- #
# 签核台账：无区分力拒绝；拔掉决定性传感器撤销签核
# --------------------------------------------------------------------------- #
def test_ledger_refuses_nondiscriminating_observation():
    m = generate_analytic_motion(duration=2.0)
    ev = evaluate_discrimination(m, use_truth=True)
    ledger = CalibrationLedger()
    sid = m.placement["l_thigh"]
    commit = ledger.attempt(segments=m.skeleton.segments,
                            sensor_ids=tuple(m.placement.values()),
                            evidence=ev, decisive_sensors={sid},
                            trigger=ev.mode)
    assert commit.signed is False
    assert ledger.signed == []


def test_ledger_revokes_only_decisive_sensor():
    m = generate_analytic_motion(duration=5.0, turn=(1.5, 1.5))
    ev = evaluate_discrimination(m, use_truth=False)
    decisive = m.placement["l_thigh"]
    other = next(s for s in m.placement.values() if s != decisive)
    ledger = CalibrationLedger()
    assert ledger.attempt(segments=m.skeleton.segments,
                          sensor_ids=tuple(m.placement.values()),
                          evidence=ev, decisive_sensors={decisive},
                          trigger=ev.mode).signed
    # 拔非决定性传感器：签核保留
    assert ledger.revoke({other}) == []
    assert ledger.signed
    # 拔决定性传感器：必须撤销
    revoked = ledger.revoke({decisive})
    assert len(revoked) == 1
    assert ledger.signed == []
    assert decisive not in ledger.active_sensors


# --------------------------------------------------------------------------- #
# 三类误差：单位 / 左右手系 / 时标差 必须可辨
# --------------------------------------------------------------------------- #
def test_three_injected_errors_are_classified_correctly():
    motion = generate_known_axis_motion(
        axis_label="z", rate_deg_s=45.0, duration=4.0, offset_deg=5.0)
    inj = inject_three_errors(motion, axis_label="z", rate_deg_s=45.0,
                              time_offset_s=0.2)
    for kind in ("clean", "unit", "handedness", "time_shift"):
        measured = inj.measured_angle_rad(kind)
        label = classify_injected_error(inj.truth_angle_rad, measured,
                                        dt=inj.dt)
        assert label == kind, f"{kind} -> {label}"


def test_handedness_is_sign_flip_and_unit_is_scaled():
    motion = generate_known_axis_motion(duration=4.0, rate_deg_s=45.0)
    inj = inject_three_errors(motion, rate_deg_s=45.0)
    truth = inj.truth_angle_rad
    nz = np.abs(truth) > np.deg2rad(5.0)
    # 手系：符号相反
    assert np.mean(np.sign(inj.measured_angle_rad("handedness")[nz]) !=
                   np.sign(truth[nz])) > 0.9
    # 单位：幅度比约 π/180
    ratio = (np.max(np.abs(inj.measured_angle_rad("unit"))) /
             np.max(np.abs(truth)))
    assert abs(ratio - np.pi / 180.0) < 0.3 * np.pi / 180.0


# --------------------------------------------------------------------------- #
# 独立参照：不依赖被测标定，且在无噪时与被测残差一致
# --------------------------------------------------------------------------- #
def test_independent_reference_matches_engine_when_clean():
    m = generate_analytic_motion(duration=5.0, turn=(1.5, 1.5),
                                 dip=(0.5, 1.5, 0.5))
    cal = make_calibration(m)
    ref = reference_residual_deg(m)
    ref_max = max(float(np.max(v)) for v in ref.values())
    assert ref_max < 1e-6
    assert abs(ref_max - float(np.max(cal.residual_deg))) < 1e-6


def test_independent_reference_reacts_to_static_bias():
    m = generate_analytic_motion(duration=4.0, turn=(1.0, 1.2))
    q_fix = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.deg2rad(8.0))
    biased = {s: np.array([quat.normalize(quat.mul(q_fix, q)) for q in qq])
              for s, qq in m.sensor_q_WS.items()}
    ok = max(float(np.max(v)) for v in reference_residual_deg(m).values())
    bad = max(float(np.max(v)) for v in
              reference_residual_deg(m, q_WS=biased).values())
    assert bad > ok + 1.0


# --------------------------------------------------------------------------- #
# 防伪报：执行证据来自真实求解口径，原始记录可复算
# --------------------------------------------------------------------------- #
def test_raw_record_verifies_joint_and_residual():
    m = generate_analytic_motion(duration=4.0, turn=(1.0, 1.0),
                                 dip=(0.5, 1.5, 0.5))
    cal = make_calibration(m)
    solver, res = solve_motion(m, cal)
    raw = verify_raw_record(m, cal, res, joint="l_knee")
    assert raw["consistent"] is True      # 引擎残差与独立参照一致
    assert raw["joint_max_deg"] < 1e-5
    assert raw["calibration_residual_max_deg"] < 1e-6


def test_evidence_binds_execution_and_raw_record():
    report = run_acceptance()
    m = generate_analytic_motion(duration=4.0, turn=(1.0, 1.0))
    cal = make_calibration(m)
    _, res = solve_motion(m, cal)
    raw = verify_raw_record(m, cal, res)
    ev = evidence_from_disk(report, raw)
    assert ev["passed"] is True

    # 篡改原始记录一致性 -> 整体判失败（不允许伪报）
    tampered = dict(raw)
    tampered["consistent"] = False
    ev_bad = evidence_from_disk(report, tampered)
    assert ev_bad["passed"] is False


def test_raw_record_detects_inconsistent_engine():
    """若被测引擎与独立参照不符（模拟坏实现），核对必须判不一致。"""
    m = generate_analytic_motion(duration=4.0, turn=(1.0, 1.0))
    cal = make_calibration(m)
    _, res = solve_motion(m, cal)
    raw = verify_raw_record(m, cal, res)
    assert raw["consistent"]

    # 人为篡改引擎残差 -> 与独立参照不再一致
    cal.residual_deg = cal.residual_deg + 5.0
    raw_bad = verify_raw_record(m, cal, res)
    assert raw_bad["consistent"] is False


# --------------------------------------------------------------------------- #
# 旧样例回归：必须运行到底（无异常），且结果为零误差
# --------------------------------------------------------------------------- #
def test_legacy_inertial_examples_run_to_completion():
    out = run_legacy_inertial_examples()
    for action in ("walk", "squat", "arm_raise"):
        assert out[action]["ran"] is True, out[action].get("error")
        assert out[action]["joint_mean_deg"] < 1e-6
        assert out[action]["cal_residual_max_deg"] < 1e-6
        assert out[action]["frames"] > 0


# --------------------------------------------------------------------------- #
# 捕捉视窗：签核门来自可观测量 + 异常标注 + 回归样例可打开
# --------------------------------------------------------------------------- #
def test_capture_payload_has_calibration_gate():
    p = capture.build_capture_payload({"action": "walk", "duration": 8.0,
                                       "seed": 2, "drift_rate": 0.12})
    assert p["ok"] is True
    gate = p["calibration_gate"]
    assert gate["signed"] is True          # 漂移引入显著航向跨度 -> 可区分
    assert gate["trigger"] == "turn"
    assert "anomaly_flags" in p


def test_capture_short_observation_gate_refuses():
    # 短时深蹲、无转身：无法区分两种解释 -> 拒绝签核
    p = capture.build_capture_payload({"action": "squat", "duration": 1.0,
                                       "seed": 5})
    assert p["calibration_gate"]["signed"] is False
    assert p["calibration_gate"]["trigger"] == "insufficient"


def test_capture_regression_samples_exposed_via_options():
    from inertial_pose.calibration_acceptance import regression_samples
    samples = regression_samples()
    assert len(samples) >= 4
    for s in samples:
        assert {"action", "duration", "seed"} <= set(s)
        # 每个样例都能真的跑起来
        params = {k: v for k, v in s.items()
                  if k in ("action", "duration", "seed", "noise_deg",
                           "drift_rate")}
        if s.get("axis_remap"):
            params["axis_remap"] = json.dumps(s["axis_remap"])
        p = capture.build_capture_payload(params)
        assert p["ok"] is True


def test_anomaly_flags_mark_joint_and_residual():
    from inertial_pose.calibration_acceptance import anomaly_flags
    # 正常 -> 无异常
    good = {"ok": True, "joint_error_deg": {"j": {"mean": 0.0, "p95": 0.0}},
            "residual": {"max_deg": [0.0]},
            "foot_slide_total_m": {"l_foot": 0.0, "r_foot": 0.0}}
    assert anomaly_flags(good) == []
    # 关节异常 + 残差异常都要标出
    bad = {"ok": True, "joint_error_deg": {"j": {"mean": 20.0, "p95": 30.0}},
           "residual": {"max_deg": [25.0]},
           "foot_slide_total_m": {"l_foot": 0.0, "r_foot": 0.0}}
    flags = anomaly_flags(bad)
    assert any("角色姿态" in f for f in flags)
    assert any("惯性偏差" in f for f in flags)
    # 链路失败
    assert anomaly_flags({"ok": False, "error": "boom"})
