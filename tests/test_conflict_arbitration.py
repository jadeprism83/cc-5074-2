"""惯导航向与足锁冲突的仲裁核验：修正航向 vs 释放锚点。"""

from collections import Counter

import numpy as np

from inertial_pose.sensors import sequence_from_orientations
from inertial_pose.solver import PoseSolver, SolverConfig
from inertial_pose.synthetic import (
    apply_online_yaw_drift,
    generate_motion,
    make_calibration,
)


def _drifted(seed=2, rate=0.12, duration=8.0):
    m = generate_motion(action="walk", duration=duration, dt=0.05, seed=seed)
    cal = make_calibration(m)
    drift = apply_online_yaw_drift(m, rate)
    return m, cal, sequence_from_orientations(drift, m.dt)


def test_clean_baseline_has_no_conflict():
    m = generate_motion(action="walk", duration=6.0, dt=0.05, seed=2)
    cal = make_calibration(m)
    res = PoseSolver(cal, skeleton=m.skeleton).solve_sequence(
        sequence_from_orientations(m.sensor_q_WS, m.dt))
    assert all(r.conflict is None for r in res)


def test_drift_triggers_conflicts():
    m, cal, seq = _drifted()
    res = PoseSolver(cal, skeleton=m.skeleton).solve_sequence(seq)
    conflicts = [r.conflict for r in res if r.conflict is not None]
    assert len(conflicts) > 0
    for c in conflicts:
        assert c["choice"] in ("correct_heading", "release_anchor")
        assert c["foot_heading_err_deg"] >= 25.0 - 1e-6
        assert c["cost_heading"] >= 0.0 and c["cost_release"] >= 0.0


def test_weights_switch_strategy_direction():
    m, cal, seq = _drifted()
    # 强“保锚点/修正航向”偏好：释放代价权重高、航向代价权重低
    pro_heading = PoseSolver(
        cal, skeleton=m.skeleton,
        config=SolverConfig(slide_weight=2.0, jump_weight=0.2)).solve_sequence(seq)
    # 强“释放锚点”偏好：航向代价权重高、释放代价权重低
    pro_release = PoseSolver(
        cal, skeleton=m.skeleton,
        config=SolverConfig(slide_weight=0.01, jump_weight=5.0)).solve_sequence(seq)

    ch_heading = Counter(r.conflict["choice"]
                         for r in pro_heading if r.conflict)
    ch_release = Counter(r.conflict["choice"]
                         for r in pro_release if r.conflict)

    assert ch_heading["correct_heading"] > ch_heading["release_anchor"]
    assert ch_release["release_anchor"] > ch_release["correct_heading"]


def test_conflict_record_carries_hypothetical_slide():
    m, cal, seq = _drifted()
    res = PoseSolver(
        cal, skeleton=m.skeleton,
        config=SolverConfig(slide_weight=0.01, jump_weight=5.0)).solve_sequence(seq)
    rel = [r.conflict for r in res
           if r.conflict and r.conflict["choice"] == "release_anchor"]
    assert rel, "应至少出现一次释放锚点决策"
    assert all(c["slide_if_release_m"] > 0.0 for c in rel)


def test_persistent_heading_accumulates_drift():
    """场景航向跨落脚持久，漂移应让足部航向偏差随时间增大。"""
    m, cal, seq = _drifted(rate=0.15, duration=8.0)
    solver = PoseSolver(cal, skeleton=m.skeleton)
    errs = []
    for frame in seq.frames:
        r = solver.solve_frame(frame.t, frame.q_WS)
        for seg in ("l_foot", "r_foot"):
            if seg in solver._anchor_yaw:
                errs.append(solver._foot_heading_conflict(r.world_rot, seg))
    assert np.max(errs) > np.deg2rad(25.0)
