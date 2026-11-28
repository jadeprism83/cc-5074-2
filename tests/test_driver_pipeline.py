"""端到端驱动核验：传感器观测 -> 标定 -> 求解器 -> 角色姿态。

求解器的唯一输入是合成传感器观测 q_WS；真值骨骼旋转仅用于最终指标对照，
不存在任何把真值绕过标定直接赋给骨骼的路径。
"""

import numpy as np

from inertial_pose.metrics import summarize
from inertial_pose.sensors import sequence_from_orientations
from inertial_pose.solver import PoseSolver
from inertial_pose.synthetic import (
    generate_motion,
    make_calibration,
    random_mount_rotations,
)


def _solve(action, seed=0, noise=0.0, duration=5.0, config=None):
    rng = np.random.default_rng(seed)
    mounts = random_mount_rotations(None, rng)
    m = generate_motion(
        action=action, duration=duration, dt=0.05,
        mount_q_BS=mounts, world_yaw={sid: 0.25 for sid in mounts},
        noise_deg=noise, seed=seed,
    )
    cal = make_calibration(m)
    seq = sequence_from_orientations(m.sensor_q_WS, m.dt)
    res = PoseSolver(cal, skeleton=m.skeleton, config=config).solve_sequence(seq)
    return m, cal, res


def test_noiseless_joint_angles_recover_all_actions():
    for action in ("walk", "squat", "arm_raise"):
        m, cal, res = _solve(action, seed=2)
        s = summarize(m.truth_local_rot, m.truth_segment_rot,
                      m.truth_root_pos, m.skeleton, res, cal)
        assert max(s["joint_p95_deg"].values()) < 1e-6
        assert max(s["segment_p95_deg"].values()) < 1e-6


def test_noiseless_calibration_residual_zero():
    m, cal, _ = _solve("walk", seed=4)
    assert np.max(cal.residual_deg) < 1e-6


def test_noise_produces_small_but_bounded_error():
    m, cal, res = _solve("walk", seed=6, noise=1.5)
    s = summarize(m.truth_local_rot, m.truth_segment_rot,
                  m.truth_root_pos, m.skeleton, res, cal)
    mean_err = np.mean(list(s["joint_mean_deg"].values()))
    assert 0.0 < mean_err < 5.0
    assert max(s["joint_p95_deg"].values()) < 12.0


def test_solver_inputs_are_only_sensor_observations():
    """结构性保证：PoseSolver 不持有真值骨段旋转。"""
    m, cal, res = _solve("squat", seed=1)
    solver_attrs = {k for k in vars(PoseSolver(cal, skeleton=m.skeleton)).keys()}
    assert not any("truth" in k.lower() for k in solver_attrs)
    # 结果里的骨段朝向来自标定 apply，而非真值拷贝
    assert set(res[0].segment_rot.keys()) == set(cal.segments)


def test_walk_foot_lock_zero_slide():
    m, cal, res = _solve("walk", seed=3, duration=6.0)
    s = summarize(m.truth_local_rot, m.truth_segment_rot,
                  m.truth_root_pos, m.skeleton, res, cal)
    assert sum(s["foot_slide_total_m"].values()) < 1e-6


def test_squat_foot_lock_zero_slide():
    m, cal, res = _solve("squat", seed=5, duration=6.0)
    s = summarize(m.truth_local_rot, m.truth_segment_rot,
                  m.truth_root_pos, m.skeleton, res, cal)
    assert sum(s["foot_slide_total_m"].values()) < 1e-6


def test_root_position_is_explicit_prior_not_claimed_observed():
    # 无锚点动作（举臂）根位置回到固定先验，且不抛错——平移不靠方向唯一恢复
    m, cal, res = _solve("arm_raise", seed=7)
    roots = np.stack([r.root_pos for r in res])
    assert np.allclose(roots[:, 2], 0.95, atol=1e-9)
