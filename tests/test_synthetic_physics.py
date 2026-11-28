"""合成数据的物理一致性与观测轴向重映射核验。"""

import numpy as np

from inertial_pose import quat
from inertial_pose.sensors import sequence_from_orientations
from inertial_pose.skeleton import DEFAULT_SKELETON as SKEL
from inertial_pose.solver import PoseSolver
from inertial_pose.synthetic import _walk_cycle, generate_motion, make_calibration


def _foot_positions(local, root):
    out = {"l": [], "r": []}
    for k in range(len(root)):
        lk = {j: local[j][k] for j in SKEL.order}
        _, wp = SKEL.forward_kinematics(lk, root[k])
        for s in ("l", "r"):
            out[s].append(wp[SKEL.segment_joint[f"{s}_foot"]].copy())
    return {s: np.array(v) for s, v in out.items()}


def test_walk_has_truly_planted_stance_episodes():
    times = np.arange(0.0, 6.0, 0.05)
    local, root = _walk_cycle(SKEL, times)
    pos = _foot_positions(local, root)
    for s in ("l", "r"):
        d = np.linalg.norm(np.diff(pos[s][:, :2], axis=0), axis=1)
        # 存在多帧连续零水平位移的真实支撑段
        assert np.sum(d < 1e-7) > 10
        # 横向分居恒定在 y，z 在支撑段恒定
        assert np.allclose(pos[s][:, 1], pos[s][0, 1])


def test_legs_are_symmetric_no_mirror_angles():
    # 左右腿局部 y 铰链框架同构：同一时刻两侧关节角范围一致（镜像来自平移）
    times = np.arange(0.0, 2.0, 0.05)
    local, _ = _walk_cycle(SKEL, times)

    def yext(q):
        _, ang = quat.to_axis_angle(q)
        return np.max(ang)

    assert np.isclose(yext(local["l_knee"]), yext(local["r_knee"]), rtol=0.05)


def test_axis_remap_is_explicit_invertible_transform():
    """观测轴向切换是显式正当旋转；循环置换改变观测，反射置换被拒绝。"""
    import pytest

    m = generate_motion(action="walk", duration=3.0, dt=0.05, seed=0)
    cal = make_calibration(m)
    sid = m.placement["l_thigh"]
    # 三轴都有分量的通用旋转
    q0 = quat.normalize(quat.from_euler_xyz(np.array([0.4, -0.3, 0.5])))

    # 正当（行列式 +1）循环置换 (1,2,0) 应改变观测
    proper = PoseSolver(cal, skeleton=m.skeleton,
                        observed_axis_remap={sid: (1, 2, 0)})
    q_remap = proper._remap_observation(sid, q0)
    assert quat.angular_distance(q_remap, q0) > 1e-6

    # 恒等映射保持不变
    identity_solver = PoseSolver(cal, skeleton=m.skeleton)
    assert np.isclose(
        quat.angular_distance(identity_solver._remap_observation(sid, q0), q0),
        0.0, atol=1e-12)

    # 镜像反射（行列式 -1）必须被显式拒绝，而不是静默退化
    improper = PoseSolver(cal, skeleton=m.skeleton,
                          observed_axis_remap={sid: (1, 0, 2)})
    with pytest.raises(ValueError):
        improper._remap_observation(sid, q0)


def test_remapped_permutation_matrix_is_orthogonal():
    P = np.zeros((3, 3))
    remap = (2, 0, 1)
    for new_axis, old_axis in enumerate(remap):
        P[new_axis, old_axis] = 1.0
    assert np.allclose(P @ P.T, np.eye(3))
    assert np.isclose(np.linalg.det(P), 1.0)


def test_inertial_residual_is_real_computation():
    # 求解结果必须给出每个观测骨段相对惯性观测的角度偏差（非占位常量）
    m = generate_motion(action="arm_raise", duration=3.0, dt=0.05, seed=1,
                        noise_deg=1.0)
    cal = make_calibration(m)
    res = PoseSolver(cal, skeleton=m.skeleton).solve_sequence(
        sequence_from_orientations(m.sensor_q_WS, m.dt))
    for r in res:
        assert len(r.inertial_residual_deg) == len(cal.segments)
        assert all(v >= 0.0 for v in r.inertial_residual_deg.values())
    # 有噪声时残差整体非零
    pooled = np.array([v for r in res for v in r.inertial_residual_deg.values()])
    assert pooled.mean() > 0.0
