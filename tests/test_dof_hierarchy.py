"""关节自由度投影、角限位与父子层级约束核验。"""

import numpy as np

from inertial_pose import quat
from inertial_pose.skeleton import DEFAULT_SKELETON as SKEL
from inertial_pose.solver import project_to_dof


def test_hinge_knee_only_y_dof():
    # 给膝关节一个含 x/z 分量的任意旋转，投影后应只剩绕局部 y 的旋转
    q = quat.from_euler_xyz(np.array([0.5, 0.7, -0.4]))
    proj = project_to_dof(SKEL, "l_knee", q)
    axis, angle = quat.to_axis_angle(proj)
    if angle > 1e-9:
        # 旋转轴必须与 y 轴平行
        assert np.isclose(abs(float(axis[1])), 1.0, atol=1e-9)
        assert np.isclose(float(axis[0]), 0.0, atol=1e-9)
        assert np.isclose(float(axis[2]), 0.0, atol=1e-9)


def test_angle_limit_is_clipped():
    lo, hi = SKEL.joints["l_knee"].angle_limit[0]
    # 构造一个远超上限/下限的 y 轴旋转
    q_over = quat.from_axis_angle(np.array([0.0, 1.0, 0.0]), hi + 1.5)
    q_under = quat.from_axis_angle(np.array([0.0, 1.0, 0.0]), lo - 1.5)
    for q, bound in ((q_over, hi), (q_under, lo)):
        proj = project_to_dof(SKEL, "l_knee", q)
        _, ang = quat.to_axis_angle(proj)
        # 投影后的有符号 y 分量应被夹在限位内
        signed = proj  # 直接比对轴角符号值
        axis, a = quat.to_axis_angle(signed)
        signed_angle = float(np.sign(axis[1] if abs(axis[1]) > 1e-9 else 1.0) * a)
        assert lo - 1e-9 <= signed_angle <= hi + 1e-9


def test_unobserved_joint_uses_identity_prior():
    # 未观测关节（构造一个不在输入中的情形）投影任意旋转仍受 DOF 约束，
    # 而求解器对完全未观测关节使用单位先验，这里核验单位旋转投影为单位。
    out = project_to_dof(SKEL, "l_knee", quat.identity())
    assert np.allclose(out, quat.identity(), atol=1e-12)


def test_forward_kinematics_respects_parent_child():
    # 子关节世界旋转 = 父世界旋转 * 局部旋转，绝不直接等于输入四元数
    local = {n: quat.identity() for n in SKEL.order}
    local["spine"] = quat.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.3)
    local["neck"] = quat.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.2)
    world, _ = SKEL.forward_kinematics(local, np.array([0.0, 0.0, 0.95]))
    expected_neck = quat.mul(world["spine"], local["neck"])
    assert np.isclose(
        quat.angular_distance(world["neck"], expected_neck), 0.0, atol=1e-12)
    # neck 世界角是两级累积，而不是它自己的局部角
    _, neck_world_angle = quat.to_axis_angle(world["neck"])
    assert np.isclose(neck_world_angle, 0.5, atol=1e-9)


def test_rest_offsets_define_lateral_separation_on_y():
    # 侧面分居沿 y（与矢状面铰链 y 不冲突），前向为 x，Z 向上
    assert SKEL.joints["l_hip"].rest_offset[1] > 0
    assert SKEL.joints["r_hip"].rest_offset[1] < 0
    assert SKEL.joints["l_hip"].rest_offset[0] == 0.0
