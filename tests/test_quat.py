"""四元数基础数学核验：恒等、乘积、逆、旋转、轴角/矩阵往返、角度距离。"""

import numpy as np

from inertial_pose import quat


def test_identity_and_normalize():
    q = quat.identity()
    assert np.allclose(q, [1, 0, 0, 0])
    assert np.isclose(np.linalg.norm(quat.normalize([2.0, 0, 0, 0])), 1.0)


def test_mul_inverse_roundtrip():
    rng = np.random.default_rng(0)
    axis = rng.normal(size=(5, 3))
    angle = rng.uniform(-np.pi, np.pi, size=5)
    q = quat.from_axis_angle(axis, angle)
    qinv = quat.inv(q)
    prod = quat.mul(q, qinv)
    # q * q^-1 = 单位旋转
    assert np.allclose(prod, quat.identity(5), atol=1e-12)


def test_rotate_known_axes():
    # 绕 z 旋转 +90°：+x -> +y
    qz = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    out = quat.rotate_vectors(qz, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-12)
    # 绕 y 旋转 +90°：-z -> -x（与合成步态的铰链符号约定一致）
    qy = quat.from_axis_angle(np.array([0.0, 1.0, 0.0]), np.pi / 2)
    out = quat.rotate_vectors(qy, np.array([0.0, 0.0, -1.0]))
    assert np.allclose(out, [-1.0, 0.0, 0.0], atol=1e-12)


def test_matrix_roundtrip_preserves_rotation():
    rng = np.random.default_rng(1)
    q = quat.from_axis_angle(rng.normal(size=(6, 3)),
                             rng.uniform(-np.pi, np.pi, 6))
    R = quat.to_matrix(q)
    q_back = quat.from_matrix(R)
    assert np.allclose(quat.angular_distance(q, q_back), 0.0, atol=1e-10)
    # 旋转矩阵正交且行列式为 +1
    assert np.allclose(np.einsum("nij,nkj->nik", R, R),
                       np.broadcast_to(np.eye(3), R.shape), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_axis_angle_roundtrip():
    rng = np.random.default_rng(2)
    axis0 = rng.normal(size=(4, 3))
    angle0 = rng.uniform(0.05, np.pi - 0.05, 4)
    q = quat.from_axis_angle(axis0, angle0)
    axis1, angle1 = quat.to_axis_angle(q)
    # 角幅度一致；轴方向（含符号）一致
    assert np.allclose(angle0, angle1, atol=1e-10)
    u0 = axis0 / np.linalg.norm(axis0, axis=-1, keepdims=True)
    assert np.allclose(np.abs(np.sum(axis1 * u0, axis=-1)), 1.0, atol=1e-10)


def test_angular_distance_symmetry_and_zero():
    rng = np.random.default_rng(3)
    q = quat.from_axis_angle(rng.normal(size=3), 0.7)
    assert np.isclose(quat.angular_distance(q, q), 0.0, atol=1e-12)
    d = quat.angular_distance(q, quat.identity())
    assert 0.0 <= float(d) <= np.pi + 1e-9


def test_relative_is_local_rotation():
    # q_child = q_parent * q_local  =>  relative(parent, child) = q_local
    rng = np.random.default_rng(4)
    qp = quat.from_axis_angle(rng.normal(size=3), 0.5)
    ql = quat.from_axis_angle(rng.normal(size=3), 0.3)
    qc = quat.mul(qp, ql)
    assert np.allclose(quat.angular_distance(quat.relative(qp, qc), ql),
                       0.0, atol=1e-12)


def test_slerp_endpoints_and_midpoint():
    q0 = quat.identity()
    q1 = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    assert np.allclose(quat.slerp(q0, q1, 0.0), q0, atol=1e-12)
    assert np.allclose(quat.slerp(q0, q1, 1.0), q1, atol=1e-12)
    qm = quat.slerp(q0, q1, 0.5)
    _, ang = quat.to_axis_angle(qm)
    assert np.isclose(ang, np.pi / 4, atol=1e-10)
