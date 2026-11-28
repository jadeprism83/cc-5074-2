"""Mahony 互补滤波核验：原始陀螺+加速度 -> 世界朝向 q_WS。"""

import numpy as np

from inertial_pose import quat
from inertial_pose.sensors import MahonyFilter


def test_static_gravity_keeps_level_orientation():
    # 静止：陀螺为 0，加速度仅感受反向重力（传感器框架 +z 朝上时比力沿 +z）
    dt = 0.01
    flt = MahonyFilter(dt, kp=2.0, ki=0.05)
    accel = np.array([0.0, 0.0, 1.0])
    gyro = np.zeros(3)
    q_last = flt.q.copy()
    for _ in range(400):
        q_last = flt.update(gyro, accel)
    # 长时间静止后应收敛回单位（水平）朝向
    assert np.isclose(quat.angular_distance(q_last, quat.identity()),
                      0.0, atol=1e-3)


def test_pure_rotation_tracks_known_yaw():
    # 绕传感器 z 恒定角速度积分，无加速度扰动时航向应近似跟随积分角
    dt = 0.01
    rate = 0.5  # rad/s
    flt = MahonyFilter(dt, kp=0.0, ki=0.0)  # 关闭加速度修正以纯积分
    accel = np.array([0.0, 0.0, 1.0])
    n = 200
    for _ in range(n):
        flt.update(np.array([0.0, 0.0, rate]), accel)
    R = quat.to_matrix(flt.q)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    assert np.isclose(yaw, rate * dt * n, rtol=0.02)


def test_filter_sequence_shape():
    n = 50
    gyro = np.zeros((n, 3))
    accel = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    q = MahonyFilter.filter_sequence(gyro, accel, dt=0.01)
    assert q.shape == (n, 4)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)


def test_mahony_heading_observability_limitation_is_respected():
    # 6 轴滤波的航向完全由初值决定：相同运动、不同初值 -> 仅差一个恒定偏航，
    # 倾斜一致。这正是需要外部世界参考（calibration.world_yaw）的原因。
    dt = 0.01
    n = 100
    gyro = np.tile(np.array([0.0, 0.0, 0.2]), (n, 1))
    accel = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    q0_b = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.7)
    qa = MahonyFilter.filter_sequence(gyro, accel, dt, kp=0.0, ki=0.0)
    qb = MahonyFilter.filter_sequence(gyro, accel, dt, kp=0.0, ki=0.0, q0=q0_b)
    # 末态之间相差恒定偏航 0.7
    rel = quat.relative(qa[-1], qb[-1])
    _, ang = quat.to_axis_angle(rel)
    assert np.isclose(ang, 0.7, atol=1e-6)
