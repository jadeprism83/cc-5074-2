"""四元数与旋转数学。

约定：四元数为 ``[w, x, y, z]``，表示**主动旋转**，把向量从子框架
旋转到父框架（例如 q_WS 把传感器框架 S 中的向量旋转到世界框架 W）。

所有函数对 numpy 广播友好：批量四元数形状为 ``(..., 4)``，
批量旋转矩阵形状为 ``(..., 3, 3)``。
"""

from __future__ import annotations

import numpy as np

_EPS = 1.0e-12


# --------------------------------------------------------------------------- #
# 基础运算
# --------------------------------------------------------------------------- #
def normalize(q: np.ndarray) -> np.ndarray:
    """归一化四元数，输入 (...,4)。"""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, _EPS)
    return q / n


def mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton 乘积 q1 * q2，支持广播。"""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def conj(q: np.ndarray) -> np.ndarray:
    """四元数共轭（逆旋转，假设单位四元数）。"""
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def inv(q: np.ndarray) -> np.ndarray:
    """单位四元数的逆，等价于共轭后归一化。"""
    return normalize(conj(q))


def rotate_vectors(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数 q 旋转向量 v：v' = q v q*。"""
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    w = q[..., 0:1]
    u = q[..., 1:4]
    # v' = 2 (u·v) u + (2w^2 - 1) v + 2w (u × v)
    uv = np.sum(u * v, axis=-1, keepdims=True)
    cross = np.cross(u, v)
    return 2.0 * uv * u + (2.0 * w * w - 1.0) * v + 2.0 * w * cross


def identity(n: int | None = None) -> np.ndarray:
    """单位四元数；n 给定时返回 (n,4) 批量。"""
    if n is None:
        return np.array([1.0, 0.0, 0.0, 0.0])
    out = np.zeros((n, 4))
    out[:, 0] = 1.0
    return out


# --------------------------------------------------------------------------- #
# 与旋转矩阵 / 轴角 / 欧拉角互转
# --------------------------------------------------------------------------- #
def from_matrix(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 (...,3,3) -> 四元数 (...,4)，Shepperd 稳定分支。"""
    R = np.asarray(R, dtype=float)
    batch = R.shape[:-2]
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros(batch + (4,))

    c0 = tr > 0
    s = 2.0 * np.sqrt(np.maximum(1.0 + tr, _EPS))
    q[c0] = np.stack(
        [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], axis=-1
    )[c0]

    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    s = 2.0 * np.sqrt(np.maximum(1.0 + m00 - m11 - m22, _EPS))
    q[c1] = np.stack(
        [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s], axis=-1
    )[c1]

    c2 = (~c0) & (~c1) & (m11 > m22)
    s = 2.0 * np.sqrt(np.maximum(1.0 + m11 - m00 - m22, _EPS))
    q[c2] = np.stack(
        [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s], axis=-1
    )[c2]

    c3 = (~c0) & (~c1) & (~c2)
    s = 2.0 * np.sqrt(np.maximum(1.0 + m22 - m00 - m11, _EPS))
    q[c3] = np.stack(
        [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s], axis=-1
    )[c3]
    return normalize(q)


def to_matrix(q: np.ndarray) -> np.ndarray:
    """四元数 (...,4) -> 旋转矩阵 (...,3,3)。"""
    q = normalize(np.asarray(q, dtype=float))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """轴角 -> 四元数，axis 无需预归一化。"""
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)
    n = np.linalg.norm(axis, axis=-1, keepdims=True)
    n = np.maximum(n, _EPS)
    u = axis / n
    half = angle[..., None] * 0.5
    s = np.sin(half)
    return np.concatenate([np.cos(half), u * s], axis=-1)


def to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """四元数 -> (单位轴, 角[0, pi])，单位四元数返回零轴零角。"""
    q = normalize(np.asarray(q, dtype=float))
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(np.abs(w))
    s = np.sqrt(np.maximum(1.0 - w * w, _EPS))
    axis = q[..., 1:4] / s[..., None]
    # 负 w 时翻转轴，保持角度在 [0, pi]
    axis = np.where(w[..., None] < 0, -axis, axis)
    zero = s < 1.0e-8
    axis = np.where(zero[..., None], 0.0, axis)
    angle = np.where(zero, 0.0, angle)
    return axis, angle


def from_euler_xyz(angles: np.ndarray) -> np.ndarray:
    """内在 XYZ（roll-pitch-yaw）欧拉角 -> 四元数，angles (...,3) 弧度。"""
    a = np.asarray(angles, dtype=float) * 0.5
    cx, cy, cz = np.cos(a[..., 0]), np.cos(a[..., 1]), np.cos(a[..., 2])
    sx, sy, sz = np.sin(a[..., 0]), np.sin(a[..., 1]), np.sin(a[..., 2])
    return np.stack(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        axis=-1,
    )


def yaw_only(q: np.ndarray) -> np.ndarray:
    """提取绕世界 Z 轴的航向分量（投影到偏航）。"""
    R = to_matrix(q)
    # 航向 = atan2(R10, R00)
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return from_axis_angle(np.broadcast_to(np.array([0.0, 0.0, 1.0]), yaw.shape + (3,)), yaw)


def relative(q_parent: np.ndarray, q_child: np.ndarray) -> np.ndarray:
    """父/子世界朝向 -> 子在父框架下的局部旋转 q_pc = q_pw * q_wc。"""
    return mul(inv(q_parent), q_child)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """球面线性插值，t 为标量或与四元数批次对齐的数组。"""
    q0 = normalize(np.asarray(q0, dtype=float))
    q1 = normalize(np.asarray(q1, dtype=float))
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    t = np.asarray(t, dtype=float)
    small = theta[..., 0] < 1.0e-6
    s0 = np.sin((1.0 - t) * theta) / np.maximum(np.sin(theta), _EPS)
    s1 = np.sin(t * theta) / np.maximum(np.sin(theta), _EPS)
    out = s0 * q0 + s1 * q1
    lin = (1.0 - t)[..., None] * q0 + t[..., None] * q1
    return normalize(np.where(small[..., None], lin, out))


# --------------------------------------------------------------------------- #
# 误差度量
# --------------------------------------------------------------------------- #
def angular_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """两个朝向之间的最短旋转角（弧度），形状 (...)。"""
    qd = mul(inv(q1), q2)
    qd = normalize(qd)
    w = np.clip(np.abs(qd[..., 0]), 0.0, 1.0)
    return 2.0 * np.arccos(w)


def decompose_twist_swing(q: np.ndarray, twist_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把旋转 q 分解为绕给定轴的扭转 (twist) 与剩余摆动 (swing)。

    用于铰链/单自由度关节：twist 是允许分量，swing 是需要被约束掉的分量。
    """
    q = normalize(np.asarray(q, dtype=float))
    axis = np.asarray(twist_axis, dtype=float)
    axis = axis / np.maximum(np.linalg.norm(axis, keepdims=True), _EPS)
    proj = np.sum(q[..., 1:4] * axis, axis=-1, keepdims=True) * axis
    twist = normalize(np.concatenate([q[..., 0:1], proj], axis=-1))
    swing = mul(q, inv(twist))
    return twist, swing
