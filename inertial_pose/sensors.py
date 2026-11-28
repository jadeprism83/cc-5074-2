"""传感器原始数据支持。

支持两类输入：

- **方向数据**：设备/算法已经给出的世界四元数 ``q_WS``（磁力/滤波后）。
- **原始数据**：陀螺仪角速度 + 加速度计，用 Mahony 互补滤波积分出
  ``q_WS``。航向由初值确定（无磁力时航向可观性弱，这一限制显式保留，
  由 :mod:`inertial_pose.calibration` 的世界参考项处理）。

观测容器 :class:`SensorFrame` / :class:`SensorSequence` 只描述传感器
框架 S 与导航世界 W 之间的量，不包含任何骨架信息。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quat


@dataclass
class SensorFrame:
    """单帧多传感器观测。"""

    t: float
    q_WS: dict[int, np.ndarray]

    def select(self, sensor_ids: tuple[int, ...]) -> "SensorFrame":
        return SensorFrame(self.t, {s: self.q_WS[s] for s in sensor_ids if s in self.q_WS})


@dataclass
class SensorSequence:
    """整段观测序列。"""

    dt: float
    frames: list[SensorFrame]

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def sensor_ids(self) -> tuple[int, ...]:
        ids: set[int] = set()
        for f in self.frames:
            ids.update(f.q_WS.keys())
        return tuple(sorted(ids))

    def quaternion_array(self, sensor_id: int) -> np.ndarray:
        """取单个传感器的 (N,4) 朝向序列，缺帧抛错。"""
        return np.stack([f.q_WS[sensor_id] for f in self.frames])


# --------------------------------------------------------------------------- #
# Mahony 互补滤波（原始数据 -> q_WS）
# --------------------------------------------------------------------------- #
class MahonyFilter:
    """6 轴 Mahony 滤波，输出导航世界（Z 向上）到传感器的旋转 q_WS。

    参数
    ----
    kp, ki       比例/积分增益（1/s 量纲）
    dt           采样周期（秒）
    q0           初始 q_WS；缺省为单位旋转（传感器与世界对齐）
    """

    def __init__(self, dt: float, kp: float = 2.0, ki: float = 0.05,
                 q0: np.ndarray | None = None):
        self.dt = float(dt)
        self.kp = float(kp)
        self.ki = float(ki)
        self.q = quat.identity() if q0 is None else quat.normalize(q0)
        self._bias = np.zeros(3)

    def update(self, gyro: np.ndarray, accel: np.ndarray) -> np.ndarray:
        """推进一帧。

        gyro  传感器框架角速度 (rad/s, 3,)
        accel 传感器框架比力 (m/s^2, 3,)；静止时指向 -重力方向
        返回更新后的 q_WS。
        """
        gyro = np.asarray(gyro, dtype=float)
        accel = np.asarray(accel, dtype=float)
        q = self.q

        # ---- 加速度计修正：世界重力 (0,0,-g) 旋到传感器框架 ----------
        a_norm = np.linalg.norm(accel)
        if a_norm > 1.0e-6:
            v = quat.rotate_vectors(quat.inv(q), np.array([0.0, 0.0, -1.0]))
            a = accel / a_norm
            e = np.cross(a, v)  # 传感器框架下的误差
            self._bias += self.ki * e * self.dt
            gyro_corrected = gyro + self._bias + self.kp * e
        else:
            gyro_corrected = gyro + self._bias

        # ---- 角速度积分（世界系四元数导数 0.5 q ⊗ ω_S） -------------
        dq = quat.mul(q, np.array([0.0, *gyro_corrected])) * (0.5 * self.dt)
        self.q = quat.normalize(q + dq)
        return self.q.copy()

    @staticmethod
    def filter_sequence(gyro: np.ndarray, accel: np.ndarray, dt: float,
                        kp: float = 2.0, ki: float = 0.05,
                        q0: np.ndarray | None = None) -> np.ndarray:
        """整段序列便捷接口，输入 (N,3)/(N,3)，返回 (N,4) q_WS。"""
        flt = MahonyFilter(dt, kp, ki, q0)
        out = np.zeros((len(gyro), 4))
        for n in range(len(gyro)):
            out[n] = flt.update(gyro[n], accel[n])
        return out


def sequence_from_orientations(q_WS: dict[int, np.ndarray], dt: float) -> SensorSequence:
    """把各传感器的 (N,4) 朝向数组打包成观测序列。"""
    n = {len(q) for q in q_WS.values()}
    if len(n) != 1:
        raise ValueError(f"各传感器帧数不一致: {n}")
    n_frames = n.pop()
    frames = [
        SensorFrame(t=k * dt, q_WS={sid: quat.normalize(q[k]) for sid, q in q_WS.items()})
        for k in range(n_frames)
    ]
    return SensorSequence(dt=dt, frames=frames)
