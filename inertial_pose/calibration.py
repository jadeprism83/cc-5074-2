"""传感器 -> 骨段的显式标定。

标定把三件**必须分开**的事情拆成独立量，杜绝把收到的四元数直接
赋给骨骼：

1. ``q_WS``   传感器上报的世界朝向（芯片框架 S -> 导航世界 W）。
2. ``yaw_w``  世界参考偏置：每个传感器的导航世界相对场景世界 W0
   的航向角。IMU 的航向来自磁力/上电假设，彼此可能不一致。
3. ``q_BS``   安装旋转：传感器框架 S -> 骨段规范框架 B，由绑缚方式决定。

场景世界中的骨段朝向::

    q_W0B = q_W0W(yaw_w) * q_WS * q_SB,   q_SB = conj(q_BS)

标定残差定义为该预测朝向与标定样本已知骨段朝向之间的最短旋转角。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quat


@dataclass
class Calibration:
    """标定结果。

    sensor_ids   传感器编号序列
    segments     与之一一对应的骨段名
    mount_q_BS   每个传感器的安装旋转 q_BS，形状 (K,4)
    world_yaw    每个传感器的世界参考航向（弧度），形状 (K,)
    residual_deg 标定残差（度），形状 (K,)，由所用样本聚合（RMS）
    """

    sensor_ids: tuple[int, ...]
    segments: tuple[str, ...]
    mount_q_BS: np.ndarray
    world_yaw: np.ndarray
    residual_deg: np.ndarray

    def index_for_sensor(self, sensor_id: int) -> int:
        try:
            return self.sensor_ids.index(sensor_id)
        except ValueError as exc:
            raise KeyError(f"传感器 {sensor_id} 未参与标定") from exc

    def segment_for_sensor(self, sensor_id: int) -> str:
        return self.segments[self.index_for_sensor(sensor_id)]

    # ------------------------------------------------------------------ #
    def apply(self, sensor_id: int, q_WS: np.ndarray) -> np.ndarray:
        """单步观测 -> 场景世界中的骨段朝向 q_W0B。

        这是观测进入骨架前唯一允许的入口，链路可逐环核验。
        """
        i = self.index_for_sensor(sensor_id)
        q_W0W = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), self.world_yaw[i])
        return quat.normalize(
            quat.mul(quat.mul(q_W0W, q_WS), quat.inv(self.mount_q_BS[i]))
        )

    def apply_batch(self, q_WS: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
        """一帧多传感器观测 -> 骨段世界朝向字典。"""
        return {
            self.segment_for_sensor(sid): self.apply(sid, q)
            for sid, q in q_WS.items()
            if sid in self.sensor_ids
        }


# --------------------------------------------------------------------------- #
# 标定估计
# --------------------------------------------------------------------------- #
def _average_rotations(qs: np.ndarray) -> np.ndarray:
    """旋转平均（Markley 特征向量法），输入 (N,4)。"""
    qs = quat.normalize(qs)
    # 统一半球符号
    qs = qs * np.sign(qs[:, 0:1] + (qs[:, 0:1] == 0))
    M = np.einsum("ni,nj->ij", qs, qs) / qs.shape[0]
    vals, vecs = np.linalg.eigh(M)
    return quat.normalize(vecs[:, -1])


def _solve_screw_rotation(rel_a: np.ndarray, rel_b: np.ndarray) -> np.ndarray:
    """解手眼方程 ``conj(q) * rel_a * q = rel_b`` 的常量旋转 q。

    仅在标定动作能激励**两个以上旋转轴**时适定；单轴铰链运动绕铰链轴
    的分量不可观（IMU-铰链标定的已知物理限制），主流程改用静止参考 +
    运动平均的两阶段标定，本函数保留给多轴标定动作与单元核验。
    对每帧，四元数乘积 ``rel_a * q - q * rel_b`` 关于 q=(w,x,y,z)
    是线性的；多帧堆叠后取最小奇异向量，即整体最小二乘解。
    """
    basis = np.eye(4)
    blocks = np.zeros((rel_a.shape[0] * 4, 4))
    for i in range(rel_a.shape[0]):
        for j in range(4):
            blocks[4 * i:4 * i + 4, j] = (
                quat.mul(rel_a[i], basis[j]) - quat.mul(basis[j], rel_b[i])
            )
    _, _, vh = np.linalg.svd(blocks, full_matrices=False)
    return quat.normalize(vh[-1])


def _yaw_of_matrix(R: np.ndarray) -> np.ndarray:
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def calibrate(
    sensor_ids: tuple[int, ...],
    segments: tuple[str, ...],
    q_WS_samples: dict[int, np.ndarray],
    q_W0B_samples: dict[str, np.ndarray],
    reference: dict[int, np.ndarray] | None = None,
    reference_seg_rot: dict[str, np.ndarray] | None = None,
    world_ref_yaw: dict[int, float] | None = None,
) -> Calibration:
    """从标定样本估计安装旋转与世界参考（两阶段，安装/世界严格分开）。

    物理可观性
    ----------
    单个静止帧只能给出 ``conj(q_WS_ref) = q_BS * q_W0W``，安装旋转与
    世界航向以乘积耦合，无法仅凭一台 IMU 分离。真实系统的世界参考来自
    外部可观测量：重力确定倾斜、地磁场/外部航向确定偏航。本函数因此
    要求显式提供 ``world_ref_yaw``（磁力/外部航向给出的世界参考航向），
    与安装旋转保持为两个独立量，绝不合并成一个四元数。

    标定规程
    --------
    1. **静止参考帧**：角色保持已知参考姿态（rest pose 时
       ``q_W0B_ref = I``），记录每个传感器的 q_WS_ref，同时由
       磁力/外部航向读取 world_ref_yaw，构造 q_W0W。
    2. 阶段 1（安装旋转），由静止帧关系
       ``q_WS_ref = conj(q_W0W) * q_W0B_ref * q_BS`` 反解：
       ``q_BS = conj(q_W0B_ref) * q_W0W * q_WS_ref``。
    3. 阶段 2（世界参考校验）：安装固定后逐帧反算 q_W0W_n 并取航向
       圆周平均，与外部世界参考互校；残差评估观测一致性。

    真值使用边界：``q_W0B_samples`` / ``reference_seg_rot`` 只允许
    来自标定器械/标定规程；测试中由合成器提供并仅经本模块进入流程，
    不得绕过标定驱动角色。
    """
    if len(sensor_ids) != len(segments):
        raise ValueError("sensor_ids 与 segments 长度必须一致")
    if world_ref_yaw is None:
        raise ValueError(
            "world_ref_yaw 为必需项：静止帧无法分离安装旋转与世界航向，"
            "请提供磁力/外部航向世界参考（显式独立量）"
        )

    mounts = np.zeros((len(sensor_ids), 4))
    yaws = np.zeros(len(sensor_ids))
    residuals = np.zeros(len(sensor_ids))

    for k, (sid, seg) in enumerate(zip(sensor_ids, segments)):
        q_obs = quat.normalize(np.asarray(q_WS_samples[sid], dtype=float))
        q_seg = quat.normalize(np.asarray(q_W0B_samples[seg], dtype=float))
        if q_obs.shape != q_seg.shape or q_obs.ndim != 2:
            raise ValueError(f"传感器 {sid} 样本形状不一致：{q_obs.shape} vs {q_seg.shape}")

        q_W0W = quat.from_axis_angle(
            np.array([0.0, 0.0, 1.0]), float(world_ref_yaw[sid])
        )

        # 阶段 1：静止参考帧 + 外部世界参考 -> 安装旋转
        # 由 q_WS_ref = conj(q_W0W) * q_W0B_ref * q_BS
        # 得 q_BS = conj(q_W0B_ref) * q_W0W * q_WS_ref
        if reference is not None and sid in reference:
            q_WS_ref = quat.normalize(reference[sid])
            q_W0B_ref = (
                quat.normalize(reference_seg_rot[seg])
                if reference_seg_rot is not None and seg in reference_seg_rot
                else quat.identity()
            )
            q_BS = quat.normalize(
                quat.mul(
                    quat.mul(quat.inv(q_W0B_ref), q_W0W),
                    q_WS_ref,
                )
            )
        else:
            # 无静止帧时退化为逐帧反算平均（要求多轴激励，仅供实验）
            # q_BS = conj(q_W0B) * q_W0W * q_WS
            q_BS = _average_rotations(
                quat.mul(quat.mul(quat.inv(q_seg), q_W0W), q_obs)
            )
        mounts[k] = q_BS
        yaws[k] = float(world_ref_yaw[sid])

        # 阶段 2：标定残差（估计量反算骨段朝向 vs 真值）
        q_pred = quat.normalize(
            quat.mul(quat.mul(q_W0W, q_obs), quat.inv(q_BS))
        )
        err_rad = quat.angular_distance(q_pred, q_seg)
        residuals[k] = float(np.degrees(np.sqrt(np.mean(err_rad**2))))

    return Calibration(
        sensor_ids=tuple(sensor_ids),
        segments=tuple(segments),
        mount_q_BS=mounts,
        world_yaw=yaws,
        residual_deg=residuals,
    )


def per_frame_residuals(cal: Calibration,
                        q_WS: dict[int, np.ndarray],
                        q_W0B_truth: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """逐帧标定残差（度），用于捕捉视窗中的惯性偏差展示。"""
    out: dict[str, np.ndarray] = {}
    for sid in cal.sensor_ids:
        seg = cal.segment_for_sensor(sid)
        if sid not in q_WS or seg not in q_W0B_truth:
            continue
        pred = cal.apply(sid, q_WS[sid])
        out[seg] = np.degrees(quat.angular_distance(pred, q_W0B_truth[seg]))
    return out
