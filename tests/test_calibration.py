"""显式标定与完整坐标变换链核验。

关键约束：测试绝不把真值骨骼旋转绕过标定直接喂给角色。真值只允许出现在
(1) 标定样本、(2) 指标对照两处；求解器输入只能是合成的传感器观测 q_WS。
"""

import numpy as np
import pytest

from inertial_pose import quat
from inertial_pose.calibration import Calibration, calibrate
from inertial_pose.skeleton import DEFAULT_SENSOR_PLACEMENT, DEFAULT_SKELETON
from inertial_pose.synthetic import (
    generate_motion,
    make_calibration,
    random_mount_rotations,
)


def _motion(seed=0, noise=0.0, action="walk"):
    rng = np.random.default_rng(seed)
    mounts = random_mount_rotations(None, rng)
    yaw = {sid: 0.35 for sid in mounts}
    return generate_motion(
        action=action, duration=4.0, dt=0.05,
        mount_q_BS=mounts, world_yaw=yaw, noise_deg=noise, seed=seed,
    )


def test_calibration_requires_explicit_world_yaw():
    m = _motion()
    sids = tuple(m.placement[s] for s in m.placement)
    segs = tuple(m.placement.keys())
    qws = {sid: m.sensor_q_WS[sid][:10] for sid in sids}
    qseg = {seg: m.truth_segment_rot[seg][:10] for seg in segs}
    with pytest.raises(ValueError):
        calibrate(sids, segs, qws, qseg, reference=m.reference_q_WS,
                  world_ref_yaw=None)


def test_three_frames_kept_separate():
    """安装旋转与世界航向是两个独立量，不能合并成单个四元数。"""
    m = _motion()
    cal = make_calibration(m)
    # 每个传感器都有独立的安装旋转与世界航向，且形状/单位正确
    assert cal.mount_q_BS.shape == (len(cal.sensor_ids), 4)
    assert np.allclose(np.linalg.norm(cal.mount_q_BS, axis=1), 1.0)
    assert cal.world_yaw.shape == (len(cal.sensor_ids),)
    # 世界航向确实被恢复为生成时使用的常量
    for sid in cal.sensor_ids:
        assert np.isclose(cal.world_yaw[cal.index_for_sensor(sid)], 0.35)


def test_transform_chain_recovers_truth_segment_noiseless():
    """q_W0B = q_W0W * q_WS * q_BS^-1 必须还原真值骨段朝向。"""
    m = _motion(seed=5)
    cal = make_calibration(m)
    for sid in cal.sensor_ids:
        seg = cal.segment_for_sensor(sid)
        pred = cal.apply_batch({sid: m.sensor_q_WS[sid]})[seg]
        err = np.degrees(quat.angular_distance(pred, m.truth_segment_rot[seg]))
        assert np.max(err) < 1e-6


def test_mount_recovery_matches_generation():
    """静止参考 + 外部世界航向应精确恢复生成时的安装旋转。"""
    m = _motion(seed=11)
    cal = make_calibration(m)
    for sid in cal.sensor_ids:
        est = cal.mount_q_BS[cal.index_for_sensor(sid)]
        true = m.true_mount_q_BS[sid]
        # 四元数符号等价
        assert np.isclose(abs(float(np.dot(est, true))), 1.0, atol=1e-8)


def test_calibration_residual_small_noiseless():
    m = _motion(seed=3)
    cal = make_calibration(m)
    assert np.max(cal.residual_deg) < 1e-6


def test_noisy_calibration_residual_scales_with_noise():
    m = _motion(seed=4, noise=1.5)
    cal = make_calibration(m)
    # 噪声 ~1.5° 时残差应为同量级的小角度，而非完全失败
    assert np.max(cal.residual_deg) < 8.0


def test_apply_unknown_sensor_raises():
    m = _motion()
    cal = make_calibration(m)
    with pytest.raises(KeyError):
        cal.apply(999, quat.identity())


def test_observation_synthesis_is_physical_inverse():
    """生成端 q_WS = conj(q_W0W) * q_W0B * q_BS 必须与标定 apply 互逆。"""
    m = _motion(seed=8)
    cal = make_calibration(m)
    sid = cal.sensor_ids[0]
    seg = cal.segment_for_sensor(sid)
    # 不经任何真值骨骼旋转，仅凭观测 + 标定还原
    recovered = cal.apply(sid, m.sensor_q_WS[sid][0])
    assert np.isclose(
        quat.angular_distance(recovered, m.truth_segment_rot[seg][0]),
        0.0, atol=1e-8)
