# 模块架构与数据流

## 模块职责

| 模块 | 职责 | 是否依赖第三方 |
| --- | --- | --- |
| `inertial_pose/quat.py` | 四元数/旋转矩阵/轴角数学，`[w,x,y,z]` 主动旋转、Hamilton 积 | numpy |
| `inertial_pose/skeleton.py` | 骨架层级、骨段局部轴、关节自由度与角限位、正运动学 | numpy |
| `inertial_pose/calibration.py` | 传感器→骨段显式标定，安装旋转与世界参考分开估计 | numpy |
| `inertial_pose/sensors.py` | 原始 IMU（陀螺+加速度）→ `q_WS` 的 Mahony 滤波；观测容器 | numpy |
| `inertial_pose/solver.py` | 观测→满足关节关系的姿态；自由度投影、足锁、航向仲裁 | numpy |
| `inertial_pose/synthetic.py` | 已知动作 + 安装偏差的合成数据与标定样本生成 | numpy |
| `inertial_pose/metrics.py` | 角度误差、标定残差、足滑、躯干航向跃变 | numpy |
| `inertial_pose/capture.py` | **图形入口**：标准库 Web 捕捉视窗 | 仅标准库 + numpy |
| `inertial_pose/cli.py` | 命令行**辅助**：calibrate/run/sweep/bias | 仅标准库 + numpy |

## 端到端数据流

```
已知动作 (synthetic._walk/_squat/_arm_raise)
   │  真值关节局部旋转 + 根位置
   │  骨架 FK → 真值骨段世界朝向 q_W0B
   ▼
传感器观测合成：q_WS = conj(q_W0W) · q_W0B · q_BS
   │  - q_BS：每传感器独立安装旋转（安装偏差）
   │  - q_W0W：该传感器导航世界航向（可叠加运行期漂移）
   │  - 可选旋转向量噪声
   ▼
显式标定 (calibration.calibrate / synthetic.make_calibration)
   │  静止参考帧 + 外部 world_ref_yaw
   │  阶段1 安装旋转 q_BS；阶段2 世界参考校验 + 残差
   ▼
观测序列 (sensors.sequence_from_orientations 或 MahonyFilter)
   ▼
姿态求解 (solver.PoseSolver.solve_frame)
   1) 观测轴向重排（正当性校验，det 必须 +1）
   2) cal.apply_batch：q_WS → q_W0B（标定是唯一入口）
   3) 自顶向下：世界朝向 → 关节局部旋转 → 自由度投影/限位
   4) 投影后层级再传播
   5) 初始 FK → 足锁状态机 + 航向冲突比价 → 提交根位置
   6) 最终 FK → 惯性偏差残差
   ▼
PoseResult（root_pos / local_rot / world_rot / world_pos /
           inertial_residual_deg / foot_slide_m / anchors / conflict）
   ▼
metrics.summarize（与真值对照） 或 capture 视窗（实时呈现）
```

## 关键不变量

1. **唯一入口**：观测进入骨架前必须经过 `Calibration.apply`，求解器不持有
   任何真值骨段旋转（`test_solver_inputs_are_only_sensor_observations`
   以反射方式断言求解器实例属性中不含 `truth`）。
2. **三框架不合并**：`Calibration` 把 `mount_q_BS` 与 `world_yaw` 存为两个
   独立数组；标定函数在缺少外部世界参考时直接抛错。
3. **自由度受限**：`project_to_dof` 的角限位与 `JointSpec.dof` 顺序对齐，
   未登记自由度的关节回退单位旋转（显式先验）。
4. **足滑测量对应提交姿态**：先在已提交根上做测量、双路线比价，提交后再
   用最终 FK 计算实际足滑；双支撑期只由最近落脚锚点驱动根，避免两锚点
   取均值互相拉扯。
5. **轴向重排是显式正当旋转**：非置换或 det −1 反射立即 `ValueError`，
   由捕捉视窗作为异常提示。

## 足锁状态机

```
                大腿俯仰 > exit 阈值（连续 swing_confirm_frames 帧）
   支撑 STANCE ───────────────────────────────────────────▶ 摆动 SWING
       ▲                                                          │
       │  俯仰回落 < enter 阈值（滞回，enter<exit）               │
       └──────────────────────────────────────────────────────────┘
   支撑帧：
     - 新落脚 → 在实测落点种植新锚点，场景航向仅首次初始化并跨落脚持久
     - 航向偏差 > 25° → 冲突，比较 cost_jump 与 cost_slide 二选一
     - 常规支撑 → 锚点世界位置锁定，不逐帧吸收漂移
```

## 合成动作的可检验性

- `_walk_cycle`：支撑脚踝世界位置在整个支撑段恒定，根高度恒定，平面二连杆
  IK 严格闭合，因此可断言支撑段水平位移为机器精度零。
- `_squat_cycle`：根位置由“左踝世界位置恒定”反解，双脚全程零滑。
- `_arm_raise_cycle`：无足部锚点，用于验证根位置回退固定先验、不抛错。

## 测试布局

| 文件 | 覆盖 |
| --- | --- |
| `test_quat.py` | 四元数积、旋转矩阵、轴角、误差度量 |
| `test_calibration.py` | 世界参考必需、三框架分离、物理反演、禁止旁路 |
| `test_dof_hierarchy.py` | 自由度投影、角限位、父子层级 |
| `test_driver_pipeline.py` | 端到端无噪零误差、有噪有界、零足滑、求解器无真值 |
| `test_conflict_arbitration.py` | 基线无冲突、漂移触发、权重翻转、假设足滑、持久航向 |
| `test_mahony.py` | 互补滤波收敛、纯积分航向、航向可观性限制 |
| `test_synthetic_physics.py` | 真实支撑段、左右腿同构、轴向重排正当性、残差为真实计算 |
