"""惯性观测驱动的角色姿态模块。

子模块划分：

- :mod:`inertial_pose.quat`     四元数 / 旋转矩阵数学（无第三方依赖约定外的工具）
- :mod:`inertial_pose.skeleton` 骨架定义、骨段局部轴与层级关系
- :mod:`inertial_pose.calibration` 传感器->骨段标定（安装旋转、世界参考分开估计）
- :mod:`inertial_pose.sensors`  原始 IMU 数据 -> 传感器世界朝向（Mahony 滤波）
- :mod:`inertial_pose.solver`   观测 -> 满足关节关系的角色姿态（含足锁/航向仲裁）
- :mod:`inertial_pose.synthetic` 已知动作 + 安装偏差的数据生成
- :mod:`inertial_pose.metrics`  角度误差、标定残差、足滑距离、躯干转角跃变
- :mod:`inertial_pose.capture`  角色捕捉视窗（图形入口）
- :mod:`inertial_pose.cli`      命令行辅助入口

坐标框架（三个层次严格分开，禁止混用）：

1. **传感器框架 S**：IMU 芯片自身的正交轴，观测四元数把 S 转到世界系。
2. **世界/导航框架 W**：重力与地磁场（或航向先验）定义的世界系，
   每个传感器的世界参考允许有独立航向偏置 ``world_ref``。
3. **骨段/骨架局部框架 B**：骨架骨段的规范局部轴（由骨架 rest pose 定义）。

完整链路（每一步均可核验）::

    q_WS (观测) -> q_WW_i (世界参考修正) -> q_BS (标定安装旋转)
        -> q_WB_i (骨段世界朝向) -> 父子层级 -> q_local (关节局部旋转)
        -> 关节自由度约束/先验 -> 最终姿态
"""

__all__ = [
    "quat",
    "skeleton",
    "calibration",
    "sensors",
    "solver",
    "synthetic",
    "metrics",
    "capture",
    "cli",
]

__version__ = "0.1.0"
