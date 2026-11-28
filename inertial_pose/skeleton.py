"""骨架定义：层级、骨段局部轴、关节自由度与正运动学。

骨架信息只描述**角色自身**的约定（骨段框架 B 的局部轴），与传感器、
世界参考完全独立。关节局部旋转是相对父骨段框架的旋转，永远不能把
传感器世界四元数直接塞给骨骼——必须经 :mod:`inertial_pose.solver`
完成全部坐标变换。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import quat


@dataclass(frozen=True)
class JointSpec:
    """单个关节的静态定义。

    parent      父关节名（根为 None）
    rest_offset 相对父关节的 rest 平移（米）
    segment     该关节驱动的骨段名；骨段朝向取该关节的世界旋转
    dof         允许的旋转自由度轴（骨架局部轴，取自 'x','y','z' 的子集）；
                空元组表示该关节不接受观测、仅使用先验
    angle_limit 每个允许轴的 (下限, 上限) 弧度，None 表示不限制
    """

    name: str
    parent: str | None
    rest_offset: tuple[float, float, float]
    segment: str
    dof: tuple[str, ...] = ("x", "y", "z")
    angle_limit: tuple[tuple[float, float] | None, ...] = (None, None, None)


@dataclass
class Skeleton:
    joints: dict[str, JointSpec]
    order: tuple[str, ...]
    segments: tuple[str, ...]
    # 骨段 -> 驱动它的关节名
    segment_joint: dict[str, str] = field(init=False)
    children: dict[str, list[str]] = field(init=False)

    def __post_init__(self) -> None:
        self.segment_joint = {js.segment: js.name for js in self.joints.values()}
        self.children = {n: [] for n in self.order}
        for js in self.joints.values():
            if js.parent is not None:
                self.children[js.parent].append(js.name)

    # ------------------------------------------------------------------ #
    def joint_dof_indices(self, name: str) -> tuple[int, ...]:
        """关节允许轴在 3D 旋转向量中的索引（0=x,1=y,2=z）。"""
        return tuple("xyz".index(a) for a in self.joints[name].dof)

    def rest_rotation(self, name: str) -> np.ndarray:
        """rest pose 下关节的世界旋转（本骨架 rest 即单位旋转）。"""
        return quat.identity()

    def forward_kinematics(
        self,
        local_rot: dict[str, np.ndarray],
        root_pos: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """正运动学。

        local_rot : 每个关节的局部旋转四元数（相对父框架），缺失视为单位旋转
        root_pos  : 根（pelvis）世界位置，形状 (3,)
        返回 (世界旋转字典, 世界位置字典)。
        """
        world_rot: dict[str, np.ndarray] = {}
        world_pos: dict[str, np.ndarray] = {}
        for name in self.order:
            js = self.joints[name]
            q_local = local_rot.get(name, quat.identity())
            if js.parent is None:
                world_rot[name] = quat.normalize(q_local)
                world_pos[name] = np.asarray(root_pos, dtype=float).copy()
            else:
                world_rot[name] = quat.normalize(quat.mul(world_rot[js.parent], q_local))
                offset_w = quat.rotate_vectors(
                    world_rot[js.parent], np.asarray(js.rest_offset, dtype=float)
                )
                world_pos[name] = world_pos[js.parent] + offset_w
        return world_rot, world_pos

    def segment_world_rotations(
        self, world_rot: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """关节世界旋转 -> 骨段世界旋转（本骨架中骨段固连驱动关节）。"""
        return {seg: world_rot[j].copy() for seg, j in self.segment_joint.items()}


# --------------------------------------------------------------------------- #
# 标准人形骨架实例
# --------------------------------------------------------------------------- #
# rest 尺寸（米），近似成年人体型；Z 轴向上，角色面向 +X，左侧为 +Y
def _build_default_skeleton() -> Skeleton:
    J: dict[str, JointSpec] = {}

    def add(name, parent, offset, segment, dof=("x", "y", "z"), limits=(None, None, None)):
        J[name] = JointSpec(name, parent, offset, segment, dof, limits)

    add("pelvis", None, (0.0, 0.0, 0.95), "pelvis")
    add("spine", "pelvis", (0.0, 0.0, 0.12), "spine",
        dof=("x", "y", "z"))
    add("neck", "spine", (0.0, 0.0, 0.18), "neck",
        dof=("y", "z"),
        limits=((-0.9, 0.9), (-0.8, 0.8)))
    add("head", "neck", (0.0, 0.0, 0.10), "head",
        dof=("x", "y", "z"),
        limits=((-0.6, 0.6), (-0.9, 0.9), (-0.5, 0.5)))

    add("l_shoulder", "spine", (0.0, 0.18, 0.16), "l_upperarm",
        dof=("x", "y", "z"))
    add("l_elbow", "l_shoulder", (0.0, 0.27, 0.0), "l_forearm",
        dof=("y",),
        limits=((-2.4, 0.0),))
    add("l_wrist", "l_elbow", (0.0, 0.25, 0.0), "l_hand",
        dof=("x", "y"),
        limits=((-1.2, 1.2), (-0.8, 0.8)))

    add("r_shoulder", "spine", (0.0, -0.18, 0.16), "r_upperarm",
        dof=("x", "y", "z"))
    add("r_elbow", "r_shoulder", (0.0, -0.27, 0.0), "r_forearm",
        dof=("y",),
        limits=((-2.4, 0.0),))
    add("r_wrist", "r_elbow", (0.0, -0.25, 0.0), "r_hand",
        dof=("x", "y"),
        limits=((-1.2, 1.2), (-0.8, 0.8)))

    add("l_hip", "pelvis", (0.0, 0.10, -0.04), "l_thigh",
        dof=("x", "y", "z"),
        limits=((-0.8, 0.8), (-1.2, 2.0), (-1.2, 0.6)))
    add("l_knee", "l_hip", (0.0, 0.0, -0.43), "l_shank",
        dof=("y",),
        limits=((-2.4, 0.0),))
    add("l_ankle", "l_knee", (0.0, 0.0, -0.42), "l_foot",
        dof=("x", "y"),
        limits=((-0.5, 0.5), (-0.8, 0.8)))

    add("r_hip", "pelvis", (0.0, -0.10, -0.04), "r_thigh",
        dof=("x", "y", "z"),
        limits=((-0.8, 0.8), (-1.2, 2.0), (-1.2, 0.6)))
    add("r_knee", "r_hip", (0.0, 0.0, -0.43), "r_shank",
        dof=("y",),
        limits=((-2.4, 0.0),))
    add("r_ankle", "r_knee", (0.0, 0.0, -0.42), "r_foot",
        dof=("x", "y"),
        limits=((-0.5, 0.5), (-0.8, 0.8)))

    order = (
        "pelvis", "spine", "neck", "head",
        "l_shoulder", "l_elbow", "l_wrist",
        "r_shoulder", "r_elbow", "r_wrist",
        "l_hip", "l_knee", "l_ankle",
        "r_hip", "r_knee", "r_ankle",
    )
    segments = tuple(J[n].segment for n in order)
    return Skeleton(J, order, segments)


DEFAULT_SKELETON = _build_default_skeleton()

# 默认的可穿戴传感器布点：骨段名 -> 传感器编号
DEFAULT_SENSOR_PLACEMENT: dict[str, int] = {
    "pelvis": 0,
    "spine": 1,
    "head": 2,
    "l_upperarm": 3,
    "l_forearm": 4,
    "r_upperarm": 5,
    "r_forearm": 6,
    "l_thigh": 7,
    "l_shank": 8,
    "l_foot": 9,
    "r_thigh": 10,
    "r_shank": 11,
    "r_foot": 12,
}

# 末端骨段（用于足锁锚点判定）
FOOT_SEGMENTS = ("l_foot", "r_foot")


def skeleton_lines(skeleton: Skeleton, positions: dict[str, np.ndarray]):
    """提取用于绘制的骨架线段（父子关节连线）。"""
    out = []
    for name in skeleton.order:
        js = skeleton.joints[name]
        if js.parent is not None:
            out.append((positions[js.parent], positions[name]))
    return out
