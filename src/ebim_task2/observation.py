"""Whole-body observation assembly for LeRobot-trained policies.

A policy fine-tuned on the benchmark's Task 2 recordings expects exactly the
observation the recorder wrote: a 37-D ``observation.state`` row plus one RGB
frame per trained camera. This module rebuilds that row live from the same
ROS topics ``services/recording/record_task2.py`` samples, in the same order,
so a checkpoint sees at inference what it saw during training.

State layout (mirrors record_task2.py's module docstring)::

    [0:7]   left EE pose   x y z qx qy qz qw   (world)
    [7:14]  right EE pose  x y z qx qy qz qw   (world)
    [14:21] left arm joint positions           (rad)
    [21:28] right arm joint positions          (rad)
    [28]    spine height                       (m)
    [29:31] gripper open-fractions             (left, right)
    [31:34] base odometry x, y, yaw            (m, m, rad)
    [34:37] base velocity vx, vy, wz           (body frame)

Missing channels stay NaN-free by holding their last value.
"""

from __future__ import annotations

import math

import numpy as np

STATE_DIM = 37

LEFT_JOINTS = [f"left_fr3v2_joint{i}" for i in range(1, 8)]
RIGHT_JOINTS = [f"right_fr3v2_joint{i}" for i in range(1, 8)]
SPINE_JOINT = "franka_spine_vertical_joint"
LEFT_GRIPPER_DRIVER = "left_right_finger_joint"
RIGHT_GRIPPER_DRIVER = "right_right_finger_joint"
GRIPPER_CLOSED_RAD = 0.8

#: Recorder camera key -> image topic.
CAMERA_TOPICS = {
    "head": "/isaac/head_camera/image_raw",
    "wrist_left": "/isaac/left_wrist_camera/image_raw",
    "wrist_right": "/isaac/right_wrist_camera/image_raw",
    "eval_camera": "/isaac/eval_camera/image_raw",
    # Alternate training-view mapping: head -> base_0_rgb, eval_camera ->
    # left_wrist_0_rgb. right_wrist_0_rgb deliberately has no topic: that
    # camera slot is absent in the training data (zero-pad + mask).
    "base_0_rgb": "/isaac/head_camera/image_raw",
    "left_wrist_0_rgb": "/isaac/eval_camera/image_raw",
}


def _joint_name_variants(name: str):
    """Robot-USD joint-name variants, mirroring the sim bridge's resolver."""
    yield name
    if "fr3v2_joint" in name:
        yield name.replace("fr3v2_joint", "fr3v2_1_joint")
    if name == "left_right_finger_joint":
        yield "left_fr3v2_finger_joint1"
    if name == "right_right_finger_joint":
        yield "right_fr3v2_finger_joint1"


def resolve_joint(joint_map: dict, name: str, default: float = 0.0) -> float:
    for candidate in _joint_name_variants(name):
        value = joint_map.get(candidate)
        if value is not None and math.isfinite(value):
            return float(value)
    return default


def gripper_open_fraction(driver_position_rad: float) -> float:
    """Driver-joint angle -> open fraction (1 = fully open), as recorded."""
    if not math.isfinite(driver_position_rad):
        return 0.0
    frac = 1.0 - (driver_position_rad / GRIPPER_CLOSED_RAD)
    return float(min(max(frac, 0.0), 1.0))


def quaternion_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw of a quaternion (ZYX convention), matching the recorder's odom yaw."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny, cosy))


class VLAObservationCollector:
    """Accumulates the latest ROS samples and assembles the 37-D state row.

    Every setter is a plain assignment so it can be driven straight from ROS
    callbacks (or, in tests, from plain dicts) without a ROS dependency here.
    """

    def __init__(self) -> None:
        self.ee_pose: dict[str, tuple[float, ...]] = {}
        self.joint_map: dict[str, float] = {}
        self.odom_xy_yaw: tuple[float, float, float] | None = None
        self.odom_vel: tuple[float, float, float] | None = None
        self.images: dict[str, np.ndarray] = {}

    # -- ingest -----------------------------------------------------------
    def set_ee_pose(self, side: str, xyz_quat: tuple[float, ...]) -> None:
        self.ee_pose[side] = tuple(float(v) for v in xyz_quat)

    def set_joints(self, names, positions) -> None:
        for name, pos in zip(names, positions):
            self.joint_map[str(name)] = float(pos)

    def set_odom(self, x, y, qx, qy, qz, qw, vx, vy, wz) -> None:
        self.odom_xy_yaw = (float(x), float(y), quaternion_yaw(qx, qy, qz, qw))
        self.odom_vel = (float(vx), float(vy), float(wz))

    def set_image(self, key: str, frame: np.ndarray) -> None:
        self.images[key] = frame

    # -- assemble ---------------------------------------------------------
    @property
    def ready(self) -> bool:
        """True once the arm joints are known — the one channel with no
        sensible default."""
        return all(
            math.isfinite(resolve_joint(self.joint_map, n, math.nan))
            for n in RIGHT_JOINTS
        )

    def state(self) -> np.ndarray:
        s = np.zeros(STATE_DIM, dtype=np.float32)
        for i, side in enumerate(("left", "right")):
            pose = self.ee_pose.get(side)
            if pose is not None and len(pose) >= 7:
                s[i * 7:i * 7 + 7] = pose[:7]
        for i, name in enumerate(LEFT_JOINTS):
            s[14 + i] = resolve_joint(self.joint_map, name)
        for i, name in enumerate(RIGHT_JOINTS):
            s[21 + i] = resolve_joint(self.joint_map, name)
        s[28] = resolve_joint(self.joint_map, SPINE_JOINT)
        s[29] = gripper_open_fraction(
            resolve_joint(self.joint_map, LEFT_GRIPPER_DRIVER, GRIPPER_CLOSED_RAD))
        s[30] = gripper_open_fraction(
            resolve_joint(self.joint_map, RIGHT_GRIPPER_DRIVER, GRIPPER_CLOSED_RAD))
        if self.odom_xy_yaw is not None:
            s[31:34] = self.odom_xy_yaw
        if self.odom_vel is not None:
            s[34:37] = self.odom_vel
        return s
