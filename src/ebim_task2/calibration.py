"""Runtime arm-base self-calibration for the EBiM Task 2 scene.

The arm sits on a mobile base, so its base frame is NOT at the world origin
(robot spawns at world (4.4, 2.6, 0), yaw −90°). The sim publishes the link8
end-effector world pose on ``/isaac/{left,right}_ee_pose`` at 60 Hz, from
which the mount transform is recovered at runtime:

    T_world_armbase ≈ T_world_ee × inv(FK(q))

The mount is rigid, so across stationary samples the spread of the estimated
translation should be ≪ 1 cm; any constant offset from the lightweight FK
model (:mod:`ebim_task2.motion`) is absorbed into the mean.

The mobile base is castored (driveable only via `/pedal/state` tokens) and
rolls under a reaching arm; `scene_reset` relocates the robot outright. The
scoring chain therefore re-derives the transform from the live (q, ee_pose)
pair at every waypoint (``mirror_lay.relocalise``); the batch estimator below
serves the runner's one-shot ``--policy`` path only.

Pure numpy, offline-testable; only the sample collection lives in the ROS
callback (runner), which installs the *inverse* of the estimated transform
on the policy so world-frame camera goals land in the arm-base frame
before IK.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .motion import franka_fk


def pose_msg_to_matrix(
    position_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a ROS Pose's position and
    quaternion (xyzw order, as in ``geometry_msgs/Pose``). Pure numpy."""
    x, y, z = (float(v) for v in position_xyz)
    qx, qy, qz, qw = (float(v) for v in quaternion_xyzw)
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 2.0 / n if n > 0.0 else 0.0
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    t = np.eye(4)
    t[:3, :3] = [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]
    t[:3, 3] = [x, y, z]
    return t


def rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix (3x3) from fixed-axis roll-pitch-yaw angles
    (R = Rz(yaw) @ Ry(pitch) @ Rx(roll), the ROS rpy convention). Used to build
    the static fallback arm-base transform when ee_pose is unavailable."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def estimate_world_to_arm_base(
    samples: list[tuple[Sequence[float], np.ndarray]],
) -> tuple[np.ndarray, float]:
    """Average (joints q, T_world_ee) samples into T_world_armbase + spread.

    Each sample contributes ``T_world_ee @ inv(FK_flange(q))``. The ee pose
    topic reports the link8 (flange) frame, so FK is evaluated with
    ``tool_offset=0.0``. Translations
    are averaged directly; rotation matrices are averaged and re-orthonormalized
    via SVD. Returns ``(T_mean, spread)`` where T_mean is T_world_armbase
    (base-frame -> world; invert it to bring world-frame goals into the base
    frame) and spread is the maximum translation deviation from the mean
    (metres). The mount is rigid, so the spread should be ≪ 1 cm; FK model
    error shows up as a constant offset absorbed into the mean.
    """
    if not samples:
        raise ValueError("need at least one (joints, T_world_ee) sample")
    transforms = [np.asarray(t_w_ee, dtype=np.float64) @ np.linalg.inv(franka_fk(q, tool_offset=0.0))
                  for q, t_w_ee in samples]
    mean_t = np.eye(4)
    mean_t[:3, 3] = np.mean([t[:3, 3] for t in transforms], axis=0)
    mean_r = np.mean([t[:3, :3] for t in transforms], axis=0)
    u, _, vt = np.linalg.svd(mean_r)
    mean_t[:3, :3] = u @ vt
    spread = float(max(np.linalg.norm(t[:3, 3] - mean_t[:3, 3]) for t in transforms))
    return mean_t, spread
