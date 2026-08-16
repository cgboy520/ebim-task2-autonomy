"""EBiM Task 2 autonomy sidecar.

A ROS 2 node that subscribes to the EBiM Isaac Sim state/camera topics and
publishes arm + gripper commands through the direct ``/isaac/*`` contract.
Ships three interchangeable policies behind one ``Decision`` interface:

* :class:`Task2WaypointPolicy`  — fail-closed fixed-waypoint FSM (trusted fallback)
* :class:`PerceptionPolicy`     — geometric perception -> IK -> interpolation
* :class:`VLAPolicy`            — foundation VLA action-chunk policy (lazy model)

Heavy deps (``rclpy``, ``torch``, ``cv2``) are imported lazily so the library is
unit-testable offline with only numpy + pydantic + pyyaml.
"""

from __future__ import annotations

from .calibration import estimate_world_to_arm_base, pose_msg_to_matrix, rpy_to_rotation
from .config import Task2Config, load_config
from .perception import PlacementObservation, TargetPose, estimate_target_pose, observe_placement
from .policy import Decision, Phase, Task2WaypointPolicy
from .motion import IKResult, franka_fk, interpolate_waypoints, solve_ik
from .vla_policy import PerceptionPolicy, VLAConfig, VLAPolicy

__all__ = [
    "Task2Config",
    "load_config",
    "PlacementObservation",
    "TargetPose",
    "observe_placement",
    "estimate_target_pose",
    "Decision",
    "Phase",
    "Task2WaypointPolicy",
    "PerceptionPolicy",
    "VLAPolicy",
    "VLAConfig",
    "franka_fk",
    "solve_ik",
    "interpolate_waypoints",
    "IKResult",
    "pose_msg_to_matrix",
    "estimate_world_to_arm_base",
    "rpy_to_rotation",
]
