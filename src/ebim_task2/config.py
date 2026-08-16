"""Pydantic-v2 configuration models and YAML loader for EBiM Task 2 autonomy.

Fail-closed validation via Pydantic v2 field validators; serialized field
names match the example YAML exactly.
"""

from __future__ import annotations

import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JOINT_COUNT = 7
WAYPOINT_NAMES = ("pregrasp", "grasp", "lift", "preplace", "place")
_ACTIVE_ARMS = ("left", "right")


class ControlConfig(BaseModel):
    """Runtime control settings. ``enabled`` gates publishing (False =
    observe-only); True requires calibrated joint limits and waypoints
    (enforced in :meth:`Task2Config.validate_after`)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    active_arm: Literal["left", "right"] = "right"
    publish_rate_hz: float = Field(default=10.0, gt=0.0)
    joint_tolerance_rad: float = Field(default=0.035, gt=0.0)
    waypoint_timeout_s: float = Field(default=12.0, gt=0.0)
    verification_timeout_s: float = Field(default=6.0, gt=0.0)
    max_episode_s: float = Field(default=180.0, gt=0.0)
    safe_joint_limits_rad: list[tuple[float, float]] = Field(default_factory=list)
    # raise both arms to the ready pose ~2 s at startup (low-hanging arms
    # block base +x motion)
    travel_pose_on_start: bool = True
    # node clock from the simulator: all *_s durations are then SIM seconds
    use_sim_time: bool = False

    @field_validator("safe_joint_limits_rad")
    @classmethod
    def _check_limit_pairs(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for pair in v:
            if len(pair) != 2:
                raise ValueError("each joint limit must be a [min, max] pair")
            if pair[0] > pair[1]:
                raise ValueError(f"joint limit min > max: {pair}")
        return v


class TopicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arm_state: str
    arm_command: str
    gripper_command: str
    semantic_image: str
    # Simulator clock for control.use_sim_time (Isaac publishes /isaac/clock,
    # not the ROS-conventional /clock; the runner remaps).
    clock: str = "/clock"
    # Mobile-base pedal token topic (std_msgs/String; see base_command.py).
    # Only published when a policy decision carries a base twist.
    pedal_state: str = "/pedal/state"


class SemanticIds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thermalpad: int
    target: int
    liner: int


class SuccessGate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_iou: float = Field(default=0.85, ge=0.0, le=1.0)
    liner_dominance_ratio: float = Field(default=0.90, ge=0.0, le=1.0)


class CameraConfig(BaseModel):
    """Pixel→workspace mapping for the semantic camera: pixel_scale = px/m at
    the table plane, (cx, cy) principal point, origin_* workspace shift in
    metres; flip_y mirrors image-v (top-down camera: image +v -> world −Y)."""
    model_config = ConfigDict(extra="forbid")
    cx: float = 320.0
    cy: float = 240.0
    pixel_scale: float = Field(default=1000.0, gt=0.0)
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0
    flip_y: bool = False
    # height-aware px/m: pinhole fx/(camera_z-object_z). Unset
    # camera_height_m = flat scale everywhere.
    camera_height_m: float | None = None
    #: Height the pixel_scale above is stated at.
    pixel_scale_plane_z_m: float = 0.10
    #: Heights of what the grasp phases and the place phases actually aim at.
    pad_plane_z_m: float = 0.10
    target_plane_z_m: float = 0.001
    #: Depth topic used to recover the pad's geometry without ground truth.
    depth_topic: str = "/isaac/eval_camera/depth"
    #: Focal length in px from the camera_info K matrix (K[0]); required for
    #: the depth path (the flat pixel_scale cannot substitute).
    focal_px: float | None = None


class GripperConfig(BaseModel):
    """Robotiq driver-joint semantics (EBiM contract): 0.0 rad = open,
    0.8 rad = closed. The policy side always speaks open-fraction; the runner
    maps to driver radians when publishing."""
    model_config = ConfigDict(extra="forbid")
    open_rad: float = 0.0
    closed_rad: float = 0.8


class PolicyConfig(BaseModel):
    """Cartesian heights for the perception policy, in metres; with arm-base
    calibration these are WORLD-frame z (table plane 0.75). ``release_*``
    shapes the release-peel, ``servo_*`` bounds the ee servo in GRASP/PLACE."""
    model_config = ConfigDict(extra="forbid")
    grasp_height_m: float = 0.02
    place_height_m: float | None = None   # defaults to grasp_height_m when unset
    clearance_height_m: float = 0.18
    servo_iters: int = Field(default=3, ge=0)
    servo_tol_m: float = Field(default=0.006, gt=0.0)
    servo_settle_tol_rad: float = Field(default=0.03, gt=0.0)
    servo_settle_timeout_s: float = Field(default=6.0, gt=0.0)
    grasp_close_dwell_s: float = Field(default=1.5, ge=0.0)
    transport_step_rad: float = Field(default=0.05, gt=0.0)
    # "pinch": open descent, close at depth; "press": closed-jaw press-bond;
    # "push_only": floor push delivers (no grasp/carry/whip)
    grasp_mode: Literal["pinch", "press", "push_only"] = "pinch"
    # double-tap: second press after a short lift; slow final approach
    press_double_tap: bool = False
    press_double_tap_lift_m: float = Field(default=0.02, gt=0.0)
    press_slow_final_m: float = Field(default=0.0, ge=0.0)
    # grasp/press depth below the GT sheet top + per-regrasp increment
    # (capped)
    grasp_depth_m: float = Field(default=0.008, ge=0.0)
    # fingertip clearance above the sheet's support (live GT); 0 = no clamp
    grasp_support_margin_m: float = Field(default=0.0, ge=0.0)
    # carry open-fraction (1 = open, 0 = closed)
    grasp_close_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    # extra yaw (rad) for the GRASP pose only
    grasp_yaw_offset_rad: float = 0.0
    grasp_depth_step_m: float = Field(default=0.002, ge=0.0)
    grasp_depth_max_steps: int = Field(default=2, ge=0)
    release_dwell_s: float = Field(default=1.2, ge=0.0)
    release_press_m: float = Field(default=0.004, ge=0.0)
    release_wipe_m: float = Field(default=0.02, ge=0.0)
    release_lift_m: float = Field(default=0.06, ge=0.0)
    release_slow_step_rad: float = Field(default=0.015, gt=0.0)
    release_timeout_s: float = Field(default=25.0, gt=0.0)
    release_twist_rad: float = Field(default=0.0, ge=0.0)
    release_shake_cycles: int = Field(default=0, ge=0)
    release_shake_amp_m: float = Field(default=0.04, gt=0.0)
    # wrist-flip dump at the end of the press release; disabled = shakes
    # only (the dump lives in the re-shake path)
    release_dump_enabled: bool = True
    release_tilt_rad: float = Field(default=0.0, ge=0.0)
    # floor-push recovery after the release drop (press mode)
    push_enabled: bool = False
    push_height_m: float = Field(default=0.02, ge=0.0)
    push_settle_s: float = Field(default=5.0, ge=0.0)
    push_standoff_m: float = Field(default=0.08, ge=0.0)
    push_stop_short_m: float = Field(default=0.055, ge=0.0)
    # world-frame xy offset on the whip pose so the deterministic fling
    # lands the sheet on the target
    whip_offset_xy: list[float] = [0.0, 0.0]
    # two-arm pin-and-peel: LEFT fingertip pins the sheet during the whip;
    # requires left-arm calibration from /isaac/left_ee_pose (record mode)
    pin_enabled: bool = False
    pin_offset_xy: list[float] = [0.012, -0.038]
    # may be negative: a below-floor command = sustained PD press
    pin_press_z: float = Field(default=0.008, ge=-0.02)
    pin_tilt_rad: float = Field(default=0.6, ge=0.0)
    pin_clear_z: float = Field(default=0.18, gt=0.0)



class CalibrationConfig(BaseModel):
    """Runtime arm-base self-calibration from /isaac/{arm}_ee_pose:
    T_world_armbase ≈ T_world_ee × inv(FK(q)), averaged over `samples`
    stationary samples. Rigid mount ⇒ translation spread should be ≪ 1 cm.

    ee_pose only exists when the scene runs with ``--record``; the official
    organizer-run launch may not pass it. If no ee_pose arrives within
    ``timeout_s`` of startup, the runner installs the static fallback
    transform (``fallback_translation`` + ``fallback_rotation_rpy`` — the
    default-spawn estimate: arm base 0.353 m forward of base_link at the
    default height) and logs a warning."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    samples: int = Field(default=20, ge=1)
    max_translation_spread_m: float = Field(default=0.02, gt=0.0)
    timeout_s: float = Field(default=10.0, gt=0.0)
    fallback_translation: list[float] = [4.75, 2.60, 0.51]
    fallback_rotation_rpy: list[float] = [0.0, 0.0, -1.5708]


class Waypoints(BaseModel):
    """The five fixed-waypoint joint vectors. Empty lists are tolerated while
    ``control.enabled`` is False so the shipped example config stays loadable."""

    model_config = ConfigDict(extra="forbid")
    pregrasp: list[float] = Field(default_factory=list)
    grasp: list[float] = Field(default_factory=list)
    lift: list[float] = Field(default_factory=list)
    preplace: list[float] = Field(default_factory=list)
    place: list[float] = Field(default_factory=list)


class VLABackendConfig(BaseModel):
    """Deployment of a fine-tuned policy behind ``--policy vla``.

    ``checkpoint`` points at what ``lerobot-train`` writes under
    ``outputs/train/<run>/checkpoints/<step>/pretrained_model``. Leave the
    backend at "stub" and the runner holds position."""

    model_config = ConfigDict(extra="forbid")
    backend: Literal["stub", "lerobot", "smolvla", "act", "pi05", "openvla"] = "stub"
    checkpoint: str | None = None
    action_chunk_size: int = Field(default=8, ge=1)
    # Per-step joint-delta clamp (rad).
    max_joint_delta: float = Field(default=0.15, gt=0.0)
    #: Must match the recorder's `single_task` string — SmolVLA conditions
    #: on it.
    instruction: str = (
        "Pick up the thermal pad and place it on the target RAM board."
    )
    #: fps of the dataset the checkpoint was trained on;
    #: `control.publish_rate_hz` must equal it.
    dataset_fps: float | None = Field(default=None, gt=0.0)


class PadSourceConfig(BaseModel):
    """Where the pad's geometry comes from.

    ``gt`` reads ``/isaac/task2/pad_points`` — the record-mode ground-truth
    topic (topics.yaml ``ground_truth:``), a bridge interface the scene
    publishes under ``--record``. ``depth`` recovers the same four
    quantities (centroid, top-cluster centroid, crest, support surface)
    from the eval camera's depth + semantic mask.
    ``depth_with_gt_fallback`` prefers depth and falls back to GT."""

    model_config = ConfigDict(extra="forbid")
    source: Literal["gt", "depth", "depth_with_gt_fallback"] = "gt"
    #: Log the depth-vs-GT disagreement every N pad updates (0 = never).
    compare_every: int = Field(default=0, ge=0)


class Task2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    control: ControlConfig = Field(default_factory=ControlConfig)
    topics: TopicConfig
    semantic_ids: SemanticIds
    success_gate: SuccessGate = Field(default_factory=SuccessGate)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    gripper: GripperConfig = Field(default_factory=GripperConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    vla: VLABackendConfig = Field(default_factory=VLABackendConfig)
    pad_source: PadSourceConfig = Field(default_factory=PadSourceConfig)
    waypoints: Waypoints = Field(default_factory=Waypoints)

    # ---- fail-closed cross-field validation (mirrors the scaffold) ----
    @model_validator(mode="after")
    def validate_after(self) -> "Task2Config":
        arm = self.control.active_arm
        if arm not in _ACTIVE_ARMS:
            raise ValueError(f"active_arm must be one of {_ACTIVE_ARMS}, got {arm!r}")

        if self.control.enabled:
            limits = self.control.safe_joint_limits_rad
            if len(limits) != JOINT_COUNT:
                raise ValueError(
                    f"control.enabled=True requires exactly {JOINT_COUNT} "
                    f"safe_joint_limits_rad pairs, got {len(limits)}"
                )
            for name in WAYPOINT_NAMES:
                wp = getattr(self.waypoints, name)
                if len(wp) != JOINT_COUNT:
                    raise ValueError(
                        f"control.enabled=True requires waypoint '{name}' to have "
                        f"{JOINT_COUNT} joint positions, got {len(wp)}"
                    )
        return self

    @property
    def waypoint_table(self) -> dict[str, tuple[float, ...]]:
        """Frozen waypoint tuples keyed by phase name."""
        return {name: tuple(getattr(self.waypoints, name)) for name in WAYPOINT_NAMES}


def load_config(path: str | pathlib.Path) -> Task2Config:
    """Load and validate a Task 2 YAML config."""
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Task2Config.model_validate(raw)
