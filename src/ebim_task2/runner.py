"""ROS 2 entry node for the EBiM Task 2 autonomy sidecar.

Subscribes to the simulator's joint-state and semantic-segmentation topics and
publishes arm + gripper commands on the direct EBiM ``/isaac/*`` contract
(absolute 7-DOF rad targets; Robotiq driver joint ``{arm}_right_finger_joint``
with 0.0 rad = open, 0.8 rad = closed — policies always speak open-fraction,
the runner maps to driver radians). The node is fail-closed: it refuses to
publish when a target escapes the calibrated joint limits, opens the gripper on
episode-budget exhaustion, and emits nothing under ``--dry-run``.

``rclpy`` is imported lazily so this module can be imported (and its helpers
unit-tested) without a ROS install. The ``main`` entry point requires ROS.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

import numpy as np

try:
    from loguru import logger
except Exception:  # pragma: no cover - loguru is optional for tests
    import logging

    class _StdLogger:
        def __init__(self) -> None:
            self._l = logging.getLogger("ebim_task2")

        def info(self, msg, *a, **k):
            self._l.info(msg, *a, **k)

        def warning(self, msg, *a, **k):
            self._l.warning(msg, *a, **k)

        def error(self, msg, *a, **k):
            self._l.error(msg, *a, **k)

    logger = _StdLogger()  # type: ignore[assignment]

from .config import Task2Config, load_config
from .calibration import estimate_world_to_arm_base, pose_msg_to_matrix, rpy_to_rotation
from .perception import (
    PlacementObservation,
    infer_semantic_ids,
    observe_placement,
    predict_official_verdict,
)
from .policy import Decision, Phase, Task2WaypointPolicy
from .vla_policy import PerceptionPolicy, VLAPolicy

#: Policies that consume the ee-pose feed, the pad-points feed, and the
#: left-arm hold plumbing (the perception-family runtime contract).
_EE_POLICIES = (PerceptionPolicy,)


JOINT_COUNT = 7
# max benign overshoot (rad) clamped instead of refused
LIMIT_CLAMP_TOL = 0.06


def _assert_cadence_matches_training(cfg: Task2Config) -> None:
    """Refuse to run a checkpoint at a publish rate its dataset was not at.

    Executed speed is `publish_rate_hz / dataset_fps`; the lay's pace
    affects the score.

    Training-fps sources, in order of authority:

    1. the checkpoint's `train_config.json` -> dataset root ->
       `meta/info.json` (that root is the TRAINING machine's path, usually
       unreachable on a deployment host);
    2. `vla.dataset_fps`, declared in the config.

    With neither available, warn loudly rather than guess — never silently
    pass.
    """
    import json

    rate = float(cfg.control.publish_rate_hz)
    ckpt = getattr(cfg.vla, "checkpoint", None)
    if not ckpt:
        return
    declared = getattr(cfg.vla, "dataset_fps", None)
    fps: float | None = None
    tc = pathlib.Path(ckpt) / "train_config.json"
    if tc.is_file():
        try:
            root = (json.loads(tc.read_text()).get("dataset") or {}).get("root")
            fps = float(json.loads(
                (pathlib.Path(str(root)) / "meta" / "info.json").read_text()
            )["fps"])
        except Exception:
            fps = None      # unreachable training dataset: fall back to (2)
    if fps is None and declared is not None:
        fps = float(declared)
        logger.info(f"training fps taken from vla.dataset_fps={fps}")
    if fps is None:
        logger.warning(
            f"cannot verify cadence: the checkpoint's training dataset is "
            f"unreachable and vla.dataset_fps is unset; publish_rate_hz="
            f"{rate} is UNVERIFIED. Set vla.dataset_fps to the recorded fps."
        )
        return
    if abs(rate - fps) > 1e-6:
        raise SystemExit(
            f"publish_rate_hz={rate} != training dataset fps={fps}: the policy "
            f"would replay the demonstrated motion at {rate / fps:.3g}x speed, "
            f"which changes the score. Set control.publish_rate_hz to {fps}."
        )
    logger.info(f"cadence verified: publish_rate_hz={rate} == dataset fps={fps}")


def make_policy(kind: str, cfg: Task2Config):
    """Build the selected policy from the loaded config."""
    if kind == "waypoint":
        return Task2WaypointPolicy(
            cfg.waypoint_table,
            joint_tolerance_rad=cfg.control.joint_tolerance_rad,
            waypoint_timeout_s=cfg.control.waypoint_timeout_s,
            verification_timeout_s=cfg.control.verification_timeout_s,
            min_iou=cfg.success_gate.min_iou,
            liner_dominance_ratio=cfg.success_gate.liner_dominance_ratio,
        )
    if kind == "perception":
        cls = PerceptionPolicy
        extra: dict[str, Any] = {}
        return cls(
            **extra,
            joint_tolerance_rad=cfg.control.joint_tolerance_rad,
            waypoint_timeout_s=cfg.control.waypoint_timeout_s,
            verification_timeout_s=cfg.control.verification_timeout_s,
            min_iou=cfg.success_gate.min_iou,
            liner_dominance_ratio=cfg.success_gate.liner_dominance_ratio,
            safe_joint_limits=cfg.control.safe_joint_limits_rad,
            grasp_height=cfg.policy.grasp_height_m,
            place_height=cfg.policy.place_height_m,
            clearance_height=cfg.policy.clearance_height_m,
            cam_cx=cfg.camera.cx,
            cam_cy=cfg.camera.cy,
            pixel_scale=cfg.camera.pixel_scale,
            camera_height_m=cfg.camera.camera_height_m,
            pixel_scale_plane_z_m=cfg.camera.pixel_scale_plane_z_m,
            pad_plane_z_m=cfg.camera.pad_plane_z_m,
            target_plane_z_m=cfg.camera.target_plane_z_m,
            origin_xy=(cfg.camera.origin_x_m, cfg.camera.origin_y_m),
            flip_y=cfg.camera.flip_y,
            servo_iters=cfg.policy.servo_iters,
            servo_tol_m=cfg.policy.servo_tol_m,
            servo_settle_tol_rad=cfg.policy.servo_settle_tol_rad,
            servo_settle_timeout_s=cfg.policy.servo_settle_timeout_s,
            grasp_close_dwell_s=cfg.policy.grasp_close_dwell_s,
            transport_step_rad=cfg.policy.transport_step_rad,
            grasp_mode=cfg.policy.grasp_mode,
            press_double_tap=cfg.policy.press_double_tap,
            press_double_tap_lift_m=cfg.policy.press_double_tap_lift_m,
            press_slow_final_m=cfg.policy.press_slow_final_m,
            grasp_depth_m=cfg.policy.grasp_depth_m,
            grasp_support_margin_m=cfg.policy.grasp_support_margin_m,
            grasp_close_fraction=cfg.policy.grasp_close_fraction,
            grasp_yaw_offset_rad=cfg.policy.grasp_yaw_offset_rad,
            grasp_depth_step_m=cfg.policy.grasp_depth_step_m,
            grasp_depth_max_steps=cfg.policy.grasp_depth_max_steps,
            release_dwell_s=cfg.policy.release_dwell_s,
            release_press_m=cfg.policy.release_press_m,
            release_wipe_m=cfg.policy.release_wipe_m,
            release_lift_m=cfg.policy.release_lift_m,
            release_slow_step_rad=cfg.policy.release_slow_step_rad,
            release_timeout_s=cfg.policy.release_timeout_s,
            release_twist_rad=cfg.policy.release_twist_rad,
            release_shake_cycles=cfg.policy.release_shake_cycles,
            release_shake_amp_m=cfg.policy.release_shake_amp_m,
            release_tilt_rad=cfg.policy.release_tilt_rad,
            release_dump_enabled=cfg.policy.release_dump_enabled,
            push_enabled=cfg.policy.push_enabled,
            push_height_m=cfg.policy.push_height_m,
            push_settle_s=cfg.policy.push_settle_s,
            push_standoff_m=cfg.policy.push_standoff_m,
            push_stop_short_m=cfg.policy.push_stop_short_m,
            whip_offset_xy=cfg.policy.whip_offset_xy,
            pin_enabled=cfg.policy.pin_enabled,
            pin_offset_xy=cfg.policy.pin_offset_xy,
            pin_press_z=cfg.policy.pin_press_z,
            pin_tilt_rad=cfg.policy.pin_tilt_rad,
            pin_clear_z=cfg.policy.pin_clear_z,
            # pregrasp doubles as the photo pose: retreating there inside
            # RELEASE clears the top-down camera before VERIFY
            retreat_joints=cfg.waypoint_table.get("pregrasp") or None,
        )
    if kind == "vla":
        from .vla_policy import VLAConfig

        _assert_cadence_matches_training(cfg)
        pol = VLAPolicy(VLAConfig(
            backend=cfg.vla.backend,
            checkpoint=cfg.vla.checkpoint,
            action_chunk_size=cfg.vla.action_chunk_size,
            max_joint_delta=cfg.vla.max_joint_delta,
            active_arm=cfg.control.active_arm,
            instruction=cfg.vla.instruction,
        ))
        pol.load()  # best-effort; falls back to stub when no weights
        pol._min_iou = cfg.success_gate.min_iou
        pol._min_liner = cfg.success_gate.liner_dominance_ratio
        pol._verify_timeout = cfg.control.verification_timeout_s
        return pol
    raise ValueError(f"unknown policy kind: {kind!r}")


def _in_limits(target: tuple[float, ...], limits: list[tuple[float, float]]) -> bool:
    if not limits:
        return True
    if len(target) != len(limits):
        return False
    return all(lo <= v <= hi for v, (lo, hi) in zip(target, limits))


def _decode_semantic(msg: Any) -> np.ndarray | None:
    """Decode a ``sensor_msgs/Image`` semantic mask robustly.

    Checks ``msg.encoding``/dtype before reshaping and logs (rather than
    silently dropping) unexpected formats. Supports 32SC1 (primary) and the
    common ``mono8``/``8UC1`` labeled-image fallback.
    """
    try:
        import numpy as np_

        h = int(msg.height)
        w = int(msg.width)
        enc = getattr(msg, "encoding", "").lower()
        buf = bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray)) else msg.data

        if enc in ("32sc1", "s32c1", "int32") or enc == "":
            arr = np_.frombuffer(buf, dtype=np_.int32)
        elif enc in ("mono8", "8uc1", "uint8"):
            arr = np_.frombuffer(buf, dtype=np_.uint8).astype(np_.int32)
        elif enc in ("16uc1", "mono16", "uint16"):
            arr = np_.frombuffer(buf, dtype=np_.uint16).astype(np_.int32)
        else:
            logger.warning(f"unsupported semantic encoding {enc!r}; trying int32")
            arr = np_.frombuffer(buf, dtype=np_.int32)

        expected = h * w
        if arr.size < expected:
            logger.error(f"semantic image too small: got {arr.size}, expected {expected}")
            return None
        return arr[:expected].reshape(h, w).copy()
    except Exception as e:  # pragma: no cover - defensive in ROS callback
        logger.error(f"failed to decode semantic image: {e!r}")
        return None


class Task2AutonomyNode:
    """ROS 2 node shell. The ROS bits are created in :meth:`ros_init` so the
    class is importable without rclpy for testing the tick logic."""

    def __init__(self, cfg: Task2Config, policy, *, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.policy = policy
        self.dry_run = dry_run
        self.arm = cfg.control.active_arm
        self.arm_prefix = f"{self.arm}_fr3v2_joint"
        # Robotiq driver joint on the EBiM contract (yes, "left_right_finger_"
        # for the left arm); 0.0 rad = open, 0.8 rad = closed.
        self.gripper_joint = f"{self.arm}_right_finger_joint"

        self._joints: list[float] | None = None
        self._vla_obs = None            # VLAObservationCollector when policy=vla
        self._depth: np.ndarray | None = None
        self._pad_cmp_n = 0             # depth-vs-GT comparison counter
        self._joint_names_verified: bool = False
        self._mask: np.ndarray | None = None
        self._obs: PlacementObservation | None = None
        self._episode_start: float | None = None
        self._gripper_open = 1.0
        # Last spine height commanded by the policy; held and re-sent in EVERY
        # arm message (the bridge drops a group after a 1 s command gap)
        self._spine_cmd: float | None = None
        # Scene-instance semantic id map from /isaac/eval_camera/semantic_labels
        # (authoritative over the config defaults when present).
        self._label_ids: dict[str, int] | None = None
        self._label_ids_candidate: dict[str, int] | None = None
        self._label_ids_count: int = 0
        # Mask-geometry inference fallback (see on_semantic).
        self._infer_candidate: dict[str, int] | None = None
        self._infer_count: int = 0
        # Post-terminal photo-retreat bookkeeping (drives home for 8 s).
        self._photo_retreat_start: float | None = None
        # runner-side budget failure: non-terminal policies (VLA) still shut
        # down through the photo-retreat path on this flag
        self._budget_failed = False
        # set once terminal AND photo retreat done; the main loop exits on
        # this (a FAILED policy otherwise hangs the container)
        self.episode_done = False
        # Latest flange (link8) world pose from /isaac/{arm}_ee_pose (60 Hz),
        # fed to the perception policy each tick for the ee servo.
        self._ee_world: np.ndarray | None = None
        self._ee_seen: bool = False
        # arm-base self-calibration state (see calibration.py); stationary
        # samples only
        self._calib_samples: list[tuple[list[float], np.ndarray]] = []
        self._calib_done: bool = False
        self._calib_start_s: float | None = None
        # wall-clock start for the ee_pose fallback gate (the sim clock jumps
        # several seconds in under a wall second after a scene reset)
        self._calib_start_wall: float | None = None
        # True when the installed transform is the static fallback, not a
        # measured one — a later ee_pose re-arms sampling and supersedes it.
        self._calib_fallback: bool = False
        self._calib_last_q: list[float] | None = None
        self._calib_still_since: float | None = None

        self._node = None
        self._arm_pub = None
        self._gripper_pub = None
        self._pedal_pub = None
        self._timer = None
        # Left-arm (pin-and-peel) plumbing; wired only when policy.pin_enabled.
        self._left_joints: list[float] | None = None
        self._left_arm_pub = None
        self._left_gripper_pub = None
        self._left_calib_samples: list[tuple[list[float], np.ndarray]] = []
        self._left_calib_done = False
        self._left_calib_last_q: list[float] | None = None
        self._left_calib_still_since: float | None = None

    # -- ROS wiring (only when rclpy is available) ------------------------
    def ros_init(self) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState, Image
        from rclpy.qos import qos_profile_sensor_data

        # sim-time mode: config durations are SIM seconds; remap /clock onto
        # /isaac/clock
        init_args = None
        overrides = []
        if self.cfg.control.use_sim_time:
            if self.cfg.topics.clock != "/clock":
                init_args = ["--ros-args", "-r", f"/clock:={self.cfg.topics.clock}"]
            from rclpy.parameter import Parameter

            overrides = [Parameter("use_sim_time", Parameter.Type.BOOL, True)]
        rclpy.init(args=init_args)
        self._node = Node("ebim_task2_autonomy", parameter_overrides=overrides)
        if self.cfg.control.use_sim_time:
            logger.info(
                f"sim-time mode: node clock follows {self.cfg.topics.clock}; "
                f"all configured durations are sim seconds"
            )

        self._arm_pub = self._node.create_publisher(JointState, self.cfg.topics.arm_command, 10)
        self._gripper_pub = self._node.create_publisher(
            JointState, self.cfg.topics.gripper_command, 10
        )
        from std_msgs.msg import String

        # Mobile-base pedal tokens. Published at most once per control tick
        # (default 10 Hz — the bridge's pedal queue backlogs above that).
        self._pedal_pub = self._node.create_publisher(
            String, self.cfg.topics.pedal_state, 10
        )

        def on_state(msg: JointState) -> None:
            # validate joint names against the EBiM /isaac contract on the
            # first message (a mismatch silently produces a no-op arm)
            if not self._joint_names_verified and getattr(msg, "name", None):
                expected = [f"{self.arm_prefix}{i + 1}" for i in range(JOINT_COUNT)]
                got = list(msg.name)[:JOINT_COUNT]
                if got and got != expected:
                    logger.warning(
                        f"joint name mismatch: expected {expected}, got {got}. "
                        f"Arm commands may be ignored by the /isaac bridge."
                    )
                self._joint_names_verified = True
            self._joints = [float(p) for p in msg.position[:JOINT_COUNT]]

        def on_semantic(msg: Image) -> None:
            mask = _decode_semantic(msg)
            if mask is not None:
                self._mask = mask
                # second-priority id source: mask-geometry inference (the
                # labels JSON goes stale after an in-place reset)
                if self._label_ids is None:
                    inferred = infer_semantic_ids(mask)
                    if inferred is not None:
                        if inferred == self._infer_candidate:
                            self._infer_count += 1
                        else:
                            self._infer_candidate = inferred
                            self._infer_count = 1
                        if self._infer_count >= 8:
                            logger.info(f"semantic ids inferred from mask geometry: {inferred}")
                            self._label_ids = inferred
                            # fallback-id pose locks aim at the wrong blobs;
                            # drop the target lock (window open through LIFT)
                            if isinstance(self.policy, PerceptionPolicy):
                                self.policy._target_locked = None
                ids = self._label_ids or {
                    "thermalpad": self.cfg.semantic_ids.thermalpad,
                    "target": self.cfg.semantic_ids.target,
                    "liner": self.cfg.semantic_ids.liner,
                }
                self._obs = observe_placement(
                    mask,
                    thermalpad_id=ids["thermalpad"],
                    target_id=ids["target"],
                    liner_id=ids["liner"],
                )
                if (
                    self.cfg.pad_source.source != "gt"
                    and isinstance(self.policy, PerceptionPolicy)
                ):
                    self._apply_depth_pad_geometry(mask, ids)

        def on_semantic_labels(msg) -> None:
            # the labels JSON ('{"2":{"class":"thermalpad"}, ...}') is the
            # authoritative id map; adopt only after K consecutive repeats
            try:
                import json as _json

                data = _json.loads(msg.data)
                mapping: dict[str, int] = {}
                for k, v in data.items():
                    cls = v.get("class") if isinstance(v, dict) else v
                    if cls in ("thermalpad", "target", "liner"):
                        mapping[cls] = int(k)
                if len(mapping) != 3:
                    return
                if mapping == self._label_ids_candidate:
                    self._label_ids_count += 1
                else:
                    self._label_ids_candidate = mapping
                    self._label_ids_count = 1
                # first stable adopter wins; frozen for the episode (no
                # mid-flight retarget)
                if self._label_ids_count >= 8 and self._label_ids is None:
                    logger.info(f"semantic ids from labels (stable): {mapping}")
                    self._label_ids = mapping
            except Exception as e:  # pragma: no cover - defensive in callback
                logger.warning(f"semantic_labels parse failed: {e!r}")

        self._node.create_subscription(
            JointState, self.cfg.topics.arm_state, on_state, 10
        )
        self._node.create_subscription(
            Image, self.cfg.topics.semantic_image, on_semantic, qos_profile_sensor_data
        )
        if self.cfg.pad_source.source != "gt":
            def on_depth(msg: Image) -> None:
                import numpy as _np

                if msg.encoding in ("32FC1", "32fc1"):
                    d = _np.frombuffer(msg.data, dtype=_np.float32).astype(_np.float64)
                elif msg.encoding in ("16UC1", "mono16"):
                    d = _np.frombuffer(msg.data, dtype=_np.uint16) / 1000.0
                else:
                    logger.warning(f"unsupported depth encoding {msg.encoding!r}")
                    return
                try:
                    self._depth = d.reshape(msg.height, msg.width)
                except ValueError:
                    return

            self._node.create_subscription(
                Image, self.cfg.camera.depth_topic, on_depth,
                qos_profile_sensor_data)

        from std_msgs.msg import String as _String

        self._node.create_subscription(
            _String, "/isaac/eval_camera/semantic_labels",
            on_semantic_labels, qos_profile_sensor_data,
        )

        # DEV-ONLY GT pad centroid from record-mode /isaac/task2/pad_points;
        # layout [sim_time, n_points, x0, y0, z0, ...]
        if isinstance(self.policy, _EE_POLICIES):
            from std_msgs.msg import Float32MultiArray as _F32A

            def on_pad_points(msg) -> None:
                try:
                    d = list(msg.data)
                    if len(d) < 8:
                        return
                    pts = d[2:]
                    n3 = (len(pts) // 3) * 3
                    xs = pts[0:n3:3]
                    ys = pts[1:n3:3]
                    zs = pts[2:n3:3]
                    if xs and hasattr(self.policy, "pad_points_raw"):
                        self.policy.pad_points_raw = np.column_stack(
                            [xs, ys, zs])
                    if xs:
                        self.policy.pad_world_gt = (
                            sum(xs) / len(xs), sum(ys) / len(ys)
                        )
                        # sheet-top picks the PUSH sweep level and press
                        # depth; 99th percentile ignores flicked vertices
                        z_top = float(np.percentile(zs, 99.0))
                        self.policy.pad_gt_z_top = z_top
                        # lowest points = support surface (0.083 pedestal,
                        # ~0.009 floor); the grasp clamps against it
                        self.policy.pad_gt_z_bottom = float(
                            np.percentile(zs, 1.0))
                        # live footprint half-extents (5th..95th percentile);
                        # the pin point derives from these
                        self.policy.pad_gt_half_extent = (
                            float(np.percentile(xs, 95) - np.percentile(xs, 5)) / 2.0,
                            float(np.percentile(ys, 95) - np.percentile(ys, 5)) / 2.0,
                        )
                        # TOP SLAB mid-height (points within 8 mm of z_top);
                        # the stack's material band is the top ~5 mm
                        slab = [z for z in zs if z >= z_top - 0.008]
                        self.policy.pad_gt_z_slab_mid = (
                            sum(slab) / len(slab) if slab else None
                        )
                        # top-cluster xy centroid; regrasps aim here
                        top_xy = [
                            (x, y) for x, y, z in zip(xs, ys, zs)
                            if z >= z_top - 0.008
                        ]
                        if top_xy:
                            self.policy.pad_world_gt_top = (
                                sum(p[0] for p in top_xy) / len(top_xy),
                                sum(p[1] for p in top_xy) / len(top_xy),
                            )
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(f"pad_points parse failed: {e!r}")

            # Float32MultiArray on the benchmark contract; a type-mismatched
            # DDS subscription receives nothing
            self._node.create_subscription(
                _F32A, "/isaac/task2/pad_points", on_pad_points,
                qos_profile_sensor_data,
            )

        # left-arm plumbing: pin-and-peel (perception) or travel hold
        if (getattr(self.cfg.policy, "pin_enabled", False)
                and isinstance(self.policy, PerceptionPolicy)):
            from geometry_msgs.msg import PoseStamped as _PS

            self._left_arm_pub = self._node.create_publisher(
                JointState, "/isaac/left_joint_commands", 10)
            self._left_gripper_pub = self._node.create_publisher(
                JointState, "/isaac/left_robotiq_joint_commands", 10)

            def on_left_state(msg: JointState) -> None:
                self._left_joints = [float(p) for p in msg.position[:JOINT_COUNT]]

            self._node.create_subscription(
                JointState, "/isaac/left_joint_states", on_left_state, 10)

            def on_left_ee_pose(msg) -> None:
                if self._left_calib_done or self._left_joints is None:
                    return
                t_w_ee = pose_msg_to_matrix(
                    (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                    (msg.pose.orientation.x, msg.pose.orientation.y,
                     msg.pose.orientation.z, msg.pose.orientation.w),
                )
                q = self._left_joints
                now = self._node.get_clock().now().nanoseconds * 1e-9  # type: ignore[union-attr]
                if (
                    self._left_calib_last_q is not None
                    and max(abs(a - b) for a, b in zip(q, self._left_calib_last_q)) > 1e-4
                ):
                    self._left_calib_samples.clear()
                    self._left_calib_still_since = None
                if self._left_calib_still_since is None:
                    self._left_calib_still_since = now
                self._left_calib_last_q = list(q)
                if now - self._left_calib_still_since < 1.0:
                    return
                self._left_calib_samples.append((list(q), t_w_ee.copy()))
                if len(self._left_calib_samples) < self.cfg.calibration.samples:
                    return
                t, spread = estimate_world_to_arm_base(self._left_calib_samples)
                logger.info(
                    f"LEFT arm-base calibration applied: "
                    f"{len(self._left_calib_samples)} samples, spread {spread:.4f} m"
                )
                self.policy.left_world_to_base = np.linalg.inv(t)
                self.policy.left_arm_base_world = np.asarray(t)[:3, 3].copy()
                self._left_calib_done = True

            self._node.create_subscription(
                _PS, "/isaac/left_ee_pose", on_left_ee_pose, 10)

        # arm-base self-calibration: T_world_armbase ≈ T_world_ee × inv(FK(q)),
        # stationary samples only; no ee_pose without --record -> static fallback
        if isinstance(self.policy, _EE_POLICIES):
            from geometry_msgs.msg import PoseStamped

            def on_ee_pose(msg) -> None:
                t_w_ee = pose_msg_to_matrix(
                    (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                    (msg.pose.orientation.x, msg.pose.orientation.y,
                     msg.pose.orientation.z, msg.pose.orientation.w),
                )
                self._ee_world = t_w_ee
                self._ee_seen = True
                if not self.cfg.calibration.enabled:
                    return
                self._rearm_calibration_if_fallback()
                if self._calib_done or self._joints is None:
                    return
                # stationarity gate: max |Δq| < 0.002 for ~1 s before collecting
                now = self._node.get_clock().now().nanoseconds * 1e-9  # type: ignore[union-attr]
                q = self._joints
                if self._calib_last_q is None or max(
                    abs(a - b) for a, b in zip(q, self._calib_last_q)
                ) >= 0.002:
                    self._calib_still_since = now
                self._calib_last_q = list(q)
                if self._calib_still_since is None or now - self._calib_still_since < 1.0:
                    return
                self._calib_samples.append((list(q), t_w_ee.copy()))
                if len(self._calib_samples) < self.cfg.calibration.samples:
                    return
                t, spread = estimate_world_to_arm_base(self._calib_samples)
                if spread > self.cfg.calibration.max_translation_spread_m:
                    logger.error(
                        f"arm-base calibration looks inconsistent: translation spread "
                        f"{spread:.4f} m > {self.cfg.calibration.max_translation_spread_m} m; "
                        f"applying anyway (mount is rigid; FK error is absorbed)"
                    )
                else:
                    logger.info(
                        f"arm-base calibration applied: {len(self._calib_samples)} samples, "
                        f"translation spread {spread:.4f} m"
                    )
                self._install_world_to_base(t)
                self._calib_done = True

            self._node.create_subscription(
                PoseStamped, f"/isaac/{self.arm}_ee_pose", on_ee_pose, 10
            )

        # whole-body observation feed: same topics as
        # services/recording/record_task2.py samples
        if isinstance(self.policy, VLAPolicy):
            self._subscribe_vla_observation()

        # travel pose: raise BOTH arms (low arms block base +x); before the
        # timer so it cannot fight the policy's first commands
        if self.cfg.control.enabled and self.cfg.control.travel_pose_on_start:
            self._publish_travel_pose()

        self._calib_start_s = self._node.get_clock().now().nanoseconds * 1e-9
        self._calib_start_wall = time.monotonic()
        period = 1.0 / max(self.cfg.control.publish_rate_hz, 1e-3)
        self._timer = self._node.create_timer(period, self._tick)

    def _subscribe_vla_observation(self) -> None:
        """Feed the VLA observation collector from the recording topics;
        cameras subscribed only for the checkpoint's keys (~2.7 MB/frame)."""
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState

        from .observation import CAMERA_TOPICS, VLAObservationCollector

        self._vla_obs = VLAObservationCollector()
        node = self._node
        assert node is not None

        # whole-body checkpoints command the LEFT arm too
        if self._left_arm_pub is None:
            self._left_arm_pub = node.create_publisher(
                JointState, "/isaac/left_joint_commands", 10)
        if self._left_gripper_pub is None:
            self._left_gripper_pub = node.create_publisher(
                JointState, "/isaac/left_robotiq_joint_commands", 10)

        def on_full_states(msg: JointState) -> None:
            self._vla_obs.set_joints(msg.name, msg.position)

        node.create_subscription(
            JointState, "/isaac/joint_states_full", on_full_states, 10)

        for side in ("left", "right"):
            def on_pose(msg, side=side) -> None:
                p, o = msg.pose.position, msg.pose.orientation
                self._vla_obs.set_ee_pose(
                    side, (p.x, p.y, p.z, o.x, o.y, o.z, o.w))

            node.create_subscription(
                PoseStamped, f"/isaac/{side}_ee_pose", on_pose, 10)

        def on_odom(msg: Odometry) -> None:
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            t = msg.twist.twist
            self._vla_obs.set_odom(p.x, p.y, o.x, o.y, o.z, o.w,
                                   t.linear.x, t.linear.y, t.angular.z)

        node.create_subscription(Odometry, "/isaac/odom", on_odom, 10)

        keys = getattr(self.policy, "camera_keys", None) or list(CAMERA_TOPICS)
        for key in keys:
            topic = CAMERA_TOPICS.get(key)
            if topic is None:
                logger.warning(f"VLA camera key {key!r} has no known topic; skipped")
                continue

            def on_image(msg: Image, key=key) -> None:
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                try:
                    arr = arr.reshape(msg.height, msg.width, -1)[:, :, :3]
                except ValueError:
                    return
                self._vla_obs.set_image(key, arr)

            node.create_subscription(
                Image, topic, on_image, qos_profile_sensor_data)
        logger.info(f"VLA observation feed up: cameras={keys}")

    def _rearm_calibration_if_fallback(self) -> bool:
        """Re-arm sampling when ee_pose arrives after the static fallback went
        in, so a MEASURED transform supersedes it. True when re-armed."""
        if not (self._calib_done and self._calib_fallback):
            return False
        logger.info("ee_pose arrived after fallback install; re-arming calibration")
        self._calib_done = False
        self._calib_fallback = False
        self._calib_samples.clear()
        self._calib_last_q = None
        self._calib_still_since = None
        return True

    def _install_world_to_base(self, t_world_armbase: np.ndarray) -> None:
        """Install T_world_armbase on the policy: world_to_base for IK-frame
        goals (goal_base = world_to_base @ goal_world), arm_base_world for
        the adaptive clearance clamp."""
        if getattr(self.policy, "owns_base_transform", False):
            logger.info("policy owns its base transform; not overwriting")
            return
        self.policy.world_to_base = np.linalg.inv(t_world_armbase)
        self.policy.arm_base_world = np.asarray(t_world_armbase)[:3, 3].copy()

    def _publish_travel_pose(self, duration_s: float = 2.0, hz: float = 10.0) -> None:
        """Publish the scene ready pose to BOTH arms for ~2 s (ROS-only)."""
        from sensor_msgs.msg import JointState

        ready = [0.0, -1.5, 0.0, -2.2, 0.0, 1.5, 0.785]
        pubs = {
            side: self._node.create_publisher(  # type: ignore[union-attr]
                JointState, f"/isaac/{side}_joint_commands", 10
            )
            for side in ("left", "right")
        }
        logger.info("raising both arms to the travel (ready) pose")
        for _ in range(max(1, int(duration_s * hz))):
            for side, pub in pubs.items():
                msg = JointState()
                msg.header.stamp = self._node.get_clock().now().to_msg()  # type: ignore[union-attr]
                msg.name = [f"{side}_fr3v2_joint{i + 1}" for i in range(JOINT_COUNT)]
                msg.position = list(ready)
                pub.publish(msg)
            time.sleep(1.0 / hz)

    def _maybe_install_calibration_fallback(self, now_s: float) -> None:
        """Record-less eval: no ee_pose within calibration.timeout_s ->
        install the static fallback transform from config. WALL-time gate
        (the sim clock jumps several seconds right after a scene reset)."""
        cal = self.cfg.calibration
        if (
            not cal.enabled
            or self._calib_done
            or self._ee_seen
            or self._calib_start_wall is None
            or not isinstance(self.policy, _EE_POLICIES)
        ):
            return
        if time.monotonic() - self._calib_start_wall < cal.timeout_s:
            return
        t_wb = np.eye(4)
        t_wb[:3, 3] = np.asarray(cal.fallback_translation, dtype=np.float64)
        t_wb[:3, :3] = rpy_to_rotation(*cal.fallback_rotation_rpy)
        logger.warning(
            f"no ee_pose within {cal.timeout_s} s; installing static arm-base "
            f"fallback from config (translation {cal.fallback_translation})"
        )
        self._install_world_to_base(t_wb)
        self._calib_done = True
        self._calib_fallback = True

    # -- tick logic (testable without ROS) --------------------------------
    def tick(self, now_s: float) -> Decision | None:
        """Run one control iteration. Returns the decision (for tests) and
        publishes commands when not in dry-run and a ROS publisher exists."""
        if self._episode_start is None:
            if self.cfg.control.use_sim_time and now_s <= 0.0:
                # sim-time before the first /clock message reads 0; starting
                # now would poison every elapsed-time gate
                return Decision(Phase.WAIT_FOR_STATE, None, None,
                                "waiting for sim clock")
            self._episode_start = now_s
            try:
                self.policy.start(now_s)
            except Exception as e:
                logger.warning(f"policy.start() failed: {e}")

        # record-less eval: static-transform fallback; feed the live flange
        # pose to the ee servo
        self._maybe_install_calibration_fallback(now_s)
        if isinstance(self.policy, _EE_POLICIES):
            self.policy.ee_world = self._ee_world
        if isinstance(self.policy, PerceptionPolicy):
            # hold until a world->base transform exists (identity-frame IK
            # trips fail-closed)
            if self.cfg.calibration.enabled and not self._calib_done:
                return Decision(Phase.WAIT_FOR_STATE, None, None,
                                "waiting for arm-base calibration")
            # hold until semantic-id adoption (grace, then config fallback)
            if self._label_ids is None:
                if now_s - self._episode_start < 45.0:
                    return Decision(Phase.WAIT_FOR_STATE, None, None,
                                    "waiting for semantic-id adoption")
                if not getattr(self, "_ids_fallback_warned", False):
                    self._ids_fallback_warned = True
                    logger.warning(
                        "semantic ids not adopted within grace; using config fallback"
                    )

        # episode budget -> fail-closed stop + open gripper; mark failure on the
        # RUNNER (a non-terminal policy otherwise loops here forever)
        if now_s - self._episode_start > self.cfg.control.max_episode_s:
            if not self._budget_failed:
                logger.warning("episode budget exhausted; opening gripper and stopping")
                self._budget_failed = True
            self._publish_gripper(1.0)
            decision = Decision(Phase.FAILED, None, 1.0, "episode budget exhausted")

        # dispatch to the policy kind — NEVER after budget exhaustion: the
        # FAILED decision above must not be overwritten by a policy step
        elif isinstance(self.policy, VLAPolicy):
            state = None
            images = None
            if self._vla_obs is not None and self._vla_obs.ready:
                state = self._vla_obs.state()
                images = self._vla_obs.images
            decision = self.policy.step(
                now_s, self._joints, self._obs, state=state, images=images)
            if not self.policy.is_terminal:
                decision = self._maybe_verify_vla(decision, now_s)
        else:
            kwargs: dict[str, Any] = {}
            if hasattr(self.policy, "step") and self._mask is not None:
                # prefer the scene-instance ids from semantic_labels
                ids = self._label_ids or {
                    "thermalpad": self.cfg.semantic_ids.thermalpad,
                    "target": self.cfg.semantic_ids.target,
                    "liner": self.cfg.semantic_ids.liner,
                }
                try:
                    decision = self.policy.step(  # type: ignore[call-arg]
                        now_s, self._joints, self._obs,
                        mask=self._mask,
                        semantic_ids=ids,
                    )
                except TypeError:
                    decision = self.policy.step(now_s, self._joints, self._obs)
            else:
                decision = self.policy.step(now_s, self._joints, self._obs)

        self._apply_decision(decision, now_s)
        # photo retreat: drive home post-terminal so the eval camera sees the
        # scene unobstructed
        if self._budget_failed or (
                getattr(self.policy, "is_terminal", False) and self.policy.is_terminal):
            if self._photo_retreat_start is None:
                self._photo_retreat_start = now_s
            if now_s - self._photo_retreat_start < 8.0:
                home = self.cfg.waypoint_table.get("pregrasp")
                if home:
                    self._publish_arm(tuple(home))
                    self._publish_gripper(1.0)
            elif not self.episode_done:
                self.episode_done = True
                logger.info("episode terminal; photo retreat done — shutting down")
        # phase-change visibility: the policy itself is silent on progress
        if decision is not None and decision.phase != getattr(self, "_last_phase", None):
            logger.info(
                f"phase {getattr(self, '_last_phase', None)} -> {decision.phase.value}: "
                f"{decision.reason}"
            )
            self._last_phase = decision.phase
            if decision.phase.value in ("verify", "succeeded", "failed"):
                self._log_placement_residual()
        elif decision is not None and decision.arm_target is not None:
            self._tick_n = getattr(self, "_tick_n", 0) + 1
            if self._tick_n % 50 == 0:
                logger.info(f"  .. {decision.phase.value}: {decision.reason}")
        return decision

    def _apply_depth_pad_geometry(self, mask, ids: dict) -> None:
        """Install camera-derived pad geometry on the policy (no ground truth).
        Under ``depth_with_gt_fallback`` this overwrites the GT feed when
        depth succeeds, so the two are comparable on the same frames."""
        from .perception import estimate_pad_geometry_from_depth

        cam = self.cfg.camera
        if self._depth is None or cam.camera_height_m is None or cam.focal_px is None:
            return
        g = estimate_pad_geometry_from_depth(
            mask, self._depth,
            thermalpad_id=ids["thermalpad"], liner_id=ids["liner"],
            camera_height_m=cam.camera_height_m, focal_px=cam.focal_px,
            cx=cam.cx, cy=cam.cy,
            origin_xy=(cam.origin_x_m, cam.origin_y_m), flip_y=cam.flip_y,
        )
        if g is None:
            return
        pol = self.policy
        every = self.cfg.pad_source.compare_every
        if every and pol.pad_world_gt is not None:
            self._pad_cmp_n += 1
            if self._pad_cmp_n % every == 0:
                gx, gy = pol.pad_world_gt
                gz = pol.pad_gt_z_top
                logger.info(
                    f"pad source A/B: depth=({g.centroid_xy[0]:.4f},"
                    f"{g.centroid_xy[1]:.4f},z_top={g.z_top:.4f},"
                    f"z_bot={g.z_bottom:.4f}) gt=({gx:.4f},{gy:.4f},"
                    f"z_top={gz:.4f}) d_xy="
                    f"({(g.centroid_xy[0] - gx) * 1000:+.1f},"
                    f"{(g.centroid_xy[1] - gy) * 1000:+.1f}) mm "
                    f"d_ztop={(g.z_top - (gz or 0.0)) * 1000:+.1f} mm"
                )
        pol.pad_world_gt = g.centroid_xy
        pol.pad_world_gt_top = g.top_xy
        pol.pad_gt_z_top = g.z_top
        pol.pad_gt_z_bottom = g.z_bottom
        pol.pad_gt_z_slab_mid = g.z_top - 0.004

    def _log_placement_residual(self) -> None:
        """Report the placement miss in millimetres alongside the IoU."""
        obs = self._obs
        if obs is None or obs.placed_bbox is None or obs.target_bbox is None:
            return
        py0, py1, px0, px1 = obs.placed_bbox
        ty0, ty1, tx0, tx1 = obs.target_bbox
        d_px = ((px0 + px1) - (tx0 + tx1)) / 2.0
        d_py = ((py0 + py1) - (ty0 + ty1)) / 2.0
        scale = self.cfg.camera.pixel_scale        # px per metre
        dx_mm = d_px / scale * 1000.0
        # flip_y: image +v is world -Y (top-down camera, zero rotation)
        dy_mm = (-d_py if self.cfg.camera.flip_y else d_py) / scale * 1000.0
        official = ""
        if self._mask is not None:
            ids = self._label_ids or {
                "thermalpad": self.cfg.semantic_ids.thermalpad,
                "target": self.cfg.semantic_ids.target,
                "liner": self.cfg.semantic_ids.liner,
            }
            case, pred = predict_official_verdict(
                self._mask, thermalpad_id=ids["thermalpad"],
                target_id=ids["target"], liner_id=ids["liner"])
            official = f" official={case} pred_iou={pred:.3f}"
        logger.info(
            f"placement residual: iou={obs.iou:.3f} "
            f"liner_dom={obs.liner_dominance_ratio:.2f} "
            f"d_world=({dx_mm:+.1f},{dy_mm:+.1f}) mm  "
            f"placed_wh={px1 - px0 + 1}x{py1 - py0 + 1} px "
            f"target_wh={tx1 - tx0 + 1}x{ty1 - ty0 + 1} px" + official
        )

    def _maybe_verify_vla(self, decision: Decision, now_s: float) -> Decision:
        """Close out a VLA episode: succeed as soon as the local success gate
        passes (early stop; completion time is the tie-break), and open a bounded
        verification window when the episode budget is nearly spent."""
        assert self._episode_start is not None
        remaining = self.cfg.control.max_episode_s - (now_s - self._episode_start)
        gate = self.cfg.success_gate
        # report_verification force-opens the gripper; require the policy's
        # own release first — a carried pad passes the 2D-bbox gate too.
        released = (self._gripper_open is not None
                    and self._gripper_open >= 0.5)
        gate_passed = (
            released
            and self._obs is not None
            and self._obs.iou >= gate.min_iou
            and self._obs.liner_dominance_ratio >= gate.liner_dominance_ratio
        )
        if gate_passed or remaining <= self.cfg.control.verification_timeout_s:
            return self.policy.report_verification(self._obs, now_s)
        return decision

    def _apply_decision(self, decision: Decision, now_s: float) -> None:
        # Joint-limit gate: ALWAYS clamp into the limits and keep going;
        # NEVER open the gripper here. The demos ride exactly ON a limit, so
        # a fitted policy can predict just outside the bound; the clamped
        # target is inside the limits and max_joint_delta bounds the
        # per-tick step.
        if decision.arm_target is not None:
            limits = [tuple(p) for p in self.cfg.control.safe_joint_limits_rad]
            target = decision.arm_target
            if self.cfg.control.enabled and not _in_limits(target, limits):
                clamped = tuple(
                    min(max(q, lo), hi) for q, (lo, hi) in zip(target, limits)
                )
                excursion = max(abs(a - b) for a, b in zip(clamped, target))
                if excursion > LIMIT_CLAMP_TOL:
                    # Loud, but not fatal: log once per tick and continue.
                    logger.warning(
                        f"arm target {excursion:.4f} rad outside safe limits; "
                        f"clamping and holding (gripper untouched)"
                    )
                target = clamped
            self._publish_arm(target, spine=decision.spine)

        if decision.gripper_open_fraction is not None:
            self._gripper_open = float(decision.gripper_open_fraction)
            self._publish_gripper(self._gripper_open)

        if decision.base_twist is not None:
            self._publish_pedal(decision.base_twist)

        # left-arm command from whole-body VLA decisions, clamped into the
        # same safe limits as the active arm
        if decision.left_arm_target is not None:
            limits = [tuple(p) for p in self.cfg.control.safe_joint_limits_rad]
            left = decision.left_arm_target
            if self.cfg.control.enabled and not _in_limits(left, limits):
                left = tuple(
                    min(max(q, lo), hi) for q, (lo, hi) in zip(left, limits)
                )
            self._publish_left_arm(tuple(float(x) for x in left))
        if decision.left_gripper_open_fraction is not None:
            self._publish_left_gripper(float(decision.left_gripper_open_fraction))

        # left-arm (pin or travel-hold) command, if the policy has one
        if isinstance(self.policy, _EE_POLICIES):
            try:
                lc = self.policy.left_command(self._left_joints, now_s)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"left_command failed: {e!r}")
                lc = None
            if lc is not None:
                left_target, left_open = lc
                self._publish_left_arm(tuple(float(x) for x in left_target))
                self._publish_left_gripper(left_open)

    def _publish_left_arm(self, target: tuple[float, ...]) -> None:
        if not self.cfg.control.enabled or self.dry_run or self._left_arm_pub is None:
            return
        from sensor_msgs.msg import JointState

        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()  # type: ignore[union-attr]
        msg.name = [f"left_fr3v2_joint{i + 1}" for i in range(len(target))]
        msg.position = list(target)
        self._left_arm_pub.publish(msg)

    def _publish_left_gripper(self, open_fraction: float) -> None:
        if not self.cfg.control.enabled or self.dry_run or self._left_gripper_pub is None:
            return
        from sensor_msgs.msg import JointState

        g = self.cfg.gripper
        rad = g.open_rad + (1.0 - float(open_fraction)) * (g.closed_rad - g.open_rad)
        rad = min(max(rad, min(g.open_rad, g.closed_rad)), max(g.open_rad, g.closed_rad))
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()  # type: ignore[union-attr]
        # EBiM contract quirk: the LEFT driver joint is "left_right_finger_joint".
        msg.name = ["left_right_finger_joint"]
        msg.position = [float(rad)]
        self._left_gripper_pub.publish(msg)

    def _publish_arm(self, target: tuple[float, ...], *,
                     spine: float | None = None) -> None:
        if not self.cfg.control.enabled or self.dry_run or self._arm_pub is None:
            return
        from sensor_msgs.msg import JointState

        # single source of truth for the driver joint name
        from .mirror_lay import SPINE_JOINT

        if spine is not None:
            self._spine_cmd = float(spine)

        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()  # type: ignore[union-attr]
        msg.name = [f"{self.arm_prefix}{i + 1}" for i in range(len(target))]
        msg.position = list(target)
        # The spine rides inside the arm command message (same contract as
        # mirror_lay.send) and must be repeated every tick, not only when the
        # policy changes it — see _spine_cmd.
        if self._spine_cmd is not None:
            msg.name.append(SPINE_JOINT)
            msg.position.append(float(self._spine_cmd))
        self._arm_pub.publish(msg)

    def _publish_pedal(self, twist: tuple[float, float, float]) -> None:
        if not self.cfg.control.enabled or self.dry_run or self._pedal_pub is None:
            return
        from std_msgs.msg import String

        from .base_command import pedal_token

        msg = String()
        msg.data = pedal_token(float(twist[0]), float(twist[1]), float(twist[2]))
        # log non-STOP tokens: the eval trace shows whether the base was driven
        if msg.data != "STOP" and msg.data != getattr(self, "_last_pedal", "STOP"):
            logger.info(f"pedal token: {msg.data} (twist {twist})")
        self._last_pedal = msg.data
        self._pedal_pub.publish(msg)

    def _publish_gripper(self, open_fraction: float) -> None:
        if not self.cfg.control.enabled or self.dry_run or self._gripper_pub is None:
            return
        from sensor_msgs.msg import JointState

        # policy open-fraction (1.0 = open, 0.0 = closed) -> driver radians
        # (EBiM contract: 0.0 rad = open, 0.8 rad = closed)
        g = self.cfg.gripper
        rad = g.open_rad + (1.0 - float(open_fraction)) * (g.closed_rad - g.open_rad)
        rad = min(max(rad, min(g.open_rad, g.closed_rad)), max(g.open_rad, g.closed_rad))

        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()  # type: ignore[union-attr]
        msg.name = [self.gripper_joint]
        msg.position = [float(rad)]
        self._gripper_pub.publish(msg)

    # -- internal ROS callback alias --------------------------------------
    def _tick(self) -> None:
        now = time.time()
        if self._node is not None:
            now = self._node.get_clock().now().nanoseconds * 1e-9  # type: ignore[union-attr]
        self.tick(now)

    def destroy_node(self) -> None:
        """Best-effort teardown of the underlying rclpy node."""
        if self._node is not None:
            self._node.destroy_node()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EBiM Task 2 autonomy ROS 2 sidecar")
    p.add_argument("--config", required=True, help="path to task2 YAML config")
    p.add_argument(
        "--policy", default="waypoint",
        choices=("waypoint", "perception", "vla"),
        help="policy backend to run",
    )
    p.add_argument("--dry-run", action="store_true", help="observe + decide but publish nothing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    policy = make_policy(args.policy, cfg)
    node = Task2AutonomyNode(cfg, policy, dry_run=args.dry_run)

    if args.dry_run and not _rclpy_available():
        # dry-run without ROS still exercises the tick logic on empty state
        logger.info("dry-run without ROS: exercising tick logic with no observations")
        node._episode_start = None
        for _ in range(3):
            node.tick(time.time())
        return 0

    if not _rclpy_available():
        logger.error("rclpy is not available; cannot run the ROS node. Use --dry-run for tests.")
        return 2

    import rclpy

    node.ros_init()
    logger.info(
        f"Task 2 autonomy node up: policy={args.policy} arm={cfg.control.active_arm} "
        f"enabled={cfg.control.enabled} dry_run={args.dry_run}"
    )
    try:
        while rclpy.ok():
            rclpy.spin_once(node._node, timeout_sec=0.2)  # type: ignore[arg-type]
            if node.episode_done:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def _rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401

        return True
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
