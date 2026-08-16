"""Perception- and VLA-driven policies implementing the same ``Decision`` contract.

* :class:`PerceptionPolicy` — semantic-mask target estimate -> IK -> joint
  interpolation; fully-offline baseline, no neural network.
* :class:`VLAPolicy` — wraps a Foundation VLA (Pi0.5 / OpenVLA family)
  proposing joint-delta action chunks; lazily loaded via :meth:`VLAPolicy.load`,
  hold/identity decisions until then (offline without torch or weights).

Both expose ``step(now_s, joints, observation, mask=None, *, ...)`` returning a
:class:`Decision`, so the runner can switch between them with ``--policy``.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("ebim_task2.vla")  # type: ignore[assignment]

from .motion import IKResult, interpolate_waypoints, resolve_solver
from .perception import (
    PlacementObservation,
    TargetPose,
    estimate_pad_pose,
    estimate_target_pose,
)
from .policy import Decision, Phase


# ---------------------------------------------------------------------------
# Perception-driven policy
# ---------------------------------------------------------------------------
class PerceptionPolicy:
    """Geometric perception -> IK -> joint interpolation policy.

    State machine: WAIT -> MOVE_ABOVE (pre-grasp height) -> DESCEND -> GRASP
    -> LIFT -> MOVE_TO_TARGET -> DESCEND_PLACE -> RELEASE -> VERIFY.

    With :attr:`world_to_base` installed (see :mod:`ebim_task2.calibration`),
    camera-derived goals are WORLD-frame (table plane z = 0.75) and are
    transformed into arm-base before IK; ``None`` uses them as-is (offline).
    :attr:`ee_world` drives a position-Jacobian servo after GRASP/PLACE;
    :attr:`arm_base_world` clamps clearance z within 0.76 m of the arm base
    (FR3 reach 0.855 m).
    """

    #: 4x4 world->arm-base transform from runtime self-calibration, or None.
    world_to_base: np.ndarray | None = None
    #: 4x4 flange (link8) world pose, refreshed by the runner every tick.
    ee_world: np.ndarray | None = None
    #: Arm-base world position (3-vector), refreshed on (re)calibration.
    arm_base_world: np.ndarray | None = None
    #: DEV-ONLY: GT sheet world xy from /isaac/task2/pad_points; PUSH planner input.
    pad_world_gt: tuple[float, float] | None = None
    #: DEV-ONLY: highest sheet z; picks the contact sweep height.
    pad_gt_z_top: float | None = None
    #: DEV-ONLY: lowest sheet z, its resting surface (0.083 pedestal, ~0.009 floor).
    pad_gt_z_bottom: float | None = None
    #: DEV-ONLY: mean z of top-slab points (within 8 mm of z_top); PUSH sweep height.
    pad_gt_z_slab_mid: float | None = None
    #: DEV-ONLY: xy centroid of top-cluster points (within 8 mm of z_top); regrasp aim.
    pad_world_gt_top: tuple[float, float] | None = None
    #: DEV/vision: sheet footprint half-extents (x, y), 5th..95th percentile.
    pad_gt_half_extent: tuple[float, float] | None = None

    def __init__(
        self,
        *,
        joint_tolerance_rad: float = 0.05,
        waypoint_timeout_s: float = 15.0,
        verification_timeout_s: float = 6.0,
        min_iou: float = 0.85,
        liner_dominance_ratio: float = 0.90,
        safe_joint_limits: Sequence[tuple[float, float]] | None = None,
        home_joints: Sequence[float] | None = None,
        grasp_height: float = 0.02,
        place_height: float | None = None,
        clearance_height: float = 0.18,
        cam_cx: float = 320.0,
        cam_cy: float = 240.0,
        pixel_scale: float = 1000.0,
        camera_height_m: float | None = None,
        pixel_scale_plane_z_m: float = 0.10,
        pad_plane_z_m: float = 0.10,
        target_plane_z_m: float = 0.001,
        origin_xy: tuple[float, float] = (0.0, 0.0),
        flip_y: bool = False,
        servo_iters: int = 3,
        servo_tol_m: float = 0.006,
        release_dwell_s: float = 1.2,
        release_press_m: float = 0.004,
        release_wipe_m: float = 0.02,
        release_lift_m: float = 0.06,
        release_slow_step_rad: float = 0.015,
        release_timeout_s: float = 25.0,
        release_twist_rad: float = 0.0,
        release_shake_cycles: int = 0,
        release_shake_amp_m: float = 0.04,
        release_tilt_rad: float = 0.0,
        release_dump_enabled: bool = True,
        push_enabled: bool = False,
        push_height_m: float = 0.02,
        push_settle_s: float = 5.0,
        push_standoff_m: float = 0.08,
        push_stop_short_m: float = 0.055,
        whip_offset_xy: Sequence[float] = (0.0, 0.0),
        pin_enabled: bool = False,
        pin_offset_xy: Sequence[float] = (0.012, -0.038),
        pin_press_z: float = 0.008,
        pin_tilt_rad: float = 0.6,
        pin_clear_z: float = 0.18,
        retreat_joints: Sequence[float] | None = None,
        grasp_close_dwell_s: float = 1.5,
        servo_settle_tol_rad: float = 0.03,
        servo_settle_timeout_s: float = 6.0,
        transport_step_rad: float = 0.05,
        grasp_mode: str = "pinch",
        press_double_tap: bool = False,
        press_double_tap_lift_m: float = 0.02,
        press_slow_final_m: float = 0.0,
        grasp_depth_m: float = 0.008,
        grasp_support_margin_m: float = 0.0,
        grasp_close_fraction: float = 0.0,
        grasp_yaw_offset_rad: float = 0.0,
        grasp_depth_step_m: float = 0.002,
        grasp_depth_max_steps: int = 2,
    ) -> None:
        self._joint_tol = joint_tolerance_rad
        self._wp_timeout = waypoint_timeout_s
        self._verify_timeout = verification_timeout_s
        self._min_iou = min_iou
        self._min_liner = liner_dominance_ratio
        self._limits = [tuple(p) for p in (safe_joint_limits or [])]
        # A neutral FR3 home pose.
        self._home = np.asarray(
            home_joints if home_joints is not None
            else [0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8],
            dtype=np.float64,
        )
        self._grasp_h = grasp_height
        # place height defaults to grasp height (barebone: pad top 0.10, target top 0.002)
        self._place_h = place_height if place_height is not None else grasp_height
        self._clear_h = clearance_height
        # pixel_scale = px/m at table plane; (cam_cx, cam_cy) principal point;
        # origin_xy in m; flip_y: image +v -> world -Y
        self._cam_cx = cam_cx
        self._cam_cy = cam_cy
        self._pixel_scale = pixel_scale
        # px/m scales as fx/(camera_z-object_z)
        self._cam_height = camera_height_m
        self._scale_plane_z = pixel_scale_plane_z_m
        self._pad_plane_z = pad_plane_z_m
        self._target_plane_z = target_plane_z_m
        self._origin_xy = tuple(origin_xy)
        self._flip_y = flip_y
        self.world_to_base = None
        self.ee_world = None
        self.arm_base_world = None
        # ee servo: dq = pinv(J_pos) @ (T_bw[:3,:3] @ err_world), clip +-0.15 rad
        self._servo_iters = servo_iters
        self._servo_tol = servo_tol_m
        # release-peel: the tacky liner otherwise drags the pad off on retreat
        self._release_dwell_s = release_dwell_s
        self._release_press = release_press_m
        self._release_wipe = release_wipe_m
        self._release_lift = release_lift_m
        self._release_slow_step = release_slow_step_rad
        self._release_timeout = release_timeout_s
        # twist-while-pressed shears the finger bond; symmetric +θ/-θ keeps orientation
        self._release_twist = release_twist_rad
        # vertical shake: slow peels never break adhesion, fast swings do; vertical only
        self._release_shake_cycles = release_shake_cycles
        self._release_shake_amp = release_shake_amp_m
        # dump gate: wrist-flip dump used only as detach escalation
        self._release_dump_enabled = release_dump_enabled
        # tilt-dump: ~110 deg tool tilt at rise height, jaws open
        self._release_tilt = release_tilt_rad
        # floor-push recovery: re-locate the dropped sheet, slide it edge-on
        self._push_enabled = push_enabled
        self._push_height = push_height_m
        self._push_settle_s = push_settle_s
        self._push_standoff = push_standoff_m
        self._push_stop_short = push_stop_short_m
        # whip fling offset compensation: aim so the fling lands on target
        self._whip_offset = (float(whip_offset_xy[0]), float(whip_offset_xy[1]))
        # pin-and-peel: LEFT fingertip pins the sheet during the whip, launch dir
        # ~(+0.42,-0.90); left tool tilted for palm clearance
        self._pin_enabled = pin_enabled
        self._pin_offset = (float(pin_offset_xy[0]), float(pin_offset_xy[1]))
        self._pin_press_z = pin_press_z
        self._pin_tilt = pin_tilt_rad
        self._pin_clear_z = pin_clear_z
        self.left_world_to_base: np.ndarray | None = None
        self.left_arm_base_world: np.ndarray | None = None
        self._left_traj: list[np.ndarray] = []
        self._left_idx = 0
        # idle -> descend -> hold -> lift -> done (idle/done: nothing to publish)
        self._left_state = "idle"
        # clear-height IK solution of the pin descent (see _pin_release)
        self._left_clear_q: np.ndarray | None = None
        self._left_last_cmd: np.ndarray | None = None
        self._left_state_since: float | None = None
        self._push_planned = False
        self._pushes_done = 0
        self._max_pushes = 6
        # Per-stroke contact-dwell budget (see the PUSH stall guard).
        self._push_dwells = 0
        # Attach-verification regrasp budget (see _transition).
        self._regrasps_done = 0
        # detach-verification re-shake budget (see _transition)
        self._reshakes_done = 0
        self._last_push_sheet: tuple[float, float] | None = None
        self._last_push_z: float | None = None
        # contact-stall guard: a jammed push stroke never reaches waypoint tol
        self._stall_idx = -1
        self._stall_since: float | None = None
        self._live_pad: TargetPose | None = None
        self._live_target: TargetPose | None = None
        # tucked observation pose: raises the arm out of the top-down camera view
        self._tuck_joints = np.array([0.0, -1.5, 0.0, -2.2, 0.0, 1.5, 0.785])
        # press-mode release opens the jaws after the slow rise; early splaying drags the sheet
        self._release_open_after_idx = 0
        self._retreat_joints = (
            np.asarray(retreat_joints, dtype=np.float64)
            if retreat_joints is not None and len(retreat_joints) == 7 else None
        )
        # grasp-close dwell: jaws open through descent, close at the grasp pose
        self._grasp_close_dwell_s = grasp_close_dwell_s
        # ee-servo settle gate: iterate only after the tracked joints settle
        self._servo_settle_tol = servo_settle_tol_rad
        self._servo_settle_timeout = servo_settle_timeout_s
        self._servo_last_cmd: np.ndarray | None = None
        self._servo_settle_since: float | None = None
        # pad-carrying phases use finer joint steps
        self._transport_step = transport_step_rad
        # "pinch" = descend open, close at depth; "press" = closed-jaw
        # fingertip press
        self._grasp_mode = grasp_mode
        # press double-tap: lift ~2 cm, press again, slow final
        self._press_double_tap = press_double_tap
        self._press_double_tap_lift = press_double_tap_lift_m
        self._press_slow_final = press_slow_final_m
        # depth schedule below the GT sheet top (see PolicyConfig.grasp_depth_m)
        self._grasp_depth = grasp_depth_m
        # fingertip clearance above the sheet's support (open cage below it shovels the stack)
        self._grasp_support_margin = grasp_support_margin_m
        # Carry grip width (see PolicyConfig.grasp_close_fraction).
        self._grasp_close_fraction = grasp_close_fraction
        # Jaw-closing axis convention (see PolicyConfig.grasp_yaw_offset_rad).
        self._grasp_yaw_offset = grasp_yaw_offset_rad
        self._grasp_depth_step = grasp_depth_step_m
        self._grasp_depth_max_steps = grasp_depth_max_steps

        self._phase = Phase.WAIT_FOR_STATE
        self._trajectory: list[np.ndarray] = []
        self._traj_idx = 0
        self._phase_start: float | None = None
        self._verify_start: float | None = None
        self._have_object = False
        self._servo_left = 0
        self._servo_goal_world: np.ndarray | None = None
        self._place_goal_world: np.ndarray | None = None
        # pose locks: camera is occluded when a phase needs its goal; scene is static
        self._pad_locked: TargetPose | None = None
        self._target_locked: TargetPose | None = None
        # PLACE goal shift by the locked (pad-only - attach-point) px offset
        self._pad_only_locked: tuple[float, float] | None = None
        # GRASP final press-depth target; the dwell must keep commanding it
        # (reached-gate 0.08 rad ~ cm-level TCP slack)
        self._press_hold_q: np.ndarray | None = None
        # press telemetry (sampled through the grasp-close dwell)
        self._press_tcp_min: float | None = None
        self._press_ztop0: float | None = None

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def is_terminal(self) -> bool:
        return self._phase in (Phase.SUCCEEDED, Phase.FAILED)

    def start(self, now_s: float) -> None:
        self._phase = Phase.WAIT_FOR_STATE
        self._trajectory = []
        self._traj_idx = 0
        self._phase_start = now_s
        self._verify_start = None
        self._have_object = False
        self._servo_left = 0
        self._servo_goal_world = None
        self._servo_last_cmd = None
        self._place_goal_world = None
        self._pad_locked = None
        self._target_locked = None
        self._pad_only_locked = None
        self._push_planned = False
        self._pushes_done = 0
        self._regrasps_done = 0
        self._last_push_sheet = None
        self._last_push_z = None
        self._live_pad = None
        self._press_hold_q = None
        self._press_tcp_min = None
        self._press_ztop0 = None

    # -- main loop --------------------------------------------------------
    def step(
        self,
        now_s: float,
        joints: Sequence[float] | None,
        observation: PlacementObservation | None,
        mask: np.ndarray | None = None,
        *,
        target_pose: TargetPose | None = None,
        semantic_ids: dict[str, int] | None = None,
    ) -> Decision:
        if joints is None and self._phase == Phase.WAIT_FOR_STATE:
            return Decision(Phase.WAIT_FOR_STATE, None, None, "waiting for joint state")

        cur = np.asarray(joints, dtype=np.float64) if joints is not None else self._home

        # resolve target (placement) and pad (grasp point) poses from the mask
        pad_pose: TargetPose | None = None
        if mask is not None and semantic_ids:
            if target_pose is None:
                target_pose = estimate_target_pose(
                    mask, target_id=semantic_ids["target"], liner_id=semantic_ids["liner"]
                )
            pad_pose = estimate_pad_pose(
                mask, thermalpad_id=semantic_ids["thermalpad"], liner_id=semantic_ids["liner"]
            )

        # the PUSH planner needs the LIVE estimates (locks are frozen pre-grasp)
        self._live_pad = pad_pose
        self._live_target = target_pose

        # occlusion-safe pose locks (see __init__)
        if (
            pad_pose is not None and pad_pose.visible and self._pad_locked is None
            and self._phase in (Phase.WAIT_FOR_STATE, Phase.PREGRASP)
        ):
            self._pad_locked = pad_pose
            # also lock the THERMALPAD-only centroid for delivery-offset comp
            if mask is not None and semantic_ids:
                from .perception import _centroid_axis

                axis = _centroid_axis(np.asarray(mask) == semantic_ids["thermalpad"])
                if axis is not None:
                    self._pad_only_locked = (axis[0], axis[1])
        if (
            target_pose is not None and target_pose.visible
            and self._phase in (Phase.WAIT_FOR_STATE, Phase.PREGRASP, Phase.GRASP, Phase.LIFT)
        ):
            self._target_locked = target_pose
        if self._pad_locked is not None:
            pad_pose = self._pad_locked
        if self._target_locked is not None:
            target_pose = self._target_locked

        if self._phase == Phase.WAIT_FOR_STATE:
            if self._grasp_mode == "push_only":
                # push_only: the pad lock anchors the pedestal for the clearing stroke
                if not self._push_enabled:
                    self._fail("push_only requires push_enabled")
                else:
                    self._phase = Phase.PUSH
                    self._phase_start = now_s
                    self._trajectory = []
                    self._traj_idx = 0
                    self._push_planned = False
            else:
                self._begin_phase(Phase.PREGRASP, now_s, cur, target_pose, pad_pose)

        if self.is_terminal:
            return Decision(self._phase, None, 1.0 if self._phase == Phase.FAILED else 0.0,
                            f"terminal {self._phase.value}")

        if self._phase == Phase.VERIFY:
            return self._step_verify(now_s, observation)

        # release dwell: hold at place pose; pinch opens here, press opens after the rise
        if (
            self._phase == Phase.RELEASE and self._phase_start is not None
            and ((now_s - self._phase_start) < self._release_dwell_s
                 or self._left_state == "descend")
        ):
            # pin mode: dwell also waits for the left fingertip (left_command
            # times out into "hold", so no deadlock)
            g = 0.0 if self._grasp_mode == "press" else 1.0
            return Decision(Phase.RELEASE, tuple(float(x) for x in cur), g,
                            "release dwell"
                            + (" (pin descending)" if self._left_state == "descend"
                               else ""))

        # grasp-close dwell: pinch at depth; press mode holds _press_hold_q
        if (
            self._phase == Phase.LIFT and self._phase_start is not None
            and (now_s - self._phase_start) < self._grasp_close_dwell_s
        ):
            hold = self._press_hold_q if (
                self._grasp_mode == "press" and self._press_hold_q is not None
            ) else cur
            # press telemetry: lowest TCP z + sheet z_top drift
            if self._grasp_mode == "press":
                if self.ee_world is not None:
                    ee = np.asarray(self.ee_world, dtype=np.float64)
                    # calibrated tool offset (motion._TCP_OFFSET_Z), not the
                    # 0.15 nominal: 11 mm stale = 11 mm phantom clearance
                    from .motion import _TCP_OFFSET_Z

                    tcp_z = float(
                        ee[2, 3] + (ee[:3, :3] @ np.array([0.0, 0.0, _TCP_OFFSET_Z]))[2]
                    )
                    self._press_tcp_min = (
                        tcp_z if self._press_tcp_min is None
                        else min(self._press_tcp_min, tcp_z)
                    )
                if self.pad_gt_z_top is not None and self._press_ztop0 is None:
                    self._press_ztop0 = float(self.pad_gt_z_top)
            return Decision(Phase.LIFT, tuple(float(x) for x in hold), 0.0,
                            "grasp close dwell (jaws closing)")
        if (
            self._phase == Phase.LIFT
            and self._grasp_mode == "press"
            and self._press_ztop0 is not None
        ):
            # dwell just ended: one telemetry line per press attempt
            logger.warning(
                f"press telemetry: tcp_z_min="
                f"{self._press_tcp_min if self._press_tcp_min is not None else float('nan'):.3f}, "
                f"z_top {self._press_ztop0:.3f} -> "
                f"{float(self.pad_gt_z_top) if self.pad_gt_z_top is not None else float('nan'):.3f}"
            )
            self._press_tcp_min = None
            self._press_ztop0 = None

        # engaged left pin lifts clear once on leaving PLACE/RELEASE
        if (
            self._phase not in (Phase.PLACE, Phase.RELEASE)
            and self._left_state in ("descend", "hold")
        ):
            self._pin_release()

        # PUSH: wait for the sheet to settle, then plan from the LIVE mask estimate
        if self._phase == Phase.PUSH and not self._push_planned:
            if (
                self._phase_start is not None
                and ((now_s - self._phase_start) < self._push_settle_s
                     or self._left_state == "lift")
            ):
                return Decision(Phase.PUSH, tuple(float(x) for x in cur), 0.0,
                                "push settle (sheet landing)"
                                + (" (left lifting)" if self._left_state == "lift"
                                   else ""))
            self._push_planned = True
            self._pushes_done += 1
            # prefer the LIVE target estimate over the frozen lock
            tgt = (
                self._live_target
                if self._live_target is not None and self._live_target.visible
                else self._target_locked
            )
            traj = self._build_push_trajectory(cur, self._live_pad, tgt)
            if traj is None:
                if (
                    (self._live_pad is None or not self._live_pad.visible)
                    and self._pushes_done < self._max_pushes
                ):
                    # sheet occluded: tuck; next round re-locates with a clean mask
                    self._trajectory = interpolate_waypoints(
                        cur, self._tuck_joints, max_joint_step=0.1)
                    self._traj_idx = 0
                    return Decision(Phase.PUSH, tuple(float(x) for x in cur), 0.0,
                                    "sheet occluded -> tucking to re-locate")
                self._phase = Phase.VERIFY
                self._verify_start = None
                self._phase_start = now_s
                return Decision(Phase.VERIFY, None, 1.0, "push unplannable -> verifying")
            self._push_dwells = 0
            self._trajectory = traj
            self._traj_idx = 0

        if self._traj_idx >= len(self._trajectory):
            servo = self._maybe_servo(cur, now_s)
            if servo is not None:
                return servo
            return self._transition(now_s, cur, observation, target_pose, pad_pose)

        # contact-stall guard (PUSH only): no waypoint progress for 30 sim-s -> replan
        if self._phase == Phase.PUSH and self._trajectory:
            if self._stall_idx != self._traj_idx:
                self._stall_idx = self._traj_idx
                self._stall_since = now_s
            elif (
                self._stall_since is not None
                and now_s - self._stall_since > 6.0
                and self._push_dwells < 6
            ):
                # contact dwell: advance the waypoint and keep sweeping; budget (6) = jam
                self._push_dwells += 1
                self._traj_idx += 1
                self._stall_idx = self._traj_idx
                self._stall_since = now_s
                if self._traj_idx >= len(self._trajectory):
                    return self._transition(now_s, cur, observation,
                                            target_pose, pad_pose)
            elif (
                self._stall_since is not None
                and now_s - self._stall_since > 30.0
            ):
                self._stall_idx = -1
                self._stall_since = None
                if self._pushes_done < self._max_pushes:
                    logger.warning("push stroke stalled; re-locating early")
                    self._phase_start = now_s
                    self._trajectory = []
                    self._traj_idx = 0
                    self._push_planned = False
                    return Decision(Phase.PUSH, tuple(float(x) for x in cur), 0.0,
                                    "push stalled -> re-locating")
                self._phase = Phase.VERIFY
                self._verify_start = None
                self._phase_start = now_s
                return Decision(Phase.VERIFY, None, 1.0,
                                "push stalled -> verifying")

        # timeout guard; RELEASE/PUSH timeouts go to VERIFY, never FAIL (pad already down)
        timeout = (
            self._release_timeout
            if self._phase in (Phase.RELEASE, Phase.PUSH) else self._wp_timeout
        )
        if self._phase_start is not None and (now_s - self._phase_start) > timeout:
            if self._phase == Phase.PUSH and self._pushes_done < self._max_pushes:
                self._phase_start = now_s
                self._trajectory = []
                self._traj_idx = 0
                self._push_planned = False
                return Decision(Phase.PUSH, tuple(float(x) for x in cur), 0.0,
                                "push timeout -> re-locating")
            if self._phase in (Phase.RELEASE, Phase.PUSH):
                timed_out = self._phase.value
                self._phase = Phase.VERIFY
                self._verify_start = None
                self._phase_start = now_s
                return Decision(Phase.VERIFY, None, 1.0,
                                f"{timed_out} timeout -> verifying")
            return self._fail(f"{self._phase.value} timeout")

        target = self._trajectory[self._traj_idx]
        if self._reached(cur, target):
            self._traj_idx += 1
            if self._traj_idx >= len(self._trajectory):
                servo = self._maybe_servo(cur, now_s)
                if servo is not None:
                    return servo
                return self._transition(now_s, cur, observation, target_pose, pad_pose)
            target = self._trajectory[self._traj_idx]
        gripper = self._gripper_for(self._phase)
        if self._phase == Phase.RELEASE and self._grasp_mode == "press":
            # late open: jaws closed through press + rise, open post-rise
            gripper = 1.0 if self._traj_idx >= self._release_open_after_idx else 0.0
        return Decision(self._phase, tuple(float(x) for x in target), gripper,
                        f"{self._phase.value} -> {self._traj_idx}/{len(self._trajectory)}")

    # -- phase construction ----------------------------------------------
    def _begin_phase(self, phase: Phase, now_s: float, cur: np.ndarray,
                     target: TargetPose | None,
                     pad: TargetPose | None = None) -> None:
        self._phase = phase
        self._phase_start = now_s
        self._traj_idx = 0
        self._trajectory = []
        # ee servo refines GRASP/PLACE only; never press-mode GRASP
        self._servo_left = (
            self._servo_iters
            if phase in (Phase.GRASP, Phase.PLACE)
            and not (phase == Phase.GRASP and self._grasp_mode == "press")
            else 0
        )
        self._servo_goal_world = None
        self._servo_last_cmd = None
        if phase == Phase.RELEASE:
            # plan the left pin descent first; the release dwell holds the right arm
            self._plan_pin()
            traj = self._build_release_trajectory(cur, target, pad)
        else:
            traj = self._build_trajectory(phase, cur, target, pad)
        if phase == Phase.GRASP:
            # fresh press telemetry for this attempt (see _press_tcp_min)
            self._press_tcp_min = None
            self._press_ztop0 = None
        if traj is None:
            self._fail(f"ik failed for {phase.value}")
            return
        self._trajectory = traj

    def _build_trajectory(self, phase: Phase, cur: np.ndarray,
                          target: TargetPose | None,
                          pad: TargetPose | None = None) -> list[np.ndarray] | None:
        """Plan the joint-space trajectory for the requested Cartesian phase.

        Returns None when IK fails (caller fails closed); a near-solution
        (pos_error <= 2 cm) is accepted — the ee servo refines the pose."""
        goal_world = self._cartesian_goal(phase, target, pad)
        if goal_world is None:
            return [cur]
        # world TCP goal for the ee servo; PLACE's goal also anchors the release-peel
        self._servo_goal_world = goal_world
        if phase == Phase.PLACE:
            self._place_goal_world = np.asarray(goal_world, dtype=np.float64).copy()
        if phase == Phase.GRASP and self._grasp_mode == "press":
            return self._build_press_trajectory(cur, goal_world)
        if phase == Phase.LIFT and self._grasp_mode == "press":
            return self._build_lift_trajectory(cur, goal_world)
        goal_pose = goal_world
        if self.world_to_base is not None:
            # camera-derived goal is WORLD-frame; IK runs in arm-base
            goal_pose = np.asarray(self.world_to_base) @ goal_world
        solver = resolve_solver()
        res: IKResult = solver(goal_pose, cur, safe_joint_limits=self._limits)
        if not res.success:
            if res.pos_error > 0.02:
                logger.warning(
                    f"ik failed for {phase.value}: pos_error={res.pos_error:.4f}"
                )
                return None
            logger.warning(
                f"ik near-solution for {phase.value} accepted: "
                f"pos_error={res.pos_error:.4f} (servo will refine)"
            )
        step = (
            self._transport_step
            if phase in (Phase.LIFT, Phase.PREPLACE, Phase.PLACE) else 0.05
        )
        return interpolate_waypoints(cur, res.q, max_joint_step=step)

    def _build_press_trajectory(self, cur: np.ndarray,
                                goal_world: np.ndarray) -> list[np.ndarray] | None:
        """Press-mode GRASP descent: fast approach, slow final seat at the
        transport step, optional double-tap. Returns None when a segment's IK
        fails; near-solutions (<=2 cm) accepted."""
        depth = np.asarray(goal_world, dtype=np.float64)
        approach = self._press_slow_final
        if self._press_double_tap:
            approach = max(approach, self._press_double_tap_lift)
        above = depth.copy()
        above[2, 3] = depth[2, 3] + approach
        # (segment pose, max joint step)
        segments: list[tuple[np.ndarray, float]] = [(above, 0.05)]
        if self._press_slow_final > 0.0:
            # re-solve IK every few mm along the slow final descent (a coarse
            # joint lerp bows the tool sideways)
            n = max(1, int(round(self._press_slow_final / 0.006)))
            for i in range(1, n + 1):
                mid = depth.copy()
                mid[2, 3] = above[2, 3] + (depth[2, 3] - above[2, 3]) * i / n
                segments.append((mid, self._transport_step))
        if self._press_double_tap:
            lift = depth.copy()
            lift[2, 3] = depth[2, 3] + self._press_double_tap_lift
            segments.append((lift, self._transport_step))
            segments.append((depth.copy(), self._transport_step))
        solver = resolve_solver()
        traj: list[np.ndarray] = []
        q = np.asarray(cur, dtype=np.float64)
        for goal_w, max_step in segments:
            goal_pose = goal_w
            if self.world_to_base is not None:
                goal_pose = np.asarray(self.world_to_base) @ goal_w
            res: IKResult = solver(goal_pose, q, safe_joint_limits=self._limits)
            if not res.success:
                if res.pos_error > 0.02:
                    logger.warning(
                        f"press segment ik failed (pos_error={res.pos_error:.4f})"
                    )
                    return None
                logger.warning(
                    f"press segment ik near-solution accepted: "
                    f"pos_error={res.pos_error:.4f} (servo will refine)"
                )
            traj.extend(interpolate_waypoints(q, res.q, max_joint_step=max_step))
            q = np.asarray(res.q, dtype=np.float64)
        return traj if traj else None

    def _build_lift_trajectory(self, cur: np.ndarray,
                               goal_world: np.ndarray) -> list[np.ndarray] | None:
        """Press-mode LIFT: creep the first ~2 cm at a very fine step, then
        rise at the transport step."""
        from .motion import franka_fk

        goal = np.asarray(goal_world, dtype=np.float64)
        creep = goal.copy()
        cur_z = float(franka_fk(list(cur))[2, 3])
        creep[2, 3] = min(goal[2, 3], cur_z + 0.02)
        solver = resolve_solver()
        traj: list[np.ndarray] = []
        q = np.asarray(cur, dtype=np.float64)
        for goal_w, step in ((creep, 0.006), (goal, self._transport_step)):
            goal_pose = goal_w
            if self.world_to_base is not None:
                goal_pose = np.asarray(self.world_to_base) @ goal_w
            res: IKResult = solver(goal_pose, q, safe_joint_limits=self._limits)
            if not res.success:
                if res.pos_error > 0.02:
                    logger.warning(
                        f"lift segment ik failed (pos_error={res.pos_error:.4f})"
                    )
                    return None
                logger.warning(
                    f"lift segment ik near-solution accepted: "
                    f"pos_error={res.pos_error:.4f}"
                )
            traj.extend(interpolate_waypoints(q, res.q, max_joint_step=step))
            q = np.asarray(res.q, dtype=np.float64)
        return traj if traj else None

    def _build_release_trajectory(self, cur: np.ndarray,
                                  target: TargetPose | None,
                                  pad: TargetPose | None = None) -> list[np.ndarray]:
        """Release-peel trajectory (the tacky liner otherwise drags the pad
        back off on retreat): press -> wipe -> slow rise -> clearance ->
        optional retreat to ``retreat_joints``. Failed-IK segments are
        skipped, not fail-closed (partial peel beats FAILED)."""
        anchor = self._place_goal_world
        if anchor is None:
            anchor_goal = self._cartesian_goal(Phase.PLACE, target, pad)
            anchor = None if anchor_goal is None else np.asarray(anchor_goal)
        if anchor is None:
            # no place reference: single-goal clearance retreat
            legacy = self._build_trajectory(Phase.RELEASE, cur, target, pad)
            return legacy if legacy is not None else [np.asarray(cur, dtype=np.float64)]
        anchor = np.asarray(anchor, dtype=np.float64)

        press = anchor.copy()
        press[2, 3] = anchor[2, 3] - self._release_press
        rise = press.copy()
        rise[2, 3] = anchor[2, 3] + self._release_lift
        # (pose, max joint step, open_here): open_here marks the first segment
        # where press-mode jaws may open
        segments: list[tuple[np.ndarray, float, bool]] = []
        # twist segments need orientation-converged IK (position-only would no-op)
        twist_segments: list[np.ndarray] = []
        if self._grasp_mode == "press":
            # max_joint_step=10 emits ONE waypoint (raw un-interpolated jump)
            segments.append((press, self._release_slow_step, False))
            if self._release_twist > 0.0:
                # twist-while-pressed at the slot; symmetric
                c, s = math.cos(self._release_twist), math.sin(self._release_twist)
                rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                twist = press.copy()
                twist[:3, :3] = press[:3, :3] @ rz
                segments.append((twist, self._release_slow_step, False))
                twist_segments.append(twist)
                segments.append((press.copy(), self._release_slow_step, False))
            segments.append((rise, self._release_slow_step, False))
            if self._release_shake_cycles > 0:
                high = rise.copy()
                high[2, 3] = max(anchor[2, 3] + 0.25, rise[2, 3] + 0.15)
                for _ in range(self._release_shake_cycles):
                    segments.append((high, 10.0, True))
                    segments.append((rise.copy(), 10.0, True))
            if self._release_tilt > 0.0:
                # pour-off: tilt about the tool x-axis with the jaws open
                c, s = math.cos(self._release_tilt), math.sin(self._release_tilt)
                rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
                tilt = rise.copy()
                tilt[:3, :3] = rise[:3, :3] @ rx
                segments.append((tilt, 0.08, True))
                segments.append((rise.copy(), 0.08, True))
        else:
            segments.append((press, self._release_slow_step, True))
            if self._release_twist > 0.0:
                # twist-while-pressed: tangential shear (symmetric)
                c, s = math.cos(self._release_twist), math.sin(self._release_twist)
                rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                twist = press.copy()
                twist[:3, :3] = press[:3, :3] @ rz
                segments.append((twist, self._release_slow_step, True))
                segments.append((press.copy(), self._release_slow_step, True))
            for _ in range(self._release_shake_cycles):
                up = press.copy()
                up[2, 3] = press[2, 3] + self._release_shake_amp
                segments.append((up, 0.3, True))
                segments.append((press.copy(), 0.3, True))
            if self._release_wipe > 0.0:
                wipe = press.copy()
                # lateral scrub along the tool x-axis
                wipe[:3, 3] = wipe[:3, 3] + press[:3, 0] * self._release_wipe
                segments.append((wipe, self._release_slow_step, True))
            if self._release_tilt > 0.0:
                # pour-off: jaws close to ~0.82 rad max (> the 2 cm pad), so the
                # pinch is a scoop; tip the fingers to pour the pad off
                c, s_ = math.cos(self._release_tilt), math.sin(self._release_tilt)
                rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])
                tilt = press.copy()
                tilt[:3, :3] = press[:3, :3] @ rx
                segments.append((tilt, self._release_slow_step, True))
                segments.append((press.copy(), self._release_slow_step, True))
            segments.append((rise.copy(), self._release_slow_step, True))
        if self._grasp_mode != "press":
            # pinch mode still clears out before VERIFY
            clear = rise.copy()
            clear[2, 3] = max(self._clear_h, rise[2, 3])
            if self.arm_base_world is not None:
                # same adaptive-reach clamp as _cartesian_goal
                ab = np.asarray(self.arm_base_world, dtype=np.float64)
                dx = clear[0, 3] - ab[0]
                dy = clear[1, 3] - ab[1]
                dz = math.sqrt(max(0.76 * 0.76 - dx * dx - dy * dy, 0.01))
                clear[2, 3] = min(clear[2, 3], ab[2] + dz)
            segments.append((clear, 0.05, True))
        solver = resolve_solver()
        traj: list[np.ndarray] = []
        open_idx: int | None = None
        q = np.asarray(cur, dtype=np.float64)
        for goal_world, step, open_here in segments:
            goal_pose = goal_world
            if self.world_to_base is not None:
                goal_pose = np.asarray(self.world_to_base) @ goal_world
            if any(goal_world is tp for tp in twist_segments):
                res: IKResult = solver(goal_pose, q, safe_joint_limits=self._limits,
                                       ori_tol=0.05)
            else:
                res: IKResult = solver(goal_pose, q, safe_joint_limits=self._limits)
            if not res.success and res.pos_error > 0.02:
                logger.warning(
                    f"release peel segment ik failed (pos_error={res.pos_error:.4f}); skipping"
                )
                continue
            if open_here and open_idx is None:
                open_idx = len(traj)
            traj.extend(interpolate_waypoints(q, res.q, max_joint_step=step))
            q = np.asarray(res.q, dtype=np.float64)
        if self._grasp_mode == "press" and self._release_dump_enabled:
            if open_idx is None:
                open_idx = len(traj)
            # raw un-interpolated jump, end pose above the target; gated
            from .motion import franka_fk

            # the anchor already carries whip_offset (see _cartesian_goal)
            dump = anchor.copy()
            dump[2, 3] = anchor[2, 3] + 0.12
            dump[:3, :3] = franka_fk(self._home)[:3, :3]
            dump_pose = dump
            if self.world_to_base is not None:
                dump_pose = np.asarray(self.world_to_base) @ dump
            res = solver(dump_pose, q, safe_joint_limits=self._limits)
            if res.success or res.pos_error <= 0.05:
                traj.extend(interpolate_waypoints(q, res.q, max_joint_step=10.0))
                q = np.asarray(res.q, dtype=np.float64)
            elif self._retreat_joints is not None:
                # fallback: raw jump to retreat_joints
                traj.extend(interpolate_waypoints(
                    q, self._retreat_joints, max_joint_step=10.0))
                q = np.asarray(self._retreat_joints, dtype=np.float64)
        if self._grasp_mode == "press":
            # tuck for an unobstructed view while the sheet is re-located
            traj.extend(interpolate_waypoints(q, self._tuck_joints, max_joint_step=0.1))
            q = self._tuck_joints.copy()
        elif self._retreat_joints is not None:
            if open_idx is None:
                open_idx = len(traj)
            traj.extend(interpolate_waypoints(q, self._retreat_joints, max_joint_step=0.05))
        self._release_open_after_idx = open_idx if open_idx is not None else 0
        return traj if traj else [q]

    def _log_carry_geometry(self) -> None:
        """Report where the sheet hangs relative to the fingertips once the
        carry is proven: place_z = board_z + (fingertip_z - sheet_bottom)."""
        if self.pad_gt_z_top is None or self.ee_world is None:
            return
        from .motion import _TCP_OFFSET_Z

        ee = np.asarray(self.ee_world, dtype=np.float64)
        tip = float(ee[2, 3] + (ee[:3, :3] @ np.array([0.0, 0.0, _TCP_OFFSET_Z]))[2])
        top = float(self.pad_gt_z_top)
        bottom = (float(self.pad_gt_z_bottom)
                  if self.pad_gt_z_bottom is not None else float("nan"))
        logger.info(
            f"carry geometry: fingertip_z={tip:.4f} sheet_top={top:.4f} "
            f"sheet_bottom={bottom:.4f} tip_over_top={tip - top:+.4f} "
            f"tip_over_bottom={tip - bottom:+.4f}"
        )

    def _log_landing(self) -> None:
        """One line per release: landing vs the slot and vs the
        whip-compensated anchor. ``err_vs_target`` is what the whip offset
        must cancel; ``drag_vs_anchor`` is the release's systematic pull."""
        if self._place_goal_world is None or self.pad_world_gt is None:
            return
        ax = float(self._place_goal_world[0, 3])
        ay = float(self._place_goal_world[1, 3])
        # the anchor carries whip_offset: slot = anchor minus that offset
        tx, ty = ax - self._whip_offset[0], ay - self._whip_offset[1]
        # pad_points is one deformable body: union centroid = sheet centroid
        # (what the scored bbox is centred on)
        px, py = self.pad_world_gt
        logger.info(
            f"landing telemetry: pad=({px:.4f},{py:.4f}) "
            f"slot=({tx:.4f},{ty:.4f}) anchor=({ax:.4f},{ay:.4f}) "
            f"err_vs_slot=({px - tx:+.4f},{py - ty:+.4f}) "
            f"drag_vs_anchor=({px - ax:+.4f},{py - ay:+.4f}) "
            f"z_top={float(self.pad_gt_z_top):.4f}"
            if self.pad_gt_z_top is not None else
            f"landing telemetry: pad=({px:.4f},{py:.4f}) "
            f"slot=({tx:.4f},{ty:.4f}) "
            f"err_vs_slot=({px - tx:+.4f},{py - ty:+.4f})"
        )

    def _build_reshake_trajectory(self, cur: np.ndarray) -> list[np.ndarray] | None:
        """Extra whip for a sheet that survived the whole release: return
        above the anchor, one raw high/rise jump cycle jaws-open plus the
        dump jump, end tucked. None when the anchor is unknown or IK fails."""
        anchor = self._place_goal_world
        if anchor is None:
            return None
        from .motion import franka_fk

        anchor = np.asarray(anchor, dtype=np.float64)
        rise = anchor.copy()
        rise[2, 3] = anchor[2, 3] + self._release_lift
        high = rise.copy()
        high[2, 3] = max(anchor[2, 3] + 0.25, rise[2, 3] + 0.15)
        dump = anchor.copy()
        dump[2, 3] = anchor[2, 3] + 0.12
        dump[:3, :3] = franka_fk(self._home)[:3, :3]
        solver = resolve_solver()
        traj: list[np.ndarray] = []
        q = np.asarray(cur, dtype=np.float64)
        # come in from above: a lerp from tuck to a low anchor sweeps the tool
        # through the board
        for goal_world, step in ((high, 0.05), (rise, 0.05), (high, 10.0),
                                 (rise, 10.0), (dump, 10.0)):
            goal_pose = goal_world
            if self.world_to_base is not None:
                goal_pose = np.asarray(self.world_to_base) @ goal_world
            res: IKResult = solver(goal_pose, q, safe_joint_limits=self._limits)
            if not res.success and res.pos_error > 0.02:
                logger.warning(
                    f"re-shake segment ik failed (pos_error={res.pos_error:.4f})")
                return None
            traj.extend(interpolate_waypoints(q, res.q, max_joint_step=step))
            q = np.asarray(res.q, dtype=np.float64)
        traj.extend(interpolate_waypoints(q, self._tuck_joints, max_joint_step=0.1))
        return traj if traj else None

    def _grasp_z(self) -> float | None:
        """Commanded fingertip height for the grasp, or None without GT:
        depth below the crest, escalating per regrasp up to the cap, clamped
        ``grasp_support_margin_m`` above the resting surface."""
        if self.pad_gt_z_top is None:
            return None
        z = (float(self.pad_gt_z_top) - self._grasp_depth
             - self._grasp_depth_step
             * min(self._regrasps_done, self._grasp_depth_max_steps))
        if self.pad_gt_z_bottom is not None and self._grasp_support_margin > 0.0:
            z = max(z, float(self.pad_gt_z_bottom) + self._grasp_support_margin)
        return max(0.005, z)

    def _scale_at(self, plane_z: float) -> float:
        """Pixels per metre for a feature lying at height ``plane_z``.

        ``pixel_scale`` is fx / (camera_z - pixel_scale_plane_z); rescaling to
        another height is just re-dividing by the new standoff."""
        if self._cam_height is None:
            return self._pixel_scale
        standoff = self._cam_height - plane_z
        if standoff <= 1e-6:
            return self._pixel_scale
        focal = self._pixel_scale * (self._cam_height - self._scale_plane_z)
        return focal / standoff

    def _px_to_world_xy(self, px: float, py: float) -> tuple[float, float]:
        """Camera-model pixel -> world xy (same mapping as _cartesian_goal)."""
        x = self._origin_xy[0] + (px - self._cam_cx) / self._pixel_scale
        dy = (py - self._cam_cy) / self._pixel_scale
        y = self._origin_xy[1] + (-dy if self._flip_y else dy)
        return x, y

    # -- two-arm pin-and-peel ---------------------------------------------
    def _pin_pose(self, x: float, y: float, z: float,
                  tilt: float, spin: float) -> np.ndarray:
        """Left fingertip pose at (x, y, z): tool axis tilted ``tilt`` from
        vertical AWAY from the sheet centre, wrist spun ``spin`` (free DOF)."""
        az = math.atan2(self._pin_offset[1], self._pin_offset[0]) + math.pi
        ca, sa = math.cos(az), math.sin(az)
        ct, st = math.cos(tilt), math.sin(tilt)
        cs, ss = math.cos(spin), math.sin(spin)
        rz_az = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[ct, 0.0, -st], [0.0, 1.0, 0.0], [st, 0.0, ct]])
        down = np.array([[cs, ss, 0.0], [ss, -cs, 0.0], [0.0, 0.0, -1.0]])
        p = np.eye(4)
        p[:3, :3] = rz_az @ ry @ rz_az.T @ down
        p[:3, 3] = (x, y, z)
        return p

    def _plan_pin(self) -> None:
        """Plan the left-arm pin descent (at RELEASE start); candidate
        (tilt, spin) pairs tried until the staged descent solves. Any
        infeasibility skips the pin — the whip behaves single-arm."""
        if not self._pin_enabled or self._left_state != "idle":
            return
        if self.left_world_to_base is None or self._place_goal_world is None:
            if self._pin_enabled:
                logger.warning("pin skipped: no left calibration/place anchor")
            self._left_state = "done"
            return
        # pin on the sheet: config offset picks the side, reach from the footprint
        ox, oy = self._pin_offset
        norm = math.hypot(ox, oy)
        if norm < 1e-6:
            ox, oy, norm = 1.0, 0.0, 1.0
        ux, uy = ox / norm, oy / norm
        reach = norm
        if self.pad_gt_half_extent is not None:
            hx, hy = self.pad_gt_half_extent
            # half-extent along the chosen direction, minus a 10 mm inset
            along = math.hypot(hx * ux, hy * uy)
            reach = min(max(along - 0.010, 0.015), 0.045)
        # the sheet rides the right fingertips: live centroid is the origin
        base_x = float(self._place_goal_world[0, 3])
        base_y = float(self._place_goal_world[1, 3])
        if self.pad_world_gt is not None:
            base_x, base_y = self.pad_world_gt
        px = base_x + ux * reach
        py = base_y + uy * reach
        az = math.atan2(uy, ux) + math.pi
        solver = resolve_solver()
        zs = [self._pin_clear_z]
        while zs[-1] - 0.05 > self._pin_press_z:
            zs.append(zs[-1] - 0.05)
        zs.append(self._pin_press_z)
        best_err = math.inf
        for tilt in (self._pin_tilt, 0.45, 0.3):
            for spin in (az, az + math.pi / 2, az + math.pi, az - math.pi / 2):
                traj: list[np.ndarray] = []
                q = self._tuck_joints.copy()
                ok = True
                clear_q: np.ndarray | None = None
                for i, zz in enumerate(zs):
                    goal = (np.asarray(self.left_world_to_base)
                            @ self._pin_pose(px, py, zz, tilt, spin))
                    res: IKResult = solver(goal, q, safe_joint_limits=self._limits)
                    if not (res.success or res.pos_error <= 0.02):
                        if i == 0 or zz == self._pin_press_z:
                            best_err = min(best_err, res.pos_error)
                            ok = False
                            break
                        continue  # intermediate stage: seeding only
                    traj.extend(interpolate_waypoints(q, res.q, max_joint_step=0.02))
                    q = np.asarray(res.q, dtype=np.float64)
                    if i == 0:
                        clear_q = q.copy()
                if ok and traj:
                    self._left_clear_q = clear_q
                    self._left_traj = traj
                    self._left_idx = 0
                    self._left_state = "descend"
                    self._left_state_since = None
                    logger.info(
                        f"pin planned: left fingertip -> ({px:.3f},{py:.3f},"
                        f"{self._pin_press_z:.3f}), reach={reach:.3f} from "
                        f"sheet ({base_x:.3f},{base_y:.3f}), tilt={tilt:.2f} "
                        f"spin={spin:.2f}, {len(traj)} waypoints")
                    return
        logger.warning(
            f"pin skipped: no left ik candidate solved (best pos_error "
            f"{best_err:.4f})")
        self._left_state = "done"

    def _pin_release(self) -> None:
        """Start the left-arm lift-off: one RAW un-interpolated jump back to
        the clear-height pose (a slow lift drags the friction-100 sheet up),
        then an interpolated tuck; falls back to the reversed descent."""
        if self._left_state not in ("descend", "hold"):
            return
        if self._left_clear_q is not None:
            back: list[np.ndarray] = [self._left_clear_q]
            back.extend(interpolate_waypoints(
                self._left_clear_q, self._tuck_joints, max_joint_step=0.05))
        else:
            back = list(reversed(self._left_traj)) if self._left_traj else []
            tail = back[-1] if back else self._tuck_joints
            back.extend(interpolate_waypoints(tail, self._tuck_joints,
                                              max_joint_step=0.05))
        self._left_traj = back
        self._left_idx = 0
        self._left_state = "lift"
        self._left_state_since = None

    def left_command(self, left_q: Sequence[float] | None,
                     now_s: float | None = None,
                     ) -> tuple[np.ndarray, float] | None:
        """Per-tick left-arm command (joints, gripper_open_fraction) or None.
        Loose reached gate; a stuck descent times out into "hold", a stuck
        lift into "done"."""
        if self._left_state in ("idle", "done"):
            return None
        if self._left_state == "hold":
            if self._left_last_cmd is None:
                return None
            return self._left_last_cmd, 0.0
        if self._left_state_since is None:
            self._left_state_since = now_s
        if (
            now_s is not None and self._left_state_since is not None
            and now_s - self._left_state_since > 90.0
        ):
            logger.warning(f"left {self._left_state} timed out; forcing on")
            self._left_state = "hold" if self._left_state == "descend" else "done"
            return self.left_command(left_q, now_s)
        if self._left_idx >= len(self._left_traj):
            self._left_state = "hold" if self._left_state == "descend" else "done"
            if self._left_state == "hold":
                logger.info("pin engaged (left fingertip down)")
            return self.left_command(left_q, now_s)
        cmd = self._left_traj[self._left_idx]
        reached = (
            left_q is None
            or float(np.max(np.abs(np.asarray(left_q[:7]) - cmd))) < 0.15
        )
        if reached:
            self._left_idx += 1
        self._left_last_cmd = cmd
        return cmd, 0.0

    def _solve_push_goals(self, cur: np.ndarray, start: tuple[float, float],
                          end: tuple[float, float], push_z: float,
                          clear_z: float, yaw: float,
                          ) -> tuple[list[np.ndarray], np.ndarray] | None:
        """Solve the four-goal push chain (hover, descend, drag, rise) at one
        wrist yaw; descent seeded by stepping z down from the hover solution.
        ALL contact goals must solve — a partial plan is a contact-less
        hover."""
        c, s = math.cos(yaw), math.sin(yaw)

        def pose_at(x: float, y: float, z: float) -> np.ndarray:
            p = np.eye(4)
            p[:3, :3] = [[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]]
            p[0, 3], p[1, 3] = x, y
            p[2, 3] = z
            return p

        solver = resolve_solver()

        def solve_to(x: float, y: float, z: float,
                     seed: np.ndarray) -> IKResult | None:
            goal = pose_at(x, y, z)
            if self.world_to_base is not None:
                goal = np.asarray(self.world_to_base) @ goal
            res: IKResult = solver(goal, seed, safe_joint_limits=self._limits)
            return res if (res.success or res.pos_error <= 0.02) else None

        traj: list[np.ndarray] = []
        q = np.asarray(cur, dtype=np.float64)
        res = solve_to(start[0], start[1], clear_z, q)  # hover above start
        if res is None:
            return None
        traj.extend(interpolate_waypoints(q, res.q, max_joint_step=0.05))
        q = np.asarray(res.q, dtype=np.float64)
        z = clear_z  # staged descent: intermediate stages enrich the seed
        while z - 0.06 > push_z:
            z -= 0.06
            mid = solve_to(start[0], start[1], z, q)
            if mid is not None:  # best-effort: seeding only
                traj.extend(interpolate_waypoints(q, mid.q, max_joint_step=0.03))
                q = np.asarray(mid.q, dtype=np.float64)
        # drag in <=10 cm segments
        stroke = math.hypot(end[0] - start[0], end[1] - start[1])
        n_seg = max(1, int(math.ceil(stroke / 0.10)))
        goals: list[tuple[tuple[float, float, float], float]] = [
            ((start[0], start[1], push_z), 0.03)]
        for k in range(1, n_seg + 1):
            f = k / n_seg
            goals.append(((start[0] + (end[0] - start[0]) * f,
                           start[1] + (end[1] - start[1]) * f, push_z), 0.03))
        goals.append(((end[0], end[1], clear_z), 0.05))
        for (gx, gy, gz), step in goals:
            res = solve_to(gx, gy, gz, q)
            if res is None:
                return None
            traj.extend(interpolate_waypoints(q, res.q, max_joint_step=step))
            q = np.asarray(res.q, dtype=np.float64)
        return traj, q

    def _build_push_trajectory(self, cur: np.ndarray,
                               sheet: TargetPose | None,
                               target: TargetPose | None) -> list[np.ndarray] | None:
        """Plan the floor push: approach behind the sheet, descend, slide it
        edge-on to the target, rise, tuck. Candidate azimuths and both
        cage-preserving yaws (yaw, yaw+pi) are tried. Returns None when
        sheet/target cannot be located or no candidate solves — a partial
        plan never executes."""
        if target is None or not target.visible:
            return None
        if self.pad_world_gt is not None:
            # DEV-ONLY ground truth (see class attribute docstring)
            sx, sy = self.pad_world_gt
        elif sheet is not None and sheet.visible:
            sx, sy = self._px_to_world_xy(sheet.x, sheet.y)
        else:
            return None
        tx, ty = self._px_to_world_xy(target.x, target.y)
        d = np.array([tx - sx, ty - sy], dtype=np.float64)
        dist = float(np.linalg.norm(d))
        if dist < 0.03:
            return None  # already on target — just verify
        d /= dist

        # sweep height from the sheet's GT top; without GT cycle heights after a whiff
        if self.pad_gt_z_top is not None:
            # 0.045 boundary: a pedestal-draped sheet crests ~5 cm (needs the
            # 0.055 sweep); the 0.020 sweep is hard-vetoed around the pedestal
            if self.pad_gt_z_top < 0.045:
                # sweep at the sheet's mid-height; reachability floor 0.032 (j2 limit)
                push_z = max(0.032, min(self._push_height,
                                        max(0.006, float(self.pad_gt_z_top) - 0.008)))
            elif self.pad_gt_z_top > 0.10:
                # pedestal stack: sweep at the top slab's mid-height
                push_z = (
                    float(self.pad_gt_z_slab_mid)
                    if self.pad_gt_z_slab_mid is not None
                    else 0.095
                )
            else:
                push_z = 0.055
        else:
            push_z = self._push_height
            if (
                self._last_push_sheet is not None and self._last_push_z is not None
                and math.hypot(sx - self._last_push_sheet[0],
                               sy - self._last_push_sheet[1]) < 0.02
            ):
                if self._last_push_z < 0.04:
                    push_z = 0.055
                elif self._last_push_z < 0.08:
                    push_z = 0.095
                logger.info(f"push retry at alternate height z={push_z:.3f}")

        clearing_pedestal = False
        if self._pad_locked is not None:
            # sheet still on/next to its pedestal: shove AWAY from the pedestal
            # centre first (a wall-adjacent push deflects sideways)
            lx, ly = self._px_to_world_xy(self._pad_locked.x, self._pad_locked.y)
            if abs(sx - lx) < 0.10 and abs(sy - ly) < 0.10:
                clearing_pedestal = True
                away = np.array([sx - lx, sy - ly], dtype=np.float64)
                n_away = float(np.linalg.norm(away))
                if n_away < 0.01:
                    # sheet centred on the pedestal: shove toward the target side
                    away = d.copy()
                else:
                    away /= n_away
                d = away
        # last round failed to move the sheet: skip the straight-on candidate
        sheet_stuck = (
            self._last_push_sheet is not None
            and math.hypot(sx - self._last_push_sheet[0],
                           sy - self._last_push_sheet[1]) < 0.02
        )
        self._last_push_sheet = (sx, sy)
        self._last_push_z = push_z

        # pedestal rect for the finger-corridor check (top 8x3 cm)
        ped_rect = None
        if self._pad_locked is not None and push_z < 0.095:
            lx, ly = self._px_to_world_xy(self._pad_locked.x, self._pad_locked.y)
            # 8x3 cm top centred ~5 mm north of the lock, padded ~7/5 mm
            ped_rect = (lx - 0.047, lx + 0.047,
                        ly + 0.005 - 0.020, ly + 0.005 + 0.020)

        # pusher width: blade sweeps +-1.2 cm, open cage +-4.25 cm
        blade = self._push_blade()
        track_span = 0.012 if blade else 0.0425

        def corridor_depth(start: tuple[float, float],
                           end: tuple[float, float],
                           d_c: np.ndarray) -> float:
            """Max penetration (m) of either FINGER track into the pedestal
            rect below pedestal-top height; shallow grazes allowed (a strict
            veto strands a wall-adjacent sheet). The PALM (~9 cm above
            fingertips) clears the 8.3 cm pedestal."""
            if ped_rect is None:
                return 0.0
            x0, x1r, y0, y1r = ped_rect
            nx, ny = -d_c[1], d_c[0]
            depth = 0.0
            for side in (track_span, -track_span):
                p0 = (start[0] + nx * side, start[1] + ny * side)
                p1 = (end[0] + nx * side, end[1] + ny * side)
                for k in range(21):
                    f = k / 20.0
                    px_ = p0[0] + (p1[0] - p0[0]) * f
                    py_ = p0[1] + (p1[1] - p0[1]) * f
                    if x0 < px_ < x1r and y0 < py_ < y1r:
                        depth = max(depth, min(px_ - x0, x1r - px_,
                                               py_ - y0, y1r - py_))
            return depth

        rotations = (0.0, 0.44, -0.44, 0.87, -0.87, 1.31, -1.31)
        if clearing_pedestal:
            # sideways-escape candidates: at low z the cage corridor clips the
            # pedestal; a ~100 deg shove slides the sheet clear
            rotations = rotations + (1.75, -1.75)
        if sheet_stuck:
            rotations = rotations[1:]
        candidates: list[tuple[float, np.ndarray, tuple[float, float],
                               tuple[float, float]]] = []
        # reject-cause tallies: an unplannable round says WHY
        rejects = {"corridor": 0, "deadzone": 0, "ik": 0}
        for rot in rotations:
            cr, sr = math.cos(rot), math.sin(rot)
            d_c = np.array([d[0] * cr - d[1] * sr, d[0] * sr + d[1] * cr])
            start = (sx - d_c[0] * self._push_standoff,
                     sy - d_c[1] * self._push_standoff)
            if clearing_pedestal:
                # short clearing shove
                end = (sx + d_c[0] * 0.15, sy + d_c[1] * 0.15)
            else:
                end = (tx - d_c[0] * self._push_stop_short,
                       ty - d_c[1] * self._push_stop_short)
            depth = corridor_depth(start, end, d_c)
            # graze cap 12 mm
            if depth <= 0.012:
                candidates.append((depth, d_c, start, end))
            else:
                rejects["corridor"] += 1
        # clear corridors first, then shallow grazes by increasing depth
        candidates.sort(key=lambda c: (c[0] > 1e-9, c[0]))
        for depth, d_c, start, end in candidates:
            clear_z = self._clear_h
            if self.arm_base_world is not None:
                ab = np.asarray(self.arm_base_world, dtype=np.float64)
                # dead-zone prefilter: down-wrist solvable only in the
                # 0.28-0.70 m annulus around the base column
                skip = False
                for px_, py_ in (start, end):
                    r = math.hypot(px_ - ab[0], py_ - ab[1])
                    if r < 0.28 or r > 0.70:
                        skip = True
                if skip:
                    rejects["deadzone"] += 1
                    continue
                dxr = start[0] - ab[0]
                dyr = start[1] - ab[1]
                dz = math.sqrt(max(0.76 * 0.76 - dxr * dxr - dyr * dyr, 0.01))
                clear_z = min(clear_z, ab[2] + dz)
            yaw0 = math.atan2(d_c[1], d_c[0])
            for yaw in (yaw0, yaw0 + math.pi):
                plan = self._solve_push_goals(cur, start, end, push_z,
                                              clear_z, yaw)
                if plan is None:
                    rejects["ik"] += 1
                    continue
                traj, q = plan
                traj.extend(interpolate_waypoints(
                    q, self._tuck_joints, max_joint_step=0.1))
                logger.info(
                    f"push planned: sheet ({sx:.3f},{sy:.3f}) -> target "
                    f"({tx:.3f},{ty:.3f}), {dist * 100:.1f} cm, z={push_z:.3f}, "
                    f"az={math.degrees(math.atan2(d_c[1], d_c[0])):+.0f}deg "
                    f"yaw={yaw:.2f} graze={depth * 100:.1f}cm"
                    + (" [clearing]" if clearing_pedestal else "")
                )
                return traj
        logger.warning(
            "push unplannable: no candidate azimuth/yaw solved "
            f"(sheet ({sx:.3f},{sy:.3f}) z={push_z:.3f} clear={clear_z:.3f}, "
            f"rejects: corridor={rejects['corridor']} "
            f"deadzone={rejects['deadzone']} ik={rejects['ik']})")
        return None

    def _cartesian_goal(self, phase: Phase, target: TargetPose | None,
                        pad: TargetPose | None = None) -> np.ndarray | None:
        """4x4 SE(3) target pose for a phase. We only vary x/y/z and yaw.

        Grasp phases (PREGRASP/GRASP/LIFT) aim at the *pad* blob; place phases
        (PREPLACE/PLACE) aim at the *placement-target* blob. When the preferred
        source is missing or not visible we fall back to the other; when
        neither is visible we hover above the home FK pose at clearance height.
        """
        from .motion import franka_fk

        grasp_phases = (Phase.PREGRASP, Phase.GRASP, Phase.LIFT)
        preferred, fallback = (
            (pad, target) if phase in grasp_phases else (target, pad)
        )
        src = preferred if preferred is not None and preferred.visible else None
        if src is None and fallback is not None and fallback.visible:
            src = fallback
        if src is None:
            # default pose: above the table centre at clearance height
            base = franka_fk(self._home)
            base[2, 3] = self._clear_h
            return base

        pose = np.eye(4)
        # pixel -> workspace xy; flip_y: image +v -> world -Y
        src_x, src_y = src.x, src.y
        if (
            phase in (Phase.PREPLACE, Phase.PLACE) and src is target
            and self._pad_only_locked is not None and self._pad_locked is not None
        ):
            # shift by the pad's offset within the carried assembly (locked
            # pre-grasp, px space) so the THERMALPAD lands on target
            src_x -= self._pad_only_locked[0] - self._pad_locked.x
            src_y -= self._pad_only_locked[1] - self._pad_locked.y
        # scale at the height this phase looks at
        scale = self._scale_at(
            self._pad_plane_z if src is not target else self._target_plane_z
        )
        pose[0, 3] = self._origin_xy[0] + (src_x - self._cam_cx) / scale
        dy = (src_y - self._cam_cy) / scale
        pose[1, 3] = self._origin_xy[1] + (-dy if self._flip_y else dy)
        if phase in grasp_phases and self.pad_world_gt is not None:
            # DEV-ONLY GT override: aim at the top-cluster centroid
            if self.pad_world_gt_top is not None:
                pose[0, 3], pose[1, 3] = self.pad_world_gt_top
            else:
                pose[0, 3], pose[1, 3] = self.pad_world_gt
        if phase in (Phase.PREPLACE, Phase.PLACE):
            # aimed whip: shift the whole place/release chain; (0,0) = direct
            pose[0, 3] += self._whip_offset[0]
            pose[1, 3] += self._whip_offset[1]
        yaw = src.yaw
        if phase in grasp_phases:
            yaw += self._grasp_yaw_offset
        c, s = np.cos(yaw), np.sin(yaw)
        # yaw about Z then 180-deg flip about X: tool Z down
        pose[:3, :3] = [[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]]

        if phase == Phase.PREGRASP:
            pose[2, 3] = self._clear_h
        elif phase == Phase.GRASP:
            # press depth below the crest, escalating per regrasp
            # (see _grasp_z)
            gz = self._grasp_z()
            pose[2, 3] = self._grasp_h if gz is None else gz
        elif phase == Phase.LIFT:
            pose[2, 3] = self._clear_h
        elif phase == Phase.PREPLACE:
            pose[2, 3] = self._clear_h
        elif phase == Phase.PLACE:
            pose[2, 3] = self._place_h
        else:
            pose[2, 3] = self._clear_h

        # arm-base z is 0.42..0.69 across resets; clamp clearance z within
        # 0.76 m reach (grasp/place untouched)
        if self.arm_base_world is not None and pose[2, 3] == self._clear_h:
            ab = np.asarray(self.arm_base_world, dtype=np.float64)
            dx = pose[0, 3] - ab[0]
            dy = pose[1, 3] - ab[1]
            dz = math.sqrt(max(0.76 * 0.76 - dx * dx - dy * dy, 0.01))
            pose[2, 3] = min(pose[2, 3], ab[2] + dz)
        return pose

    def _transition(self, now_s: float, cur: np.ndarray,
                    obs: PlacementObservation | None,
                    target: TargetPose | None,
                    pad: TargetPose | None = None) -> Decision:
        nxt = {
            Phase.PREGRASP: Phase.GRASP,
            Phase.GRASP: Phase.LIFT,
            Phase.LIFT: Phase.PREPLACE,
            Phase.PREPLACE: Phase.PLACE,
            Phase.PLACE: Phase.RELEASE,
        }.get(self._phase)
        if nxt is None:
            if self._phase == Phase.RELEASE:
                if (
                    # mode-agnostic: the liner rides an OPEN pinch like a press bond
                    self._grasp_mode in ("press", "pinch")
                    and self.pad_gt_z_top is not None
                    and float(self.pad_gt_z_top) > 0.15
                    and self._reshakes_done < 3
                ):
                    # detach verification: a real drop leaves the GT top < 0.15
                    self._reshakes_done += 1
                    logger.warning(
                        f"detach check failed (sheet top "
                        f"{float(self.pad_gt_z_top):.3f}); "
                        f"re-shake {self._reshakes_done}/3")
                    traj = self._build_reshake_trajectory(cur)
                    if traj is not None:
                        self._trajectory = traj
                        self._traj_idx = 0
                        # jaws stay open through the whole re-shake
                        self._release_open_after_idx = 0
                        self._phase_start = now_s
                        return Decision(Phase.RELEASE,
                                        tuple(float(x) for x in cur), 1.0,
                                        "re-shake whip")
                self._log_landing()
                if self._push_enabled and self._grasp_mode in ("press", "pinch"):
                    # recover with a floor push (planned lazily in step())
                    self._phase = Phase.PUSH
                    self._phase_start = now_s
                    self._trajectory = []
                    self._traj_idx = 0
                    self._push_planned = False
                    return Decision(Phase.PUSH, tuple(float(x) for x in cur), 0.0,
                                    "push settle (sheet landing)")
                self._phase = Phase.VERIFY
                self._verify_start = None
                self._phase_start = now_s
                return Decision(Phase.VERIFY, None, 1.0, "verifying")
            if self._phase == Phase.PUSH:
                if self._pushes_done < self._max_pushes:
                    # planning returns None once the sheet is within 3 cm of target
                    self._phase_start = now_s
                    self._trajectory = []
                    self._traj_idx = 0
                    self._push_planned = False
                    return Decision(Phase.PUSH, tuple(float(x) for x in cur), 0.0,
                                    "push settle (re-locating)")
                self._phase = Phase.VERIFY
                self._verify_start = None
                self._phase_start = now_s
                return Decision(Phase.VERIFY, None, 1.0, "verifying")
            return Decision(self._phase, None, 1.0, f"holding in {self._phase.value}")
        if (
            nxt in (Phase.PREPLACE, Phase.PLACE)
            # push_only is the only mode with no carry to verify
            and self._grasp_mode in ("press", "pinch")
            and self.pad_gt_z_top is not None
            and float(self.pad_gt_z_top) < 0.15
            and self._regrasps_done < 4
        ):
            # attach verification: a carried sheet keeps the GT top above 0.15
            self._regrasps_done += 1
            press_goal_z = self._grasp_z() or 0.0
            if self._push_enabled and press_goal_z < 0.032:
                # regrasp press would aim below the j2-feasible floor: push instead
                logger.warning(
                    f"attach check failed before {nxt.value} and the regrasp "
                    f"press goal ({press_goal_z:.3f}) is below the feasible "
                    f"floor; push recovery")
                self._phase = Phase.PUSH
                self._phase_start = now_s
                self._trajectory = []
                self._traj_idx = 0
                self._push_planned = False
                return Decision(Phase.PUSH, tuple(float(x) for x in cur), 0.0,
                                "floor sheet unpressable -> push recovery")
            logger.warning(
                f"attach check failed before {nxt.value} "
                f"(sheet top {float(self.pad_gt_z_top):.3f}); "
                f"regrasp {self._regrasps_done}/4")
            self._begin_phase(Phase.PREGRASP, now_s, cur, target, pad)
            return Decision(Phase.PREGRASP, tuple(float(x) for x in cur), 1.0,
                            "attach failed -> regrasp")
        if nxt == Phase.PREPLACE:
            self._log_carry_geometry()
        if nxt == Phase.LIFT and self._grasp_mode == "press" and self._trajectory:
            # the dwell must keep commanding the press-depth target (settled
            # joints relax the fingertip off the stack)
            self._press_hold_q = np.asarray(
                self._trajectory[-1], dtype=np.float64).copy()
        self._begin_phase(nxt, now_s, cur, target, pad)
        if self.is_terminal:
            # IK failed inside _begin_phase -> stay fail-closed, no arm command
            return Decision(self._phase, None, 1.0, f"terminal {self._phase.value}")
        if nxt == Phase.RELEASE and self._release_dwell_s > 0.0:
            # hold at the place pose; peel starts after the dwell in step()
            g = 0.0 if self._grasp_mode == "press" else 1.0
            return Decision(Phase.RELEASE, tuple(float(x) for x in cur), g,
                            "release dwell")
        if nxt == Phase.LIFT and self._grasp_close_dwell_s > 0.0:
            # hold at the grasp pose while the jaws close (see step())
            hold = self._press_hold_q if (
                self._grasp_mode == "press" and self._press_hold_q is not None
            ) else cur
            return Decision(Phase.LIFT, tuple(float(x) for x in hold), 0.0,
                            "grasp close dwell (jaws closing)")
        if self._trajectory:
            t = self._trajectory[0]
            return Decision(self._phase, tuple(float(x) for x in t),
                            self._gripper_for(self._phase), f"begin {self._phase.value}")
        return Decision(self._phase, None, self._gripper_for(self._phase),
                        f"begin {self._phase.value}")

    def _maybe_servo(self, cur: np.ndarray, now_s: float = 0.0) -> Decision | None:
        """One ee-pose Jacobian correction after a completed GRASP/PLACE
        trajectory; None when it is time to advance.
        ``dq = pinv(J_pos(q)) @ (T_bw[:3,:3] @ err_world)``, TCP = flange +
        0.15 along flange z, clipped +-0.15 rad per iteration."""
        if self._phase not in (Phase.GRASP, Phase.PLACE):
            return None
        if self._servo_left <= 0 or self._servo_goal_world is None:
            return None
        if self.ee_world is None or self.world_to_base is None:
            return None
        # settle gate: re-issue the previous correction until tracked, cap ~6 s
        if (
            self._servo_last_cmd is not None
            and float(np.max(np.abs(cur - self._servo_last_cmd))) > self._servo_settle_tol
        ):
            if self._servo_settle_since is None:
                self._servo_settle_since = now_s
            if now_s - self._servo_settle_since < self._servo_settle_timeout:
                return Decision(
                    self._phase, tuple(float(x) for x in self._servo_last_cmd),
                    self._gripper_for(self._phase), "servo settling",
                )
        self._servo_settle_since = None
        from .motion import _jacobian

        ee = np.asarray(self.ee_world, dtype=np.float64)
        tcp_meas = ee[:3, 3] + ee[:3, :3] @ np.array([0.0, 0.0, 0.15])
        err = np.asarray(self._servo_goal_world)[:3, 3] - tcp_meas
        if float(np.linalg.norm(err)) <= self._servo_tol:
            self._servo_left = 0
            return None
        self._servo_left -= 1
        dp_base = np.asarray(self.world_to_base)[:3, :3] @ err
        dq = np.linalg.pinv(_jacobian(cur)[:3, :]) @ dp_base
        dq = np.clip(dq, -0.15, 0.15)
        goal_q = cur + dq
        # clamp into safe limits: pinv(J) walks past a joint at its limit -> refuse loop
        if self._limits:
            goal_q = np.array([
                min(max(float(q), lo), hi)
                for q, (lo, hi) in zip(goal_q, self._limits)
            ])
        self._servo_last_cmd = goal_q.copy()
        return Decision(
            self._phase, tuple(float(x) for x in goal_q),
            self._gripper_for(self._phase),
            f"servo {self._servo_iters - self._servo_left}/{self._servo_iters} "
            f"err={float(np.linalg.norm(err)) * 1000:.1f}mm",
        )

    def _step_verify(self, now_s: float, obs: PlacementObservation | None) -> Decision:
        if self._verify_start is None:
            self._verify_start = now_s
        if obs is None:
            return Decision(Phase.VERIFY, None, 1.0, "awaiting verification mask")
        if now_s - self._verify_start > self._verify_timeout:
            return self._fail("verification timeout")
        if obs.iou >= self._min_iou and obs.liner_dominance_ratio >= self._min_liner:
            self._phase = Phase.SUCCEEDED
            return Decision(Phase.SUCCEEDED, None, 1.0, "placement verified")
        return Decision(Phase.VERIFY, None, 1.0, "verifying placement")

    # -- helpers ----------------------------------------------------------
    def _reached(self, cur: np.ndarray, target: np.ndarray) -> bool:
        return bool(np.max(np.abs(cur - target)) <= self._joint_tol)

    def _gripper_for(self, phase: Phase) -> float:
        # pinch: open through GRASP descent; press: closed down and through transport
        if phase in (Phase.LIFT, Phase.PREPLACE, Phase.PLACE):
            return self._grasp_close_fraction
        if phase == Phase.GRASP and self._grasp_mode == "press":
            return 0.0
        if phase == Phase.PUSH:
            # shape-adaptive pusher (see _push_blade)
            return 0.0 if self._push_blade() else 1.0
        return 1.0

    def _push_blade(self) -> bool:
        """Blade (closed jaws) vs cage (open) pusher: crest < 4.5 cm slides
        under the cage untouched -> blade; tall crumpled piles keep the
        straddle cage."""
        return self.pad_gt_z_top is not None and float(self.pad_gt_z_top) < 0.045

    def _fail(self, reason: str) -> Decision:
        self._phase = Phase.FAILED
        return Decision(Phase.FAILED, None, 1.0, reason)


# ---------------------------------------------------------------------------
# VLA policy (lazy-loaded foundation model)
# ---------------------------------------------------------------------------
@dataclass
class VLAConfig:
    """Deployment knobs for the foundation VLA backend."""

    backend: str = "pi05"          # pi05 | openvla | lerobot | stub
    checkpoint: str | None = None
    action_chunk_size: int = 8
    delta_actions: bool = True
    max_joint_delta: float = 0.15  # rad per step safety clamp
    active_arm: str = "right"
    # must match the recorder's `single_task` string (SmolVLA conditions on it)
    instruction: str = (
        "Pick up the thermal pad and place it on the target RAM board."
    )


class VLAPolicy:
    """Foundation-VLA action-chunk policy.

    The model is loaded on first :meth:`load` call (lazy). Offline, with no
    checkpoint, the policy uses a deterministic stub backend that holds the arm
    at its current configuration, so the contract and tests exercise cleanly.
    On a GPU host, call ``load()`` with a real checkpoint.
    """

    def __init__(self, cfg: VLAConfig | None = None) -> None:
        self.cfg = cfg or VLAConfig()
        self._backend = None
        self._loaded = False
        self._phase = Phase.WAIT_FOR_STATE
        self._chunk: list[np.ndarray] = []
        self._chunk_grippers: list[float] = []
        self._chunk_base: list[np.ndarray | None] = []
        self._chunk_spine: list[float] = []
        self._chunk_left: list[np.ndarray] = []
        self._chunk_left_grip: list[float] = []
        self._chunk_idx = 0
        self._last_joints: np.ndarray | None = None
        self._have_object = False
        self._verify_start: float | None = None
        self._min_iou = 0.85
        self._min_liner = 0.90
        self._verify_timeout = 6.0

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def is_terminal(self) -> bool:
        return self._phase in (Phase.SUCCEEDED, Phase.FAILED)

    @property
    def camera_keys(self) -> list[str] | None:
        """Recorder camera keys the loaded checkpoint expects, so the runner
        subscribes to exactly those streams (None before load, or for
        backends that take a single image)."""
        keys = getattr(self._backend, "_image_keys", None)
        if not keys:
            return None
        return [k.removeprefix("observation.images.") for k in keys]

    def load(self) -> bool:
        """Lazily build the VLA backend; True only if a *real* (non-stub)
        model loaded. Failures log and leave the deterministic stub."""
        from .backends import StubBackend

        # always start from a safe stub so propose() never returns None
        self._backend = StubBackend()
        self._loaded = False
        try:
            from .backends.vla_runtime import build_backend

            built = build_backend(self.cfg)
            if built is not None and built.backend_name != "stub":
                self._backend = built
                self._loaded = True
                logger.info(f"VLA backend loaded: {built.backend_name}")
            else:
                logger.info(
                    f"VLA backend '{self.cfg.backend}' unavailable; using deterministic stub"
                )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"VLA backend load failed ({e!r}); using deterministic stub")
        return self._loaded

    def start(self, now_s: float) -> None:
        self._phase = Phase.WAIT_FOR_STATE
        self._chunk = []
        self._chunk_grippers = []
        self._chunk_base = []
        self._chunk_spine = []
        self._chunk_left = []
        self._chunk_left_grip = []
        self._chunk_idx = 0
        self._verify_start = None
        self._last_cmd = None
        self._last_left_cmd = None

    def step(
        self,
        now_s: float,
        joints: Sequence[float] | None,
        observation: PlacementObservation | None,
        image: np.ndarray | None = None,
        instruction: str | None = None,
        state: np.ndarray | None = None,
        images: dict[str, np.ndarray] | None = None,
    ) -> Decision:
        if joints is None and self._phase == Phase.WAIT_FOR_STATE:
            return Decision(Phase.WAIT_FOR_STATE, None, None, "waiting for joint state")
        cur = np.asarray(joints, dtype=np.float64) if joints is not None else self._last_joints
        if cur is None:
            return Decision(Phase.WAIT_FOR_STATE, None, None, "waiting for joint state")
        self._last_joints = cur

        if self.is_terminal:
            return Decision(self._phase, None, 1.0 if self._phase == Phase.FAILED else 0.0,
                            f"terminal {self._phase.value}")

        if self._phase == Phase.WAIT_FOR_STATE:
            self._phase = Phase.GRASP

        if self._chunk_idx >= len(self._chunk):
            self._chunk = self._propose(cur, image, instruction, state, images)
            self._chunk_idx = 0
            # hold-fallback and real chunks are indistinguishable from outside
            logger.info(
                f"vla refill: chunk={len(self._chunk)} "
                f"state={'None' if state is None else f'{np.shape(state)}'} "
                f"images={sorted(images) if images else None} "
                f"backend={'yes' if self._backend is not None else 'NO'}"
            )
            if not self._chunk:
                # no model + empty -> hold (fail-safe identity). Gripper
                # None: holds must not publish the phase-default gripper.
                return Decision(self._phase, tuple(cur.tolist()),
                                None, "vla hold (no chunk)")

        nxt = self._chunk[self._chunk_idx]
        # Only the backend's gripper_seq may command the gripper. Holds send
        # None (publish nothing; drives keep the last target) — demos start
        # OPEN; a phase-default 0.0 here closes the gripper during warm-up.
        grip: float | None = None
        if self._chunk_grippers and self._chunk_idx < len(self._chunk_grippers):
            grip = float(self._chunk_grippers[self._chunk_idx])
        base: tuple[float, float, float] | None = None
        if self._chunk_base and self._chunk_idx < len(self._chunk_base):
            b = self._chunk_base[self._chunk_idx]
            if b is not None:
                base = (float(b[0]), float(b[1]), float(b[2]))
        spine: float | None = None
        if self._chunk_spine and self._chunk_idx < len(self._chunk_spine):
            spine = float(self._chunk_spine[self._chunk_idx])
        left_target: tuple[float, ...] | None = None
        left_grip: float | None = None
        if self._chunk_left and self._chunk_idx < len(self._chunk_left):
            left_nxt = self._chunk_left[self._chunk_idx]
            # Same previous-command rate limit as the active arm; anchor from
            # the observed left joints (state[14:21]) on first step/divergence.
            left_prev = getattr(self, "_last_left_cmd", None)
            left_cur = (np.asarray(state[14:21], dtype=np.float64)
                        if state is not None and np.shape(state)[-1] >= 21
                        else None)
            if left_prev is None or (
                    left_cur is not None
                    and np.max(np.abs(left_cur - left_prev)) > 0.5):
                left_prev = left_cur if left_cur is not None else left_nxt
            left_delta = np.clip(left_nxt - left_prev,
                                 -self.cfg.max_joint_delta,
                                 self.cfg.max_joint_delta)
            left_cmd = left_prev + left_delta
            self._last_left_cmd = left_cmd
            left_target = tuple(left_cmd.tolist())
            if self._chunk_left_grip and self._chunk_idx < len(self._chunk_left_grip):
                left_grip = float(self._chunk_left_grip[self._chunk_idx])
        self._chunk_idx += 1
        # Rate-limit against the PREVIOUS COMMAND: anchoring every step to
        # the measured joints would cap executed speed below the demo
        # cadence.
        prev = self._last_cmd
        if prev is None or np.max(np.abs(cur - prev)) > 0.5:
            # first step, or hard divergence: re-anchor on measured joints
            prev = cur
        delta = np.clip(nxt - prev, -self.cfg.max_joint_delta,
                        self.cfg.max_joint_delta)
        target = prev + delta
        self._last_cmd = target
        return Decision(self._phase, tuple(target.tolist()),
                        grip, f"vla step {self._chunk_idx}", base_twist=base,
                        spine=spine, left_arm_target=left_target,
                        left_gripper_open_fraction=left_grip)

    def _propose(self, cur: np.ndarray, image: np.ndarray | None,
                 instruction: str | None,
                 state: np.ndarray | None = None,
                 images: dict[str, np.ndarray] | None = None,
                 ) -> list[np.ndarray]:
        """Ask the VLA backend for an action chunk and return it as a list of
        7-D joint vectors. On any backend error, log and fall back to a hold."""
        from .backends import VLAInput

        self._chunk_grippers = []
        self._chunk_base = []
        self._chunk_spine = []
        self._chunk_left = []
        self._chunk_left_grip = []
        if self._backend is not None:
            try:
                out = self._backend.propose(
                    VLAInput(joints=cur, gripper_open=self._gripper_for(self._phase),
                             image=image, instruction=instruction,
                             state=state, images=images),
                    horizon=self.cfg.action_chunk_size,
                )
                chunk = [np.asarray(a, dtype=np.float64) for a in out.chunk][
                    : self.cfg.action_chunk_size
                ]
                if chunk:
                    if out.gripper_seq:
                        self._chunk_grippers = [
                            float(g) for g in out.gripper_seq[: len(chunk)]
                        ]
                    if getattr(out, "base_seq", None):
                        self._chunk_base = [
                            None if b is None else np.asarray(b, dtype=np.float64)
                            for b in out.base_seq[: len(chunk)]
                        ]
                    if getattr(out, "spine_seq", None):
                        self._chunk_spine = [
                            float(s) for s in out.spine_seq[: len(chunk)]
                        ]
                    if getattr(out, "left_seq", None):
                        self._chunk_left = [
                            np.asarray(a, dtype=np.float64)
                            for a in out.left_seq[: len(chunk)]
                        ]
                    if getattr(out, "left_gripper_seq", None):
                        self._chunk_left_grip = [
                            float(g) for g in out.left_gripper_seq[: len(chunk)]
                        ]
                    return chunk
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"VLA backend.propose failed ({e!r}); holding")
        # deterministic fallback: hold the current configuration
        return [cur.copy() for _ in range(self.cfg.action_chunk_size)]

    def _gripper_for(self, phase: Phase) -> float:
        if phase in (Phase.GRASP, Phase.LIFT, Phase.PREPLACE, Phase.PLACE):
            return 0.0
        return 1.0

    def report_verification(self, obs: PlacementObservation | None, now_s: float) -> Decision:
        """Called by the runner once the gripper is open at the place site."""
        if self._verify_start is None:
            self._verify_start = now_s
        if obs is None:
            return Decision(Phase.VERIFY, None, 1.0, "awaiting verification mask")
        if now_s - self._verify_start > self._verify_timeout:
            self._phase = Phase.FAILED
            return Decision(Phase.FAILED, None, 1.0, "verification timeout")
        if obs.iou >= self._min_iou and obs.liner_dominance_ratio >= self._min_liner:
            self._phase = Phase.SUCCEEDED
            return Decision(Phase.SUCCEEDED, None, 1.0, "placement verified")
        return Decision(Phase.VERIFY, None, 1.0, "verifying placement")
