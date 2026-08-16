"""The fail-closed fixed-waypoint autonomy state machine for Task 2.

This is the scaffold's deterministic policy: drive the arm through five
hand-calibrated joint waypoints, gating the gripper by phase and verifying
placement success from the semantic camera. It fails open (gripper open, no
further commands) on timeout or verification failure.

A :class:`PerceptionPolicy` (:mod:`ebim_task2.vla_policy`) and a
:class:`VLAPolicy` implement the same ``step``/``Decision`` contract and are
selectable from the runner. The waypoint FSM remains the trusted fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .perception import PlacementObservation


class Phase(str, Enum):
    WAIT_FOR_STATE = "wait_for_state"
    PREGRASP = "pregrasp"
    GRASP = "grasp"
    LIFT = "lift"
    PREPLACE = "preplace"
    PLACE = "place"
    RELEASE = "release"
    # Post-release floor push toward the target (PerceptionPolicy press mode
    # only): the release drop lands the sheet flat but short of the target;
    # the push slides it the rest of the way.
    PUSH = "push"
    VERIFY = "verify"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class Decision:
    """A single policy output for one control tick."""

    phase: Phase
    arm_target: tuple[float, ...] | None
    gripper_open_fraction: float | None
    reason: str
    # Mobile-base twist (vx, vy, omega) for policies that command the base
    # (VLA checkpoints trained on the recorder's whole-body action row).
    # None = no base command this tick; the bridge's 1 sim-second pedal
    # watchdog then zeroes the base on its own.
    base_twist: tuple[float, float, float] | None = None
    # Vertical-spine height (metres, absolute). None = leave the runner's
    # held value alone; the runner repeats the last commanded height in every
    # arm message because the bridge drops the group after a 1 s gap.
    spine: float | None = None
    # LEFT-arm joint targets + left gripper open fraction (whole-body VLA
    # checkpoints: the demos choreograph the left arm through every episode
    # and its joints are part of the trained state). None = no left command.
    left_arm_target: tuple[float, ...] | None = None
    left_gripper_open_fraction: float | None = None


class Task2WaypointPolicy:
    """Fixed-waypoint FSM. Construct with calibrated waypoints + tolerances."""

    WAYPOINT_SEQUENCE = (
        Phase.PREGRASP,
        Phase.GRASP,
        Phase.LIFT,
        Phase.PREPLACE,
        Phase.PLACE,
    )

    def __init__(
        self,
        waypoints: dict[str, Sequence[float]],
        *,
        joint_tolerance_rad: float,
        waypoint_timeout_s: float,
        verification_timeout_s: float,
        min_iou: float,
        liner_dominance_ratio: float,
        release_settle_s: float = 1.0,
    ) -> None:
        self._waypoints = {k: tuple(v) for k, v in waypoints.items()}
        self._joint_tol = joint_tolerance_rad
        self._wp_timeout = waypoint_timeout_s
        self._verify_timeout = verification_timeout_s
        self._release_settle_s = release_settle_s
        self._min_iou = min_iou
        self._min_liner = liner_dominance_ratio

        self._phase = Phase.WAIT_FOR_STATE
        self._seq_idx = 0
        self._phase_start: float | None = None
        self._verify_start: float | None = None

    # -- lifecycle --------------------------------------------------------
    def start(self, now_s: float) -> None:
        self._phase = Phase.WAIT_FOR_STATE
        self._seq_idx = 0
        self._phase_start = now_s
        self._verify_start = None

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def is_terminal(self) -> bool:
        return self._phase in (Phase.SUCCEEDED, Phase.FAILED)

    # -- core step --------------------------------------------------------
    def step(
        self,
        now_s: float,
        joints: Sequence[float] | None,
        observation: PlacementObservation | None,
    ) -> Decision:
        # Fail-closed if we never get joint state.
        if joints is None and self._phase == Phase.WAIT_FOR_STATE:
            return Decision(Phase.WAIT_FOR_STATE, None, None, "waiting for joint state")

        if self._phase == Phase.WAIT_FOR_STATE:
            self._phase = self.WAYPOINT_SEQUENCE[0]
            self._seq_idx = 0
            self._phase_start = now_s

        if self.is_terminal:
            # Stay open after terminal.
            return Decision(
                self._phase, None, 1.0 if self._phase == Phase.FAILED else 0.0,
                f"terminal {self._phase.value}",
            )

        if self._phase == Phase.VERIFY:
            return self._step_verify(now_s, observation)

        if self._phase == Phase.RELEASE:
            # Hold the gripper-open command for a short settle delay, then
            # transition into VERIFY.
            if self._phase_start is not None and (now_s - self._phase_start) >= self._release_settle_s:
                self.enter_verify(now_s)
                return self._step_verify(now_s, observation)
            return Decision(Phase.RELEASE, None, 1.0, "releasing gripper")

        # We are driving to the current waypoint.
        waypoint_name = self._phase.value
        target = self._waypoints.get(waypoint_name)
        if not target:
            return self._fail("missing calibrated waypoint")

        # Timeout -> fail open.
        if self._phase_start is not None and (now_s - self._phase_start) > self._wp_timeout:
            return self._fail(f"{waypoint_name} waypoint timeout")

        reached = self._reached(joints, target)
        gripper = self._gripper_for(self._phase)
        if reached:
            return self._advance(now_s)
        return Decision(self._phase, tuple(target), gripper, f"driving to {waypoint_name}")

    # -- helpers ----------------------------------------------------------
    def _reached(self, joints: Sequence[float] | None, target: Sequence[float]) -> bool:
        if joints is None:
            return False
        j = list(joints)
        if len(j) != len(target):
            return False
        return all(abs(a - b) <= self._joint_tol for a, b in zip(j, target))

    def _gripper_for(self, phase: Phase) -> float:
        # Closed during transport, open otherwise.
        if phase in (Phase.LIFT, Phase.PREPLACE, Phase.PLACE):
            return 0.0
        return 1.0

    def _advance(self, now_s: float) -> Decision:
        next_idx = self._seq_idx + 1
        if next_idx < len(self.WAYPOINT_SEQUENCE):
            self._seq_idx = next_idx
            self._phase = self.WAYPOINT_SEQUENCE[next_idx]
            self._phase_start = now_s
            return Decision(
                self._phase,
                tuple(self._waypoints[self._phase.value]),
                self._gripper_for(self._phase),
                f"advanced to {self._phase.value}",
            )
        # Finished the place waypoint -> release then verify.
        self._phase = Phase.RELEASE
        self._phase_start = now_s
        return Decision(Phase.RELEASE, None, 1.0, "releasing")

    def _step_verify(self, now_s: float, obs: PlacementObservation | None) -> Decision:
        # Once in VERIFY we wait briefly for the mask to settle, then gate.
        if self._verify_start is None:
            self._verify_start = now_s
        if obs is None:
            # keep gripper open while waiting for the camera
            return Decision(Phase.VERIFY, None, 1.0, "awaiting verification mask")
        if now_s - self._verify_start > self._verify_timeout:
            return self._fail("verification timeout")
        if obs.iou >= self._min_iou and obs.liner_dominance_ratio >= self._min_liner:
            self._phase = Phase.SUCCEEDED
            return Decision(Phase.SUCCEEDED, None, 1.0, "placement verified")
        return Decision(Phase.VERIFY, None, 1.0, "verifying placement")

    def _fail(self, reason: str) -> Decision:
        self._phase = Phase.FAILED
        return Decision(Phase.FAILED, None, 1.0, reason)

    # The runner transitions RELEASE->VERIFY once the gripper reports open.
    def enter_verify(self, now_s: float) -> None:
        self._phase = Phase.VERIFY
        self._verify_start = None
        self._phase_start = now_s
