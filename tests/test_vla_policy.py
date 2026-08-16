"""Tests for the perception- and VLA-driven policies (Decision contract)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebim_task2.policy import Phase  # noqa: E402
from ebim_task2.backends import VLAOutput  # noqa: E402
from ebim_task2.perception import PlacementObservation  # noqa: E402
from ebim_task2.vla_policy import PerceptionPolicy, VLAPolicy  # noqa: E402


# ---------------- PerceptionPolicy ----------------
def test_perception_policy_waits_then_moves():
    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0)
    pol.start(0.0)
    d0 = pol.step(0.1, None, None)
    assert d0.phase == Phase.WAIT_FOR_STATE
    # Provide joints; should begin PREGRASP and emit a target.
    home = [0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8]
    d1 = pol.step(0.2, home, None)
    assert d1.phase == Phase.PREGRASP
    assert d1.arm_target is not None
    assert len(d1.arm_target) == 7


def test_perception_policy_gripper_open_above_table():
    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0)
    pol.start(0.0)
    home = [0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8]
    d = pol.step(0.2, home, None)
    assert d.phase == Phase.PREGRASP
    assert d.gripper_open_fraction == 1.0


def test_perception_policy_can_verify_success():
    pol = PerceptionPolicy(
        joint_tolerance_rad=0.5, waypoint_timeout_s=30.0,
        verification_timeout_s=5.0, min_iou=0.8, liner_dominance_ratio=0.8,
    )
    # Force the policy straight into VERIFY by exercising report path via internals.
    pol._phase = Phase.VERIFY
    pol._verify_start = None
    good = PlacementObservation(iou=0.9, liner_dominance_ratio=0.9, pad_pixels=10, target_pixels=10)
    pol._min_iou = 0.8
    pol._min_liner = 0.8
    d = pol._step_verify(1.0, good)
    assert d.phase == Phase.SUCCEEDED


def test_release_builds_peel_trajectory_and_dwells():
    """RELEASE must dwell (jaws opening, arm held), then run the peel
    trajectory: press below the place height, lateral wipe, slow rise,
    clearance — never an immediate retreat that drags the pad."""
    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                           release_dwell_s=1.0, release_press_m=0.005,
                           release_wipe_m=0.02, release_lift_m=0.06)
    home = [0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8]
    pol.start(0.0)
    # Simulate a completed PLACE: anchor pose recorded, phase at PLACE.
    from ebim_task2.motion import franka_fk
    anchor = franka_fk(home)
    anchor[2, 3] = 0.005
    pol._place_goal_world = anchor.copy()
    pol._phase = Phase.PLACE
    pol._trajectory = []
    pol._traj_idx = 0
    pol._phase_start = 0.0
    # PLACE trajectory exhausted -> transition into RELEASE (dwell decision).
    d = pol.step(0.1, home, None)
    assert d.phase == Phase.RELEASE
    assert d.gripper_open_fraction == 1.0
    assert d.arm_target == tuple(float(x) for x in home)  # held, not retreating
    # Still inside the dwell window -> keep holding.
    d2 = pol.step(0.5, home, None)
    assert d2.phase == Phase.RELEASE and "dwell" in d2.reason
    # A peel trajectory exists and starts near the current configuration
    # (slow steps), not with a jump to clearance.
    assert len(pol._trajectory) > 2
    first_step = max(abs(a - b) for a, b in zip(pol._trajectory[0], home))
    assert first_step <= 0.015 + 1e-9


def test_grasp_descends_open_then_closes_at_depth():
    """The jaws must stay OPEN through the GRASP descent and close while the
    arm holds at the grasp pose (grasp-close dwell at the start of LIFT)."""
    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                           grasp_close_dwell_s=1.0)
    home = [0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8]
    pol.start(0.0)
    pol._phase = Phase.GRASP
    pol._phase_start = 0.0
    import numpy as _np
    pol._trajectory = [_np.asarray(home) + 0.3]
    pol._traj_idx = 0
    # Mid-descent: still open.
    d = pol.step(0.1, home, None)
    assert d.phase == Phase.GRASP
    assert d.gripper_open_fraction == 1.0
    # Trajectory complete -> LIFT begins with a close dwell: arm held, closed.
    d2 = pol.step(0.2, [x + 0.3 for x in home], None)
    assert d2.phase == Phase.LIFT
    assert d2.gripper_open_fraction == 0.0
    assert "dwell" in d2.reason
    # Still dwelling before the window elapses.
    d3 = pol.step(0.7, [x + 0.3 for x in home], None)
    assert d3.phase == Phase.LIFT and "dwell" in d3.reason


def test_release_timeout_goes_to_verify_not_failed():
    """A stalled peel (fingertips pressing the board) must verify, not FAIL."""
    pol = PerceptionPolicy(joint_tolerance_rad=1e-6, waypoint_timeout_s=30.0,
                           release_dwell_s=0.0, release_timeout_s=2.0)
    home = [0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8]
    pol.start(0.0)
    pol._phase = Phase.RELEASE
    pol._phase_start = 0.0
    import numpy as _np
    pol._trajectory = [_np.asarray(home) + 0.5]  # unreachable with tiny tol
    pol._traj_idx = 0
    d = pol.step(5.0, home, None)
    assert d.phase == Phase.VERIFY
    assert not pol.is_terminal or pol._phase == Phase.VERIFY


def test_pose_locks_survive_occlusion():
    """The target pose locked while the view is clean must override a later
    (occluded, shifted) live estimate in the place phases."""
    from ebim_task2.perception import TargetPose

    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0)
    home = [0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8]
    pol.start(0.0)
    clean = TargetPose(x=500.0, y=350.0, yaw=0.0, visible=True)
    pad = TargetPose(x=200.0, y=200.0, yaw=0.0, visible=True)
    # First step (WAIT -> PREGRASP): both poses visible -> locked.
    pol.step(0.1, home, None, target_pose=clean)
    assert pol._target_locked == clean
    # Occluded estimate in PREPLACE (lock frozen there) must NOT displace
    # the goal.
    pol._phase = Phase.PREPLACE
    pol._trajectory = []
    pol._traj_idx = 0
    shifted = TargetPose(x=560.0, y=390.0, yaw=0.3, visible=True)
    goal = pol._cartesian_goal(Phase.PLACE, clean, pad)
    # Step into PLACE with the shifted live estimate; the lock wins.
    pol.step(0.2, home, None, target_pose=shifted)
    goal_after = pol._servo_goal_world
    assert goal_after is not None
    assert abs(goal_after[0, 3] - goal[0, 3]) < 1e-9
    assert abs(goal_after[1, 3] - goal[1, 3]) < 1e-9


def test_pad_lock_is_first_sighting_only():
    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0)
    home = [0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8]
    pol.start(0.0)
    mask = np.zeros((480, 640), dtype=np.int32)
    mask[190:210, 190:210] = 3
    ids = {"thermalpad": 3, "target": 4, "liner": 5}
    # estimate_pad_pose(mask) would give ~(199.5, 199.5); we inject via mask.
    pol.step(0.1, home, None, mask=mask, semantic_ids=ids)
    locked = pol._pad_locked
    assert locked is not None and locked.visible
    # A later, different-looking blob must not update the lock.
    mask2 = np.zeros((480, 640), dtype=np.int32)
    mask2[100:120, 100:120] = 3
    pol.step(0.2, home, None, mask=mask2, semantic_ids=ids)
    assert pol._pad_locked == locked


# ---------------- VLAPolicy ----------------
def test_vla_policy_falls_back_to_hold_without_model():
    pol = VLAPolicy()
    # No load() -> stub backend that holds current config.
    assert pol.load() is False
    pol.start(0.0)
    d0 = pol.step(0.1, None, None)
    assert d0.phase == Phase.WAIT_FOR_STATE
    home = [0.0] * 7
    d1 = pol.step(0.2, home, None)
    assert d1.arm_target is not None
    assert len(d1.arm_target) == 7
    # hold: target == current (no delta from stub)
    assert tuple(d1.arm_target) == tuple(home)


def test_vla_policy_clamps_per_step_delta():
    pol = VLAPolicy()
    pol.start(0.0)
    cur = [0.0] * 7

    class _LurchBackend:
        """Backend with the real propose(VLAInput, *, horizon) signature that
        proposes a huge absolute jump, so the per-step clamp is exercised."""

        def propose(self, inp, *, horizon):
            return VLAOutput(
                chunk=[np.asarray(inp.joints) + 5.0 for _ in range(horizon)],
                is_delta=False,
            )

    pol._backend = _LurchBackend()
    pol._loaded = True
    d = pol.step(0.2, cur, None)
    delta = max(abs(a - b) for a, b in zip(d.arm_target, cur))
    assert delta <= pol.cfg.max_joint_delta + 1e-6
    # > 0 proves the backend path ran (not the TypeError -> hold fallback).
    assert delta > 0.0


def test_vla_policy_hold_does_not_command_gripper():
    """Holds (stub/fallback chunks with no gripper_seq) must send gripper
    None: the demos start OPEN, and the old phase-default (GRASP -> 0.0)
    closed the gripper during every warm-up hold."""
    pol = VLAPolicy()
    pol.start(0.0)
    pol._phase = Phase.GRASP  # simulate active grasp phase
    d = pol.step(0.2, [0.0] * 7, None)
    assert d.gripper_open_fraction is None


def test_vla_policy_threads_base_twist_through_decisions():
    """A backend base_seq must surface per-step on Decision.base_twist (and
    stay None for backends that do not command the base). Official fixpos
    demos hold base ≡ 0; the channel serves mobile-trained checkpoints."""
    pol = VLAPolicy()
    pol.start(0.0)

    class _BaseBackend:
        def propose(self, inp, *, horizon):
            n = min(horizon, 2)
            return VLAOutput(
                chunk=[np.asarray(inp.joints) for _ in range(n)],
                is_delta=False,
                base_seq=[np.array([0.0, 0.5, 0.0]), None],
            )

    pol._backend = _BaseBackend()
    pol._loaded = True
    d1 = pol.step(0.2, [0.0] * 7, None)
    assert d1.base_twist == (0.0, 0.5, 0.0)
    d2 = pol.step(0.3, [0.0] * 7, None)
    assert d2.base_twist is None
    # Stub / no-base backends leave the field at its default.
    pol2 = VLAPolicy()
    pol2.start(0.0)
    assert pol2.step(0.2, [0.0] * 7, None).base_twist is None


def test_vla_policy_threads_left_arm_through_decisions():
    """A backend left_seq must surface on Decision.left_arm_target (rate-
    limited against the previous left command) with the left gripper; stub
    backends leave both None."""
    pol = VLAPolicy()
    pol.start(0.0)

    class _LeftBackend:
        def propose(self, inp, *, horizon):
            n = min(horizon, 2)
            return VLAOutput(
                chunk=[np.asarray(inp.joints) for _ in range(n)],
                is_delta=False,
                left_seq=[np.full(7, 0.05), np.full(7, 5.0)],
                left_gripper_seq=[1.0, 1.0],
            )

    pol._backend = _LeftBackend()
    pol._loaded = True
    state = np.zeros(37)
    d1 = pol.step(0.2, [0.0] * 7, None, state=state)
    assert d1.left_arm_target is not None
    assert np.allclose(d1.left_arm_target, [0.05] * 7)
    assert d1.left_gripper_open_fraction == 1.0
    # Second step lurches to 5.0 rad: the per-step clamp must cap it
    # relative to the previous left command.
    d2 = pol.step(0.3, [0.0] * 7, None, state=state)
    assert max(d2.left_arm_target) <= 0.05 + pol.cfg.max_joint_delta + 1e-9
    # Stub / joints-only backends leave the left fields at their defaults.
    pol2 = VLAPolicy()
    pol2.start(0.0)
    d = pol2.step(0.2, [0.0] * 7, None)
    assert d.left_arm_target is None and d.left_gripper_open_fraction is None


def test_push_trajectory_planned_from_live_sheet():
    """The floor push must approach behind the dropped sheet, descend, drag
    toward the target, and rise — planned from live estimates."""
    from ebim_task2.perception import TargetPose

    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                           grasp_mode="press", push_enabled=True,
                           cam_cx=640.0, cam_cy=360.0, pixel_scale=660.6,
                           origin_xy=(0.837, -0.065), flip_y=True)
    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    sheet = TargetPose(x=615.0, y=505.0, yaw=0.0, visible=True)   # ~(0.80,-0.28)
    target = TargetPose(x=616.0, y=319.0, yaw=0.0, visible=True)  # ~(0.80, 0.00)
    traj = pol._build_push_trajectory(home, sheet, target)
    assert traj is not None and len(traj) > 5
    # Unplannable when the sheet is invisible.
    hidden = TargetPose(0.0, 0.0, 0.0, visible=False)
    assert pol._build_push_trajectory(home, hidden, target) is None
    # Already on target -> None (skip straight to verify).
    near = TargetPose(x=617.0, y=320.0, yaw=0.0, visible=True)
    assert pol._build_push_trajectory(home, near, target) is None


def test_release_ends_with_raw_jump_and_tuck_in_press_mode():
    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                           grasp_mode="press", release_shake_cycles=0,
                           retreat_joints=[0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    from ebim_task2.motion import franka_fk
    anchor = franka_fk(home.tolist())
    anchor[2, 3] = 0.025
    pol._place_goal_world = anchor.copy()
    traj = pol._build_release_trajectory(home, None, None)
    assert traj is not None and len(traj) >= 2
    # The trajectory must end at the tuck pose (view-clearing).
    assert np.allclose(traj[-1], pol._tuck_joints, atol=1e-9)


def _push_policy(**kw):
    from ebim_task2.perception import TargetPose

    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                           grasp_mode="press", push_enabled=True,
                           cam_cx=640.0, cam_cy=360.0, pixel_scale=660.6,
                           origin_xy=(0.837, -0.065), flip_y=True, **kw)
    target = TargetPose(x=616.0, y=319.0, yaw=0.0, visible=True)  # ~(0.80, 0.00)
    return pol, target


def test_push_never_executes_contactless_plan():
    """If no candidate yaw/azimuth can solve the CONTACT goals, the planner
    must return None — a partial plan hovers above the sheet without
    touching it."""
    from ebim_task2.motion import IKResult, load_solver

    def floor_blind_solver(goal_pose, seed, **kw):
        if goal_pose[2, 3] < 0.1:  # any low goal fails
            return IKResult(False, np.asarray(seed), 0.5, 1)
        return IKResult(True, np.asarray(seed), 0.0, 1)

    pol, target = _push_policy()
    pol.pad_world_gt = (0.80, -0.28)
    load_solver(floor_blind_solver)
    try:
        traj = pol._build_push_trajectory(
            np.asarray([0.0, -1.5, 0.0, -2.2, 0.0, 1.5, 0.785]), None, target)
    finally:
        load_solver(None)
    assert traj is None


def test_push_yaw_flip_rescues_plan():
    """A solver that rejects the azimuth-slaved yaw but accepts the flipped
    (cage-preserving) yaw must still yield a plan."""
    from ebim_task2.motion import IKResult, load_solver

    def yaw_picky_solver(goal_pose, seed, **kw):
        # world x-axis of the goal frame: accept only x-component <= 0
        if goal_pose[0, 0] > 0.0:
            return IKResult(False, np.asarray(seed), 0.5, 1)
        return IKResult(True, np.asarray(seed), 0.0, 1)

    pol, target = _push_policy()
    pol.pad_world_gt = (0.80, -0.28)  # push azimuth ~ +y -> yaw ~ +90deg
    load_solver(yaw_picky_solver)
    try:
        traj = pol._build_push_trajectory(
            np.asarray([0.0, -1.5, 0.0, -2.2, 0.0, 1.5, 0.785]), None, target)
    finally:
        load_solver(None)
    assert traj is not None


def test_push_height_from_gt_sheet_top():
    """With GT z-top available the sweep height must match the sheet's actual
    lie: floor-flat -> reachability floor 0.032 (near-floor tool-down sweeps
    are j2-infeasible under the USD limits), on the pedestal -> 0.095,
    draped -> 0.055."""
    for z_top, want in ((0.035, 0.032), (0.12, 0.095), (0.07, 0.055)):
        pol, target = _push_policy(push_height_m=0.02)
        pol.pad_world_gt = (0.70, -0.28)
        pol.pad_gt_z_top = z_top
        pol._build_push_trajectory(
            np.asarray([0.0, -1.5, 0.0, -2.2, 0.0, 1.5, 0.785]), None, target)
        assert pol._last_push_z == want, (z_top, pol._last_push_z)


def test_pin_skipped_without_left_calibration():
    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                           grasp_mode="press", pin_enabled=True)
    pol._place_goal_world = np.eye(4)
    pol._plan_pin()
    assert pol._left_state == "done"
    assert pol.left_command([0.0] * 7, 1.0) is None


def test_pin_descend_hold_lift_cycle():
    """The left arm must descend along the planned pin trajectory, hold at
    the pin, and lift back through the reversed path to done."""
    from ebim_task2.motion import IKResult, load_solver

    def permissive_solver(goal_pose, seed, **kw):
        q = np.asarray(seed, dtype=np.float64) + 0.05
        return IKResult(True, q, 0.0, 1)

    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                           grasp_mode="press", pin_enabled=True,
                           pin_clear_z=0.06, pin_press_z=0.01)
    pol.left_world_to_base = np.eye(4)
    anchor = np.eye(4)
    anchor[0, 3], anchor[1, 3], anchor[2, 3] = 0.5, 0.0, 0.02
    pol._place_goal_world = anchor
    load_solver(permissive_solver)
    try:
        pol._plan_pin()
    finally:
        load_solver(None)
    assert pol._left_state == "descend" and len(pol._left_traj) > 0
    # Walk the descent: echo each command back as the measured left joints.
    q = [0.0] * 7
    for _ in range(len(pol._left_traj) + 2):
        out = pol.left_command(q, 1.0)
        if pol._left_state == "hold":
            break
        assert out is not None
        q = [float(v) for v in out[0]]
        assert out[1] == 0.0  # jaws stay closed
    assert pol._left_state == "hold"
    # Hold keeps repeating the last command.
    held = pol.left_command(q, 2.0)
    assert held is not None and np.allclose(held[0], q)
    # Lift: reversed path then tuck, ending done with nothing to publish.
    pol._pin_release()
    assert pol._left_state == "lift"
    for _ in range(len(pol._left_traj) + 2):
        out = pol.left_command(q, 3.0)
        if out is None:
            break
        q = [float(v) for v in out[0]]
    assert pol._left_state == "done"
    assert pol.left_command(q, 4.0) is None


def _press_policy(**kw):
    """Press-mode policy with the barebone camera model, started and holding
    at the home pose (the offline solver context for release/push tests)."""
    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                           grasp_mode="press", push_enabled=True,
                           cam_cx=640.0, cam_cy=360.0, pixel_scale=660.6,
                           origin_xy=(0.837, -0.065), flip_y=True, **kw)
    pol.start(0.0)
    return pol


def test_regrasp_escalates_press_depth():
    """Each regrasp attempt must press 2 mm deeper: a weld that already
    failed gets a deeper seat."""
    from ebim_task2.perception import TargetPose

    pol = _press_policy()
    pad = TargetPose(x=615.0, y=505.0, yaw=0.0, visible=True)
    pol.pad_gt_z_top = 0.102
    depths = []
    for n in range(4):
        pol._regrasps_done = n
        goal = pol._cartesian_goal(Phase.GRASP, None, pad)
        depths.append(goal[2, 3])
    assert depths[0] == pytest.approx(0.102 - 0.008, abs=1e-9)
    assert depths[1] == pytest.approx(0.102 - 0.010, abs=1e-9)
    assert depths[2] == pytest.approx(0.102 - 0.012, abs=1e-9)
    # Capped at regrasp 2: deeper seats weld too hard for the twist to
    # shear off.
    assert depths[3] == pytest.approx(0.102 - 0.012, abs=1e-9)


def _exhausted_release(pol, home):
    from ebim_task2.motion import franka_fk
    anchor = franka_fk(list(home))
    anchor[2, 3] = 0.025
    pol._place_goal_world = anchor.copy()
    pol._phase = Phase.RELEASE
    pol._trajectory = []
    pol._traj_idx = 0
    # Long past the release-dwell window so step() falls straight through to
    # the trajectory-exhausted transition.
    pol._phase_start = -100.0


def test_reshake_when_sheet_still_carried():
    """Sheet GT top still high after the release whip -> whip again, jaws
    open, instead of planning a push against the finger position."""
    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    pol = _press_policy()
    _exhausted_release(pol, home)
    pol.pad_gt_z_top = 0.40  # still riding the fingers
    d = pol.step(0.1, home.tolist(), None)
    assert d.phase == Phase.RELEASE
    assert "re-shake" in d.reason
    assert pol._reshakes_done == 1
    assert len(pol._trajectory) > 0
    assert pol._release_open_after_idx == 0  # jaws open from the first step


def test_no_reshake_after_budget_exhausted():
    """After 3 re-shakes the episode moves on (push against whatever the GT
    says) rather than whipping forever."""
    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    pol = _press_policy()
    _exhausted_release(pol, home)
    pol.pad_gt_z_top = 0.40
    pol._reshakes_done = 3
    d = pol.step(0.1, home.tolist(), None)
    assert d.phase == Phase.PUSH
    assert pol._reshakes_done == 3


def test_no_reshake_when_drop_confirmed():
    """A real drop leaves the GT top at floor height -> straight to PUSH."""
    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    pol = _press_policy()
    _exhausted_release(pol, home)
    pol.pad_gt_z_top = 0.002  # floor-flat sheet
    d = pol.step(0.1, home.tolist(), None)
    assert d.phase == Phase.PUSH
    assert pol._reshakes_done == 0


def _release_policy(dump: bool):
    return PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                            grasp_mode="press", release_shake_cycles=0,
                            release_dump_enabled=dump,
                            retreat_joints=[0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])


def _release_traj(pol):
    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    from ebim_task2.motion import franka_fk
    anchor = franka_fk(home.tolist())
    anchor[2, 3] = 0.025
    pol._place_goal_world = anchor.copy()
    return pol._build_release_trajectory(home, None, None), home


def test_release_dump_gate_removes_raw_jump():
    """dump disabled: no wrist-flip raw PD jump in the release — the
    trajectory still ends tucked."""
    traj, _ = _release_traj(_release_policy(dump=False))
    assert traj is not None and len(traj) >= 2
    # No raw PD jump: every step is an interpolated one (tuck uses 0.1).
    for a, b in zip(traj, traj[1:]):
        assert max(abs(x - y) for x, y in zip(a, b)) < 0.12
    assert np.allclose(traj[-1], PerceptionPolicy(
        joint_tolerance_rad=0.2)._tuck_joints, atol=1e-9)


def test_release_dump_enabled_keeps_raw_jump():
    """dump enabled (default): the release keeps exactly one raw PD jump —
    the proven detach for stubborn welds (used by the re-shake path)."""
    traj, _ = _release_traj(_release_policy(dump=True))
    assert traj is not None
    jumps = [1 for a, b in zip(traj, traj[1:])
             if max(abs(x - y) for x, y in zip(a, b)) >= 0.12]
    assert len(jumps) >= 1


def test_push_only_starts_in_push_not_pregrasp():
    """push_only must skip the whole grasp/carry/whip pipeline: once joints
    arrive the policy enters PUSH (settle), keeping the pad lock for the
    pedestal-clearing logic."""
    home = [0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8]
    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                           grasp_mode="push_only", push_enabled=True)
    pol.start(0.0)
    d = pol.step(0.1, home, None)
    assert d.phase == Phase.PUSH
    assert "settle" in d.reason
    assert pol._pushes_done == 0


def test_push_only_requires_push_enabled():
    """push_only without push_enabled fails closed, silently never moving."""
    home = [0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8]
    pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                           grasp_mode="push_only", push_enabled=False)
    pol.start(0.0)
    d = pol.step(0.1, home, None)
    assert d.phase == Phase.FAILED
    assert pol.is_terminal


def test_push_height_pedestal_stack_uses_slab_mid():
    """Pedestal-stack sheets must sweep at the top slab's mid-height, not a
    fixed height that passes under the slab."""
    pol, target = _push_policy()
    pol.pad_world_gt = (0.80, -0.30)
    pol.pad_gt_z_top = 0.102
    pol.pad_gt_z_slab_mid = 0.0995
    pol._build_push_trajectory(
        np.asarray([0.0, -1.5, 0.0, -2.2, 0.0, 1.5, 0.785]), None, target)
    assert pol._last_push_z == pytest.approx(0.0995)
    # Without the slab-mid feed the legacy 0.095 fallback applies.
    pol2, target = _push_policy()
    pol2.pad_world_gt = (0.80, -0.30)
    pol2.pad_gt_z_top = 0.102
    pol2._build_push_trajectory(
        np.asarray([0.0, -1.5, 0.0, -2.2, 0.0, 1.5, 0.785]), None, target)
    assert pol2._last_push_z == pytest.approx(0.095)


def test_push_sideways_escape_when_wedged_off_pedestal_edge():
    """Sheet wedged just north of the pedestal rect: every target-ward
    corridor clips the rect at low z. The clearing-mode +-100 deg
    candidates must slide it sideways out of the pedestal x-band instead
    of declaring the round unplannable."""
    from ebim_task2.motion import IKResult, load_solver
    from ebim_task2.perception import TargetPose

    def permissive_solver(goal_pose, seed, **kw):
        return IKResult(True, np.asarray(seed, dtype=np.float64) + 0.05, 0.0, 1)

    pol, target = _push_policy()
    # Pedestal lock at world (0.80,-0.30) -> pixel (615.6, 515.2).
    pol._pad_locked = TargetPose(x=615.6, y=515.2, yaw=0.0, visible=True)
    pol.pad_world_gt = (0.777, -0.243)   # sheet 3 cm north of the rect edge
    pol.pad_gt_z_top = 0.037             # floor pile -> low sweep, rect active
    pol._last_push_sheet = (0.777, -0.243)  # stuck -> straight-on skipped
    load_solver(permissive_solver)
    try:
        traj = pol._build_push_trajectory(
            np.asarray([0.0, -1.5, 0.0, -2.2, 0.0, 1.5, 0.785]), None, target)
    finally:
        load_solver(None)
    assert traj is not None



def _grasp_traj(pol):
    """Build the press-mode GRASP trajectory from the home pose against the
    barebone camera model (pad top 0.102 -> press depth 0.096)."""
    from ebim_task2.perception import TargetPose

    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    pad = TargetPose(x=500.0, y=450.0, yaw=0.0, visible=True)
    pol.pad_gt_z_top = 0.102
    traj = pol._build_trajectory(Phase.GRASP, home, None, pad)
    return traj, 0.102 - 0.008


def _z_profile(traj):
    from ebim_task2.motion import franka_fk

    return [float(franka_fk(list(q))[2, 3]) for q in traj]


def test_press_grasp_default_is_single_fast_descent():
    """With the reliability flags off the GRASP trajectory stays the legacy
    single fast descent (no re-lift after first touching the press depth)."""
    pol = _press_policy()
    traj, depth = _grasp_traj(pol)
    assert traj is not None and len(traj) >= 1
    zs = _z_profile(traj)
    assert zs[-1] == pytest.approx(depth, abs=0.01)
    first_touch = next((i for i, z in enumerate(zs) if z <= depth + 0.004), None)
    assert first_touch is not None
    assert all(z < depth + 0.015 for z in zs[first_touch + 1:])


def test_press_double_tap_seats_twice_with_slow_final():
    """Double-tap press: reach the depth, re-lift ~2 cm, seat again — the
    second press recovers welds the first impact failed to form. All
    segments after the first touch run at the slow transport step (no
    impact shovel)."""
    slow = 0.012
    pol = _press_policy(transport_step_rad=slow, press_double_tap=True,
                        press_double_tap_lift_m=0.02, press_slow_final_m=0.02)
    traj, depth = _grasp_traj(pol)
    assert traj is not None and len(traj) >= 3
    zs = _z_profile(traj)
    assert zs[-1] == pytest.approx(depth, abs=0.01)
    first_touch = next((i for i, z in enumerate(zs) if z <= depth + 0.004), None)
    assert first_touch is not None
    # A re-lift of >=15 mm after the first seat...
    lifts = [i for i, z in enumerate(zs) if i > first_touch and z >= depth + 0.015]
    assert lifts, "expected a re-lift after the first press seat"
    # ...and the trajectory comes back down to the depth afterwards.
    assert min(zs[max(lifts):]) <= depth + 0.004
    # Every step after the first touch is a slow (transport-step) one.
    for a, b in zip(traj[first_touch:], traj[first_touch + 1:]):
        assert max(abs(x - y) for x, y in zip(a, b)) <= slow + 1e-9


def test_regrasp_aims_top_cluster_centroid():
    """Every press aims at the TOP-CLUSTER centroid when it exists: the
    sheet is 2 cm wide and the union centroid sits ~1 cm off the top sheet
    after any disturbance — an edge press flips the sheet off the pedestal
    instead of welding."""
    from ebim_task2.perception import TargetPose

    pol = _press_policy()
    pad = TargetPose(x=615.0, y=505.0, yaw=0.0, visible=True)
    pol.pad_world_gt = (0.80, -0.30)
    pol.pad_world_gt_top = (0.82, -0.28)
    pol._regrasps_done = 0
    goal = pol._cartesian_goal(Phase.GRASP, None, pad)
    assert (goal[0, 3], goal[1, 3]) == pytest.approx((0.82, -0.28))
    pol._regrasps_done = 1
    goal = pol._cartesian_goal(Phase.GRASP, None, pad)
    assert (goal[0, 3], goal[1, 3]) == pytest.approx((0.82, -0.28))
    # Without the top-cluster feed the union centroid is the fallback.
    pol.pad_world_gt_top = None
    goal = pol._cartesian_goal(Phase.GRASP, None, pad)
    assert (goal[0, 3], goal[1, 3]) == pytest.approx((0.80, -0.30))


def test_press_mode_grasp_skips_ee_servo():
    """Press-mode GRASP must not run the ee servo: servoing to a Cartesian
    depth goal against the compliant stack turns contact residual into
    uncontrolled force that bulldozes the sheet off the pedestal.
    PLACE keeps the servo, and so does a pinch-mode GRASP."""
    from ebim_task2.perception import TargetPose

    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    pad = TargetPose(x=500.0, y=450.0, yaw=0.0, visible=True)
    pol = _press_policy(servo_iters=12)
    pol.pad_gt_z_top = 0.102
    pol._begin_phase(Phase.GRASP, 0.0, home, None, pad)
    assert pol._servo_left == 0
    pol._begin_phase(Phase.PLACE, 0.0, home, None, pad)
    assert pol._servo_left == 12
    pinch = PerceptionPolicy(joint_tolerance_rad=0.2, servo_iters=12,
                             grasp_mode="pinch")
    pinch.start(0.0)
    pinch._begin_phase(Phase.GRASP, 0.0, home, None, pad)
    assert pinch._servo_left == 12


def test_press_dwell_holds_press_depth_command_not_settled():
    """The grasp-close dwell must keep COMMANDING the press-depth waypoint:
    holding the settled joints relaxes the fingertip off the stack for the
    whole weld-forming window."""
    from ebim_task2.perception import TargetPose

    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    pad = TargetPose(x=500.0, y=450.0, yaw=0.0, visible=True)
    pol = _press_policy()
    pol.pad_gt_z_top = 0.102
    traj = pol._build_trajectory(Phase.GRASP, home, None, pad)
    assert traj
    pol._phase = Phase.GRASP
    pol._trajectory = list(traj)
    pol._traj_idx = len(traj)
    settled = home + 0.01  # the arm stopped short of the depth waypoint
    d = pol._transition(1.0, settled, None, None, pad)
    assert d.phase == Phase.LIFT
    assert np.allclose(d.arm_target, np.asarray(traj[-1]), atol=1e-9)
    # step() keeps publishing the press-depth command through the dwell
    d2 = pol.step(1.0 + pol._grasp_close_dwell_s / 2, settled.tolist(), None)
    assert d2.phase == Phase.LIFT
    assert np.allclose(d2.arm_target, np.asarray(traj[-1]), atol=1e-9)
    # after the dwell the LIFT trajectory runs instead
    d3 = pol.step(1.0 + pol._grasp_close_dwell_s + 0.5, settled.tolist(), None)
    assert d3.arm_target is not None
    assert not np.allclose(d3.arm_target, np.asarray(traj[-1]), atol=1e-9)


def test_press_release_twist_shears_between_press_and_rise():
    """Press-mode release with release_twist>0 must insert the symmetric
    twist pair between the press and the rise: the shear breaks the finger
    bond AT the slot for sub-cm placement."""
    from ebim_task2.motion import franka_fk

    anchor = franka_fk([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    anchor[2, 3] = 0.025

    def build(twist_rad):
        pol = PerceptionPolicy(joint_tolerance_rad=0.2, waypoint_timeout_s=30.0,
                               grasp_mode="press", release_shake_cycles=0,
                               release_dump_enabled=False,
                               release_twist_rad=twist_rad,
                               retreat_joints=[0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
        pol._place_goal_world = anchor.copy()
        home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
        return pol._build_release_trajectory(home, None, None)

    def max_yaw_dev_at_press_depth(traj):
        # Only waypoints near the press depth count (the end-of-release
        # tuck changes orientation by >1 rad in BOTH variants).
        worst = 0.0
        for q in traj:
            fk = franka_fk(list(q))
            if fk[2, 3] > 0.03:
                continue
            r_diff = anchor[:3, :3].T @ fk[:3, :3]
            ang = math.acos(min(1.0, max(-1.0, (float(np.trace(r_diff)) - 1.0) / 2.0)))
            worst = max(worst, ang)
        return worst

    import math

    twisted = build(0.6)
    plain = build(0.0)
    assert twisted is not None and plain is not None
    # At the press depth the twisted release reaches the +0.6 rad shear
    # orientation; the plain one never leaves the anchor orientation.
    assert max_yaw_dev_at_press_depth(twisted) == pytest.approx(0.6, abs=0.1)
    assert max_yaw_dev_at_press_depth(plain) < 0.2


def test_press_lift_creeps_off_the_seat_first():
    """Press-mode LIFT must creep the first ~2 cm off the press seat at a
    very fine step before rising to clearance — a marginal weld peels
    exactly in that first stretch."""
    from ebim_task2.perception import TargetPose

    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    pad = TargetPose(x=500.0, y=450.0, yaw=0.0, visible=True)
    pol = _press_policy(transport_step_rad=0.012)
    pol.pad_gt_z_top = 0.102
    # Realistic LIFT context: start from the press-depth pose (the GRASP
    # trajectory's end), not the high home pose.
    press_traj = pol._build_trajectory(Phase.GRASP, home, None, pad)
    assert press_traj
    seat = np.asarray(press_traj[-1], dtype=np.float64)
    traj = pol._build_trajectory(Phase.LIFT, seat, None, pad)
    assert traj is not None and len(traj) >= 2
    zs = _z_profile(traj)
    from ebim_task2.motion import franka_fk

    z0 = float(franka_fk(list(seat))[2, 3])
    assert zs[-1] > z0 + 0.05  # rises to clearance
    # The first transport-sized step may only appear after ~2 cm of creep.
    big = next((i for i, (a, b) in enumerate(zip(traj, traj[1:]))
                if max(abs(x - y) for x, y in zip(a, b)) > 0.008), None)
    assert big is not None
    assert zs[big] >= z0 + 0.018


def test_unpressable_floor_sheet_routes_to_push_recovery():
    """A sheet bulldozed to the bare floor cannot be re-pressed within the
    USD joint limits: the attach check must route to PUSH recovery instead
    of regrasping."""
    from ebim_task2.perception import TargetPose

    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    pad = TargetPose(x=500.0, y=450.0, yaw=0.0, visible=True)
    pol = _press_policy()
    pol.pad_gt_z_top = 0.035  # floor pile: regrasp press goal ~0.025 < 0.032
    pol._phase = Phase.LIFT
    pol._trajectory = [home.copy()]
    pol._traj_idx = 1  # trajectory exhausted
    d = pol._transition(1.0, home, None, None, pad)
    assert d.phase == Phase.PUSH
    assert "push recovery" in d.reason
    assert pol._push_planned is False


def test_pressable_draped_sheet_still_regrasps():
    """A draped pile (z_top 0.05) keeps the normal regrasp path (its press
    goal stays above the feasible floor)."""
    from ebim_task2.perception import TargetPose

    home = np.asarray([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    pad = TargetPose(x=500.0, y=450.0, yaw=0.0, visible=True)
    pol = _press_policy()
    pol.pad_gt_z_top = 0.05  # press goal 0.05-0.008-0.002 = 0.040 >= 0.032
    pol._phase = Phase.LIFT
    pol._trajectory = [home.copy()]
    pol._traj_idx = 1
    d = pol._transition(1.0, home, None, None, pad)
    assert d.phase == Phase.PREGRASP
    assert "regrasp" in d.reason
