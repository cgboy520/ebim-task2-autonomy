"""Tests for the runner gates and the perception policy.

Covers: the ``control.enabled`` publish gate, the VLA verification wiring in
``Task2AutonomyNode.tick``, pad-vs-target localization in
``PerceptionPolicy._cartesian_goal``, the diagonal-blob PCA yaw, and the
IK-failure fail-closed path. Pure numpy, no ROS / GPU required (sensor_msgs is
stubbed through sys.modules).
"""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ebim_task2.vla_policy as vla_mod  # noqa: E402
from ebim_task2.calibration import rpy_to_rotation  # noqa: E402
from ebim_task2.config import (  # noqa: E402
    ControlConfig,
    SemanticIds,
    Task2Config,
    TopicConfig,
    Waypoints,
)
from ebim_task2.motion import IKResult, franka_fk  # noqa: E402
from ebim_task2.perception import (  # noqa: E402
    PlacementObservation,
    TargetPose,
    _centroid_axis,
    estimate_pad_pose,
    estimate_target_pose,
)
from ebim_task2.policy import Decision, Phase  # noqa: E402
from ebim_task2.runner import Task2AutonomyNode  # noqa: E402
from ebim_task2.vla_policy import PerceptionPolicy, VLAPolicy  # noqa: E402


# ---- fakes ---------------------------------------------------------------
class _RecPub:
    """Recording stand-in for a ROS publisher."""

    def __init__(self) -> None:
        self.msgs: list = []

    def publish(self, msg) -> None:
        self.msgs.append(msg)


class _FakeClockNow:
    def to_msg(self):
        return "stamp"

    @property
    def nanoseconds(self) -> int:
        return 0


class _FakeClock:
    def now(self) -> _FakeClockNow:
        return _FakeClockNow()


class _FakeNode:
    def __init__(self) -> None:
        self.destroyed = False

    def get_clock(self) -> _FakeClock:
        return _FakeClock()

    def destroy_node(self) -> None:
        self.destroyed = True


@pytest.fixture(autouse=True)
def _fake_sensor_msgs(monkeypatch):
    """Stub ``sensor_msgs.msg.JointState`` so publish paths run without ROS."""
    msg_mod = types.ModuleType("sensor_msgs.msg")

    class _JointState:
        def __init__(self) -> None:
            self.header = types.SimpleNamespace(stamp=None)
            self.name: list[str] = []
            self.position: list[float] = []

    msg_mod.JointState = _JointState  # type: ignore[attr-defined]
    pkg = types.ModuleType("sensor_msgs")
    pkg.msg = msg_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sensor_msgs", pkg)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", msg_mod)


def _cfg(enabled: bool) -> Task2Config:
    limits = [(-1.0, 1.0)] * 7 if enabled else []
    waypoints = (
        Waypoints(
            pregrasp=[0.0] * 7,
            grasp=[0.1] * 7,
            lift=[0.2] * 7,
            preplace=[0.3] * 7,
            place=[0.4] * 7,
        )
        if enabled
        else Waypoints()
    )
    return Task2Config(
        control=ControlConfig(enabled=enabled, safe_joint_limits_rad=limits),
        topics=TopicConfig(
            arm_state="/a",
            arm_command="/b",
            gripper_command="/c",
            semantic_image="/d",
        ),
        semantic_ids=SemanticIds(thermalpad=3, target=4, liner=5),
        waypoints=waypoints,
    )


def _wired_node(cfg: Task2Config) -> tuple[Task2AutonomyNode, _RecPub, _RecPub]:
    node = Task2AutonomyNode(cfg, policy=None)
    arm_pub, grip_pub = _RecPub(), _RecPub()
    node._arm_pub = arm_pub
    node._gripper_pub = grip_pub
    node._node = _FakeNode()
    return node, arm_pub, grip_pub


# ---- (a) control.enabled gates ALL publishing ----------------------------
def test_enabled_false_suppresses_all_publishing():
    node, arm_pub, grip_pub = _wired_node(_cfg(enabled=False))
    node._apply_decision(Decision(Phase.GRASP, (0.0,) * 7, 0.5, "t"), 1.0)
    assert arm_pub.msgs == []
    assert grip_pub.msgs == []


def test_enabled_true_publishes_arm_and_gripper():
    node, arm_pub, grip_pub = _wired_node(_cfg(enabled=True))
    node._apply_decision(Decision(Phase.GRASP, (0.0,) * 7, 0.5, "t"), 1.0)
    assert len(arm_pub.msgs) == 1
    assert len(grip_pub.msgs) == 1
    assert list(arm_pub.msgs[0].position) == [0.0] * 7
    # open-fraction 0.5 -> driver radians halfway between open 0.0 / closed 0.8
    assert grip_pub.msgs[0].position == [pytest.approx(0.4)]
    assert grip_pub.msgs[0].name == ["right_right_finger_joint"]


def test_epsilon_limit_overshoot_is_clamped_not_refused():
    """Live regression: hold decisions echo measured joints, and physics
    settles a joint ~0.3 mrad past its limit. That must clamp and publish,
    not refuse (a refusal would open the gripper mid-RELEASE)."""
    node, arm_pub, grip_pub = _wired_node(_cfg(enabled=True))
    target = (0.0, 0.0, 0.0, 0.0, -1.0003, 0.0, 0.0)  # j5 past -1.0 by 0.3 mrad
    node._apply_decision(Decision(Phase.RELEASE, target, 0.0, "hold"), 1.0)
    assert len(arm_pub.msgs) == 1
    assert arm_pub.msgs[0].position[4] == pytest.approx(-1.0)
    # gripper published its commanded fraction (closed), not the refuse-open
    assert grip_pub.msgs[0].position == [pytest.approx(0.8)]


def test_spine_rides_in_the_arm_message_and_is_repeated_every_tick():
    """The spine is a commanded dof carried inside the arm command message.

    Two things must hold. (1) A policy-supplied spine height reaches the
    wire — without it the arm base sits 0.2509 m below the board plane and
    every demonstrated waypoint is unreachable. (2) It is REPEATED in later
    messages even when the policy
    stops supplying one, because the bridge re-applies a group's cached
    targets each tick and drops the group after a 1 s command gap (see
    mirror_lay.send) — an intermittent channel makes the spine sag.
    """
    node, arm_pub, _ = _wired_node(_cfg(enabled=True))
    q = (0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0)

    node._apply_decision(Decision(Phase.GRASP, q, 0.0, "vla", spine=0.4858), 1.0)
    assert arm_pub.msgs[0].name[-1] == "franka_spine_vertical_joint"
    assert arm_pub.msgs[0].position[-1] == pytest.approx(0.4858)

    # no spine on this decision -> the held value must still be sent
    node._apply_decision(Decision(Phase.GRASP, q, 0.0, "vla"), 1.1)
    assert arm_pub.msgs[1].name[-1] == "franka_spine_vertical_joint"
    assert arm_pub.msgs[1].position[-1] == pytest.approx(0.4858)


def test_no_spine_channel_leaves_the_arm_message_untouched():
    """The waypoint/mirror policies manage the spine themselves, so a decision
    without a spine must not append a joint the message never had."""
    node, arm_pub, _ = _wired_node(_cfg(enabled=True))
    q = (0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0)
    node._apply_decision(Decision(Phase.GRASP, q, 0.0, "waypoint"), 1.0)
    assert len(arm_pub.msgs[0].name) == 7
    assert "franka_spine_vertical_joint" not in arm_pub.msgs[0].name


def test_learned_policy_overshoot_is_clamped_not_refused():
    """Live regression (10k ACT smoke): the checkpoint predicted j2 up to
    ~0.02 rad past the USD limit (its demos ride that limit in the scoop),
    and every tick was refused + gripper opened. Overshoots within
    LIMIT_CLAMP_TOL must clamp to the limit and publish."""
    node, arm_pub, grip_pub = _wired_node(_cfg(enabled=True))
    target = (0.0, 0.0, 0.0, 0.0, -1.02, 0.0, 0.0)  # j5 0.02 past -1.0
    node._apply_decision(Decision(Phase.GRASP, target, 0.0, "vla step"), 1.0)
    assert len(arm_pub.msgs) == 1
    assert arm_pub.msgs[0].position[4] == pytest.approx(-1.0)
    assert grip_pub.msgs[0].position == [pytest.approx(0.8)]


def test_large_limit_violation_clamps_and_never_opens_the_gripper():
    """A big excursion is clamped and published; the gripper is NOT touched.

    The recorded demos ride exactly ON a limit for part of the motion (the
    scripted controller's own commands were clamped by this same gate before
    being recorded), so a policy fitted to that plateau predicts just
    outside the bound routinely. Opening the gripper there drops the pad
    mid-carry. Clamping is safe on its own — the published target is inside
    the limits by construction, and max_joint_delta still bounds the
    per-tick step.
    """
    node, arm_pub, grip_pub = _wired_node(_cfg(enabled=True))
    target = (0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 0.0)  # j5 0.5 rad past the limit
    node._apply_decision(Decision(Phase.RELEASE, target, 0.0, "hold"), 1.0)
    assert len(arm_pub.msgs) == 1
    assert arm_pub.msgs[0].position[4] == pytest.approx(-1.0)  # clamped to limit
    # the commanded (closed) fraction, never the refuse-open 1.0
    assert grip_pub.msgs[0].position == [pytest.approx(0.8)]


# ---- gripper open-fraction -> driver-radian mapping ----------------------
@pytest.mark.parametrize(
    "fraction,expected_rad",
    [(1.0, 0.0), (0.0, 0.8), (0.5, 0.4)],
)
def test_gripper_open_fraction_maps_to_driver_rad(fraction, expected_rad):
    node, _, grip_pub = _wired_node(_cfg(enabled=True))
    node._apply_decision(Decision(Phase.GRASP, None, fraction, "t"), 1.0)
    assert len(grip_pub.msgs) == 1
    assert grip_pub.msgs[0].name == ["right_right_finger_joint"]
    assert grip_pub.msgs[0].position == [pytest.approx(expected_rad)]


# ---- (b) VLA verification wiring -----------------------------------------
def test_vla_episode_succeeds_via_verification_wiring():
    cfg = _cfg(enabled=False)
    pol = VLAPolicy()
    pol._min_iou = 0.8
    pol._min_liner = 0.8
    pol._verify_timeout = 6.0
    node = Task2AutonomyNode(cfg, pol)
    node._obs = PlacementObservation(
        iou=0.9, liner_dominance_ratio=0.9, pad_pixels=10, target_pixels=10
    )
    node._joints = [0.0] * 7
    d1 = node.tick(1.0)  # first tick starts the episode; gate already passes
    d2 = node.tick(2.0)
    assert d1 is not None and d1.phase == Phase.SUCCEEDED
    assert d2 is not None and d2.phase == Phase.SUCCEEDED
    assert pol.is_terminal


# ---- (c) pad-vs-target localization + top-down frame ----------------------
def test_cartesian_goal_uses_pad_for_grasp_and_target_for_place():
    mask = np.zeros((480, 640), dtype=np.int32)
    mask[190:210, 190:210] = 3  # thermalpad near (x=200, y=200)
    mask[190:210, 210:230] = 5  # liner adjacent
    mask[340:360, 490:510] = 4  # target far away near (x=500, y=350)

    pad_pose = estimate_pad_pose(mask, thermalpad_id=3, liner_id=5)
    target_pose = estimate_target_pose(mask, target_id=4, liner_id=5)
    assert pad_pose.visible and target_pose.visible

    pol = PerceptionPolicy()
    grasp = pol._cartesian_goal(Phase.GRASP, target_pose, pad_pose)
    place = pol._cartesian_goal(Phase.PLACE, target_pose, pad_pose)
    assert grasp is not None and place is not None

    # Default camera mapping: x = (px - 320)/1000, y = (py - 240)/1000.
    pad_cx, pad_cy = 209.5, 199.5  # centroid of the pad ∪ liner blob
    tgt_cx, tgt_cy = 499.5, 349.5  # centroid of the target blob
    assert grasp[0, 3] == pytest.approx((pad_cx - 320.0) / 1000.0, abs=1e-6)
    assert grasp[1, 3] == pytest.approx((pad_cy - 240.0) / 1000.0, abs=1e-6)
    assert place[0, 3] == pytest.approx((tgt_cx - 320.0) / 1000.0, abs=1e-6)
    assert place[1, 3] == pytest.approx((tgt_cy - 240.0) / 1000.0, abs=1e-6)

    # Top-down frame: the third rotation column is (0, 0, -1) — tool Z down.
    for goal in (grasp, place):
        assert goal[2, 2] == -1.0
        assert goal[0, 2] == 0.0
        assert goal[1, 2] == 0.0


# ---- (d) diagonal-blob PCA yaw --------------------------------------------
def test_centroid_axis_diagonal_blob_yaw_is_pi_over_4():
    mask = np.zeros((64, 64), dtype=np.int32)
    idx = np.arange(10, 50)
    mask[idx, idx] = 1
    axis = _centroid_axis(mask == 1)
    assert axis is not None
    assert abs(abs(axis[2]) - np.pi / 4) < 0.01


# ---- camera flip_y (top-down camera: image +v -> world -Y) -----------------
def test_cartesian_goal_flip_y_mirrors_image_v():
    # A pixel ABOVE the principal point (v < cy) must map to world y > origin
    # when flip_y is on, and keep the old sign when off.
    pad = TargetPose(x=640.0, y=200.0, yaw=0.0, visible=True)
    pol_flip = PerceptionPolicy(cam_cx=640.0, cam_cy=360.0, pixel_scale=1000.0,
                                flip_y=True)
    pol_plain = PerceptionPolicy(cam_cx=640.0, cam_cy=360.0, pixel_scale=1000.0,
                                 flip_y=False)
    goal_flip = pol_flip._cartesian_goal(Phase.GRASP, None, pad)
    goal_plain = pol_plain._cartesian_goal(Phase.GRASP, None, pad)
    assert goal_flip[1, 3] == pytest.approx(0.16)
    assert goal_plain[1, 3] == pytest.approx(-0.16)


# ---- world_to_base transform applied before IK -----------------------------
def test_world_to_base_transform_applied_before_ik(monkeypatch):
    captured: dict[str, np.ndarray] = {}

    def _capture_solver(pose, seed, **kwargs) -> IKResult:
        captured["goal"] = np.asarray(pose, dtype=np.float64).copy()
        return IKResult(False, np.zeros(7), 9.9, 0)

    monkeypatch.setattr(vla_mod, "resolve_solver", lambda: _capture_solver)
    t_world_base = np.eye(4)
    t_world_base[:3, 3] = [0.1, 0.2, 0.3]
    pol = PerceptionPolicy()
    pol.world_to_base = t_world_base
    pol.start(0.0)
    pol.step(0.1, [0.0] * 7, None)  # no mask -> home-FK hover fallback goal
    from ebim_task2.motion import franka_fk

    untransformed = franka_fk(pol._home)
    untransformed[2, 3] = pol._clear_h
    assert np.allclose(
        captured["goal"][:3, 3], untransformed[:3, 3] + [0.1, 0.2, 0.3],
        atol=1e-12,
    )


# ---- (e) IK failure fails closed ------------------------------------------
def _bad_solver(pose, seed, **kwargs) -> IKResult:
    return IKResult(False, np.zeros(7), 9.9, 0)


def test_ik_failure_fails_closed_at_phase_begin(monkeypatch):
    monkeypatch.setattr(vla_mod, "resolve_solver", lambda: _bad_solver)
    pol = PerceptionPolicy()
    pol.start(0.0)
    d = pol.step(0.1, [0.0] * 7, None)
    assert d.phase == Phase.FAILED
    assert pol.is_terminal
    assert d.arm_target is None
    assert d.gripper_open_fraction == 1.0


def test_ik_failure_fails_closed_during_transition(monkeypatch):
    monkeypatch.setattr(vla_mod, "resolve_solver", lambda: _bad_solver)
    pol = PerceptionPolicy()
    pol.start(0.0)
    # Force the PREGRASP -> GRASP transition on the next step.
    pol._phase = Phase.PREGRASP
    pol._trajectory = []
    d = pol.step(0.1, [0.0] * 7, None)
    assert d.phase == Phase.FAILED
    assert pol.is_terminal
    assert d.arm_target is None
    assert d.gripper_open_fraction == 1.0


# ---- IK near-solutions: accepted within 2 cm, fail closed above ------------
def test_ik_near_solution_accepted(monkeypatch):
    def _near_solver(pose, seed, **kwargs) -> IKResult:
        return IKResult(False, np.zeros(7), 0.01, 200)

    monkeypatch.setattr(vla_mod, "resolve_solver", lambda: _near_solver)
    pol = PerceptionPolicy()
    pol.start(0.0)
    d = pol.step(0.1, [0.0] * 7, None)
    # Near-solution (<= 2 cm) is accepted: a trajectory is produced and the
    # FSM keeps driving (it may already have advanced past PREGRASP).
    assert d.phase != Phase.FAILED
    assert not pol.is_terminal
    assert d.arm_target is not None and len(d.arm_target) == 7


def test_ik_above_two_cm_fails_closed(monkeypatch):
    def _far_solver(pose, seed, **kwargs) -> IKResult:
        return IKResult(False, np.zeros(7), 0.03, 200)

    monkeypatch.setattr(vla_mod, "resolve_solver", lambda: _far_solver)
    pol = PerceptionPolicy()
    pol.start(0.0)
    d = pol.step(0.1, [0.0] * 7, None)
    assert d.phase == Phase.FAILED
    assert pol.is_terminal
    assert d.arm_target is None


# ---- ee-pose Jacobian servo (GRASP/PLACE only) ------------------------------
def _servo_setup():
    """Fixed world: arm base at (1.4, 1.612, 0.55) identity yaw; flange pose
    fabricated from exact FK so the servo geometry is deterministic."""
    t_wb = np.eye(4)
    t_wb[:3, 3] = [1.4, 1.612, 0.55]
    q = np.array([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    pol = PerceptionPolicy()
    pol.world_to_base = np.linalg.inv(t_wb)
    pol.ee_world = t_wb @ franka_fk(q, tool_offset=0.0)  # flange (link8) world
    return pol, t_wb, q


def test_grasp_servo_correction_shrinks_tcp_error():
    pol, t_wb, q = _servo_setup()
    tcp_meas = (t_wb @ franka_fk(q))[:3, 3]  # FK default tool = TCP point
    goal = np.eye(4)
    goal[:3, 3] = tcp_meas + np.array([0.02, 0.01, 0.015])
    pol._phase = Phase.GRASP
    pol._trajectory = []  # exhausted -> servo hook fires
    pol._servo_left = 3
    pol._servo_goal_world = goal

    d = pol.step(0.1, q.tolist(), None)
    assert d.phase == Phase.GRASP
    assert d.arm_target is not None and "servo" in d.reason
    err_before = float(np.linalg.norm(goal[:3, 3] - tcp_meas))
    tcp_after = (t_wb @ franka_fk(np.asarray(d.arm_target)))[:3, 3]
    err_after = float(np.linalg.norm(goal[:3, 3] - tcp_after))
    assert err_after < err_before


def test_servo_skipped_without_ee_pose():
    pol, _, q = _servo_setup()
    pol.ee_world = None
    pol._phase = Phase.GRASP
    pol._trajectory = []
    pol._servo_left = 3
    pol._servo_goal_world = np.eye(4)
    assert pol._maybe_servo(q) is None


def test_servo_never_runs_outside_grasp_place():
    pol, _, q = _servo_setup()
    pol._phase = Phase.PREGRASP
    pol._trajectory = []
    pol._servo_left = 3
    pol._servo_goal_world = np.eye(4)
    assert pol._maybe_servo(q) is None


# ---- adaptive clearance ------------------------------------------------------
def test_adaptive_clearance_clamps_goal_z_within_reach():
    pol = PerceptionPolicy(clearance_height=0.95, cam_cx=0.0, cam_cy=0.0,
                           pixel_scale=1.0, origin_xy=(0.0, 0.0))
    # Spine low (arm-base z = 0): a 0.95 m clearance would exceed 0.76 m reach.
    pol.arm_base_world = np.array([2.0, 1.6, 0.0])
    pad = TargetPose(x=2.1, y=1.65, yaw=0.0, visible=True)
    goal = pol._cartesian_goal(Phase.PREPLACE, None, pad)
    dist = float(np.linalg.norm(goal[:3, 3] - pol.arm_base_world))
    assert dist <= 0.76 + 1e-9
    assert goal[2, 3] < 0.95  # the clamp actually engaged


def test_clearance_unclamped_without_arm_base_world():
    pol = PerceptionPolicy(clearance_height=0.95, cam_cx=0.0, cam_cy=0.0,
                           pixel_scale=1.0, origin_xy=(0.0, 0.0))
    pad = TargetPose(x=2.1, y=1.65, yaw=0.0, visible=True)
    goal = pol._cartesian_goal(Phase.PREPLACE, None, pad)
    assert goal[2, 3] == pytest.approx(0.95)  # legacy fixed height


# ---- calibration fallback (record-less eval) --------------------------------
def test_calibration_fallback_installs_static_transform():
    cfg = _cfg(enabled=False)
    pol = PerceptionPolicy()
    node = Task2AutonomyNode(cfg, pol)
    # The fallback gate runs on WALL time (the sim clock jumps right after
    # a scene reset) — simulate ros_init having run 11 wall-s ago.
    import time as _time

    node._calib_start_wall = _time.monotonic() - 11.0
    node.tick(1.0)
    assert pol.world_to_base is not None
    assert node._calib_fallback is True
    assert pol.arm_base_world == pytest.approx([4.75, 2.60, 0.51])
    expected = np.eye(4)
    expected[:3, 3] = [4.75, 2.60, 0.51]
    expected[:3, :3] = rpy_to_rotation(0.0, 0.0, -1.5708)
    assert np.allclose(pol.world_to_base, np.linalg.inv(expected), atol=1e-12)


def test_calibration_fallback_not_installed_before_timeout():
    cfg = _cfg(enabled=False)
    pol = PerceptionPolicy()
    node = Task2AutonomyNode(cfg, pol)
    import time as _time

    node._calib_start_wall = _time.monotonic()  # just now
    node.tick(5.0)  # within the timeout window
    assert pol.world_to_base is None


def test_ee_pose_rearms_after_fallback_install():
    """A measured transform supersedes the static fallback: ee_pose showing
    up after the fallback went in (post-reset clock jump vs slow discovery)
    must re-arm sampling instead of keeping the guess."""
    cfg = _cfg(enabled=False)
    pol = PerceptionPolicy()
    node = Task2AutonomyNode(cfg, pol)
    node._calib_done = True
    node._calib_fallback = True
    node._calib_samples.append(([0.0] * 7, np.eye(4)))
    assert node._rearm_calibration_if_fallback() is True
    assert node._calib_done is False
    assert node._calib_fallback is False
    assert node._calib_samples == []
    # A measured calibration (fallback=False) is never re-armed.
    node._calib_done = True
    assert node._rearm_calibration_if_fallback() is False
    assert node._calib_done is True


def test_episode_done_after_photo_retreat():
    """The main loop exits on episode_done: set once the policy is terminal
    AND the 8-s photo retreat has elapsed (rclpy.spin has no terminal
    break, so a FAILED policy would otherwise hang the container)."""
    cfg = _cfg(enabled=False)
    pol = VLAPolicy()
    pol._min_iou = 0.8
    pol._min_liner = 0.8
    pol._verify_timeout = 6.0
    node = Task2AutonomyNode(cfg, pol)
    node._obs = PlacementObservation(
        iou=0.9, liner_dominance_ratio=0.9, pad_pixels=10, target_pixels=10
    )
    node._joints = [0.0] * 7
    node.tick(1.0)  # SUCCEEDED via the verification wiring
    node.tick(2.0)  # terminal -> photo retreat starts here
    assert pol.is_terminal
    assert not node.episode_done
    node.tick(5.0)  # inside the 8-s retreat window
    assert not node.episode_done
    node.tick(11.0)  # 9 s past the retreat start -> done
    assert node.episode_done


# ---- (m) grasp-depth schedule + mode-agnostic attach verification ---------
def test_grasp_depth_schedule_is_configurable_and_capped():
    """The press/pinch depth below the GT sheet top is config-driven and its
    per-regrasp escalation stops at the configured cap (deeper seats weld
    too hard for the twist to shear off)."""
    pol = PerceptionPolicy(grasp_depth_m=0.010, grasp_depth_step_m=0.003,
                           grasp_depth_max_steps=2)
    pol.pad_gt_z_top = 0.100
    pad = TargetPose(x=320.0, y=240.0, yaw=0.0, visible=True)

    depths = []
    for regrasps in (0, 1, 2, 3, 4):
        pol._regrasps_done = regrasps
        goal = pol._cartesian_goal(Phase.GRASP, None, pad)
        assert goal is not None
        depths.append(round(goal[2, 3], 6))

    assert depths[0] == pytest.approx(0.090)   # 0.100 - 0.010
    assert depths[1] == pytest.approx(0.087)   # one 3 mm step
    assert depths[2] == pytest.approx(0.084)   # two steps
    assert depths[3] == pytest.approx(0.084)   # capped
    assert depths[4] == pytest.approx(0.084)


@pytest.mark.parametrize("mode", ["press", "pinch"])
def test_attach_verification_runs_for_both_carry_modes(mode):
    """A sheet still at pedestal height after the lift means the gripper is
    empty — true whether the carry was a squeeze (pinch) or a press bond, so
    the GT height gate must regrasp in both modes."""
    pol = PerceptionPolicy(grasp_mode=mode)
    pol._phase = Phase.LIFT
    pol.pad_gt_z_top = 0.10          # still on the pedestal -> not carried
    pol._regrasps_done = 0

    decision = pol._transition(1.0, np.zeros(7), None, None, None)

    assert decision.phase == Phase.PREGRASP
    assert pol._regrasps_done == 1


def test_placement_observation_exposes_bboxes_for_residual_reporting():
    """observe_placement carries the two boxes its IoU came from so the
    residual can be reported in millimetres — IoU 0.0 alone cannot tell a
    1.6 cm miss from a sheet that never moved."""
    from ebim_task2.perception import observe_placement

    mask = np.zeros((64, 64), dtype=np.int32)
    # Target first, pad on top: a top-down mask shows the pad occluding the
    # slot where they overlap, exactly as the eval camera sees it.
    mask[12:22, 25:45] = 4   # target slot
    mask[10:20, 10:30] = 2   # placed pad, offset north-west

    obs = observe_placement(mask, thermalpad_id=2, target_id=4, liner_id=5)

    assert obs.placed_bbox == (10, 19, 10, 29)
    assert obs.target_bbox == (12, 21, 25, 44)
    assert obs.iou == pytest.approx(40.0 / 360.0)


# ---- (n) height-aware pixel->world scale ---------------------------------
def test_scale_at_rescales_between_object_heights():
    """A top-down pinhole's pixels-per-metre is fx/(camera_z - object_z), so
    one flat scale is only right at the height it was calibrated at."""
    pol = PerceptionPolicy(pixel_scale=660.6, camera_height_m=1.95,
                           pixel_scale_plane_z_m=0.1006)

    assert pol._scale_at(0.1006) == pytest.approx(660.6)
    # fx = 660.6 * (1.95 - 0.1006) = 1221.6
    assert pol._scale_at(0.0012) == pytest.approx(1221.6 / 1.9488, rel=1e-4)
    # Lower object -> longer standoff -> fewer pixels per metre.
    assert pol._scale_at(0.0012) < pol._scale_at(0.1006)


def test_scale_at_is_inert_without_camera_height():
    """Unset camera_height keeps the single-scale behaviour every existing
    config relies on."""
    pol = PerceptionPolicy(pixel_scale=660.6)
    assert pol._scale_at(0.0) == pytest.approx(660.6)
    assert pol._scale_at(0.5) == pytest.approx(660.6)


def test_place_goal_uses_the_board_height_not_the_pad_height():
    """The place chain aims at the slot, which lives ~10 cm below the pad
    crest the scale was calibrated on; using the pad scale there biases the
    goal by millimetres on an 1.8 cm target."""
    pol = PerceptionPolicy(cam_cx=640.0, cam_cy=360.0, pixel_scale=660.6,
                           origin_xy=(0.837, -0.065), flip_y=True,
                           camera_height_m=1.95, pixel_scale_plane_z_m=0.1006,
                           pad_plane_z_m=0.100, target_plane_z_m=0.0012)
    target = TargetPose(x=616.0, y=319.0, yaw=0.0, visible=True)
    pad = TargetPose(x=615.0, y=515.0, yaw=0.0, visible=True)

    place = pol._cartesian_goal(Phase.PLACE, target, pad)
    grasp = pol._cartesian_goal(Phase.GRASP, target, pad)
    assert place is not None and grasp is not None

    slot_scale = pol._scale_at(0.0012)
    pad_scale = pol._scale_at(0.100)
    assert place[0, 3] == pytest.approx(0.837 + (616.0 - 640.0) / slot_scale)
    assert place[1, 3] == pytest.approx(-0.065 + (360.0 - 319.0) / slot_scale)
    assert grasp[0, 3] == pytest.approx(0.837 + (615.0 - 640.0) / pad_scale)


# ---- (o) press descent stays vertical ------------------------------------
def test_press_descent_resolves_ik_along_a_vertical_line(monkeypatch):
    """interpolate_waypoints lerps in JOINT space, so a single hop between two
    IK solutions 6 cm apart bows the tool sideways on the way down and sweeps
    the sheet off its tray before the fingertip lands. The descent must be
    re-solved every few millimetres so the commanded path is the vertical
    line it is meant to be."""
    seen: list[np.ndarray] = []

    def fake_solver(pose, seed, **kwargs):
        seen.append(np.asarray(pose).copy())
        return IKResult(q=np.zeros(7), success=True, pos_error=0.0,
                        iterations=1)

    monkeypatch.setattr(vla_mod, "resolve_solver", lambda: fake_solver)

    pol = PerceptionPolicy(grasp_mode="press", press_slow_final_m=0.06,
                           transport_step_rad=0.012)
    goal = np.eye(4)
    goal[:3, 3] = (0.80, -0.30, 0.096)

    traj = pol._build_press_trajectory(np.zeros(7), goal)
    assert traj is not None

    zs = [float(p[2, 3]) for p in seen]
    # First the approach 6 cm up, then a monotonic descent to the goal.
    assert zs[0] == pytest.approx(0.156)
    assert zs[-1] == pytest.approx(0.096)
    assert len(zs) >= 6, f"descent resolved in only {len(zs)} IK calls: {zs}"
    assert all(b <= a + 1e-9 for a, b in zip(zs, zs[1:])), zs
    # Every step stays within a few mm, and xy never moves.
    assert max(a - b for a, b in zip(zs, zs[1:])) <= 0.011
    assert all(p[0, 3] == pytest.approx(0.80) and p[1, 3] == pytest.approx(-0.30)
               for p in seen)


# ---- (p) official-evaluator parity prediction ----------------------------
def test_official_verdict_scores_liner_only_and_gates_the_tiebreak_at_090():
    """The upstream evaluator picks the pad blob
    by which pad class is PRESENT and breaks a both-present tie on LIVE
    pixel counts at 0.9 dominance. Exactly 0.9 is NOT dominant (strictly
    greater), so a placement whose underside shows 1-in-10 rows sits right
    on the boundary and stays sideways/0."""
    from ebim_task2.perception import predict_official_verdict

    mask = np.ones((120, 120), dtype=np.int32)
    mask[40:52, 20:100] = 3        # target slot
    mask[41:51, 22:98] = 5         # liner sitting on the slot, fully covering

    case, iou = predict_official_verdict(
        mask, thermalpad_id=2, target_id=3, liner_id=5)
    assert case == "liner_only"
    assert iou > 0.5

    mask[50:51, 22:98] = 2         # 1 of 10 liner rows flips to pad: exactly
    case2, iou2 = predict_official_verdict(  # 684/760 = 0.90, not > 0.90
        mask, thermalpad_id=2, target_id=3, liner_id=5)
    assert case2 == "sideways"
    assert iou2 == 0.0


def test_official_verdict_tiebreak_uses_live_ids_not_stale_hints():
    """A thin pad sliver against a dominant liner resolves both_liner_dominant
    through the LIVE ids — never through hardcoded hint ids (3 = TARGET
    pixels in this scene, which could never clear the bar). The replica must
    match the evaluator here, or official_run re-rolls scenes the evaluator
    would score."""
    from ebim_task2.perception import predict_official_verdict

    mask = np.ones((120, 120), dtype=np.int32)
    mask[40:52, 20:100] = 3        # target slot (would drown a hint count)
    mask[41:51, 22:98] = 5         # liner covering the slot
    mask[50:51, 22:30] = 2         # 8-px pad sliver: 684 liner vs 8 pad

    case, iou = predict_official_verdict(
        mask, thermalpad_id=2, target_id=3, liner_id=5)
    assert case == "both_liner_dominant"
    assert iou > 0.5


def test_official_verdict_target_override_scores_a_fully_covered_plate():
    """Since upstream 9ac717b the evaluator resolves the TARGET through the
    loose annotator: full extent, occlusion-proof. A mask-only replica
    under-scores exactly the good lays (a perfect cover leaves no visible
    target pixels -> no_target_bbox 0.0 on what the evaluator scores ~1.0).
    The tracked-frame override must restore the loose semantics."""
    from ebim_task2.perception import predict_official_verdict

    mask = np.ones((120, 120), dtype=np.int32)
    mask[41:51, 22:98] = 5         # liner fully covering: NO target visible

    case, iou = predict_official_verdict(
        mask, thermalpad_id=2, target_id=3, liner_id=5)
    assert case == "no_target_bbox"     # mask-only fallback: degenerate

    case2, iou2 = predict_official_verdict(
        mask, thermalpad_id=2, target_id=3, liner_id=5,
        target_bbox=(40, 51, 20, 99))   # the tracked plate's full extent
    assert case2 == "liner_only"
    assert iou2 > 0.5

    # And a partially visible target must not shrink the override either.
    mask[40:41, 20:100] = 3
    case3, iou3 = predict_official_verdict(
        mask, thermalpad_id=2, target_id=3, liner_id=5,
        target_bbox=(40, 51, 20, 99))
    assert case3 == "liner_only"
    assert iou3 == pytest.approx(iou2)


def test_official_run_parses_the_chains_target_aabb_line():
    """official_run reads the chain's ``final target AABB px`` marker into
    the (y0, y1, x0, x1) bbox convention, tolerating float pixels and
    ignoring unrelated lines."""
    from ebim_task2.official_run import parse_target_aabb_px

    line = "final target AABB px (571.3,292.9,645.0,322.1)\n"
    assert parse_target_aabb_px(line) == (293, 322, 571, 645)
    assert parse_target_aabb_px("crude bbox IoU estimate 0.61") is None
    assert parse_target_aabb_px("target AABB x 0.76..0.88") is None


# ---- (q) grasp yaw offset applies to the grasp chain only ----------------
def test_grasp_yaw_offset_rotates_only_the_grasp_chain():
    """Which way the jaws close is a robot convention, not a scene fact: if
    they close along the tool x the fingers land on the pad's 9.3 cm ENDS
    instead of across its 2 cm width. The offset must rotate the grasp poses
    and leave the place chain (which has no jaw-width constraint) alone."""
    pol = PerceptionPolicy(grasp_yaw_offset_rad=np.pi / 2)
    pad = TargetPose(x=320.0, y=240.0, yaw=0.0, visible=True)
    target = TargetPose(x=400.0, y=240.0, yaw=0.0, visible=True)

    grasp = pol._cartesian_goal(Phase.GRASP, target, pad)
    place = pol._cartesian_goal(Phase.PLACE, target, pad)
    assert grasp is not None and place is not None

    # yaw 0 + pi/2 -> tool x-axis swings from world +x to world +y.
    assert grasp[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert grasp[1, 0] == pytest.approx(1.0)
    # The place pose keeps the unrotated convention.
    assert place[0, 0] == pytest.approx(1.0)
    assert place[1, 0] == pytest.approx(0.0, abs=1e-9)
    # Both stay tool-down.
    assert grasp[2, 2] == -1.0 and place[2, 2] == -1.0


# ---- (r) pour-off release for a scooped pad ------------------------------
def test_pinch_release_includes_a_pour_off_tilt(monkeypatch):
    """The jaws saturate at ~0.82 rad and still cannot close on a 2 cm pad, so
    the carry is a scoop: the pad rides ON TOP of the fingertips. Lifting
    straight up carries it back out, so the pinch release has to tip the
    fingers over the placement."""
    poses: list[np.ndarray] = []

    def fake_solver(pose, seed, **kwargs):
        poses.append(np.asarray(pose).copy())
        return IKResult(q=np.zeros(7), success=True, pos_error=0.0, iterations=1)

    monkeypatch.setattr(vla_mod, "resolve_solver", lambda: fake_solver)

    anchor = np.eye(4)
    anchor[:3, :3] = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    anchor[:3, 3] = (0.80, 0.0, 0.008)

    pol = PerceptionPolicy(grasp_mode="pinch", release_tilt_rad=0.7,
                           release_shake_cycles=0, release_twist_rad=0.0,
                           release_wipe_m=0.0)
    pol._place_goal_world = anchor
    traj = pol._build_release_trajectory(np.zeros(7), None, None)
    assert traj

    # A tilted pose is one whose tool z-axis is no longer straight down.
    tilted = [p for p in poses if abs(p[2, 2] + 1.0) > 1e-6]
    assert tilted, "release never tips the fingers over"
    # and the tilt is about the tool x-axis, so that axis is untouched
    assert tilted[0][0, 0] == pytest.approx(1.0)
    # ...and the chain comes back level before rising.
    assert abs(poses[-1][2, 2] + 1.0) < 1e-6


def test_press_release_unaffected_by_the_pinch_pour_off(monkeypatch):
    """press mode already had its own tilt segment; the pinch addition must
    not duplicate or reorder it."""
    poses: list[np.ndarray] = []

    def fake_solver(pose, seed, **kwargs):
        poses.append(np.asarray(pose).copy())
        return IKResult(q=np.zeros(7), success=True, pos_error=0.0, iterations=1)

    monkeypatch.setattr(vla_mod, "resolve_solver", lambda: fake_solver)
    anchor = np.eye(4)
    anchor[:3, :3] = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    anchor[:3, 3] = (0.80, 0.0, 0.05)
    pol = PerceptionPolicy(grasp_mode="press", release_tilt_rad=0.0,
                           release_shake_cycles=0, release_twist_rad=0.0,
                           release_dump_enabled=False)
    pol._place_goal_world = anchor
    assert pol._build_release_trajectory(np.zeros(7), None, None)
    assert all(abs(p[2, 2] + 1.0) < 1e-6 for p in poses)


# ---- (s) re-shake approaches the anchor from above -----------------------
def test_reshake_reaches_clearance_before_descending(monkeypatch):
    """The re-shake starts from the tuck pose. Lerping in joint space straight
    to a low place anchor sweeps the tool through the board — live, the
    trajectory stalled at waypoint 8/67 and burned the entire 120 s release
    timeout. The first commanded pose must therefore be the clearance pose
    above the anchor, not the anchor itself."""
    zs: list[float] = []

    def fake_solver(pose, seed, **kwargs):
        zs.append(float(pose[2, 3]))
        return IKResult(q=np.zeros(7), success=True, pos_error=0.0, iterations=1)

    monkeypatch.setattr(vla_mod, "resolve_solver", lambda: fake_solver)

    anchor = np.eye(4)
    anchor[:3, :3] = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    anchor[:3, 3] = (0.80, 0.0, 0.012)
    pol = PerceptionPolicy(release_lift_m=0.03)
    pol._place_goal_world = anchor

    assert pol._build_reshake_trajectory(np.zeros(7)) is not None
    assert zs, "re-shake solved nothing"
    # First goal is the high clearance pose, and it is well above the anchor.
    assert zs[0] == pytest.approx(0.262)
    assert zs[0] > anchor[2, 3] + 0.2


# ---- (t) the pin presses ON the sheet, not at a guessed distance ---------
def test_pin_reach_tracks_the_sheet_footprint(monkeypatch):
    """The pad is a 9.3 cm bar at rest but a ~4 cm wad by release time, so
    a fixed pin reach can land past its edge and pin bare board. The reach
    has to come from the measured footprint."""
    planned: list[tuple[float, float]] = []

    def fake_solver(pose, seed, **kwargs):
        planned.append((float(pose[0, 3]), float(pose[1, 3])))
        return IKResult(q=np.zeros(7), success=True, pos_error=0.0, iterations=1)

    monkeypatch.setattr(vla_mod, "resolve_solver", lambda: fake_solver)

    def _plan(half_extent):
        pol = PerceptionPolicy(pin_enabled=True, pin_offset_xy=[0.035, 0.0],
                               pin_press_z=0.002, pin_clear_z=0.18)
        pol.left_world_to_base = np.eye(4)
        anchor = np.eye(4)
        anchor[:3, 3] = (0.806, -0.010, 0.012)
        pol._place_goal_world = anchor
        pol.pad_world_gt = (0.806, -0.010)
        pol.pad_gt_half_extent = half_extent
        planned.clear()
        pol._plan_pin()
        assert pol._left_state == "descend", "pin was skipped"
        return planned[0][0] - 0.806      # x displacement from the sheet

    # A bunched 4 cm wad: press just inside its edge, not 3.5 cm out.
    assert _plan((0.020, 0.020)) == pytest.approx(0.015, abs=1e-6)
    # A flat 9.3 cm bar: reach further, capped.
    assert _plan((0.0465, 0.010)) == pytest.approx(0.0365, abs=1e-6)
    # No footprint measured -> fall back to the configured offset magnitude.
    assert _plan(None) == pytest.approx(0.035, abs=1e-6)


def test_pin_uses_the_live_sheet_centroid_as_its_origin(monkeypatch):
    """The sheet rides the right fingertips at release time, so its measured
    centroid — not the nominal place anchor — is where the pin displacement
    starts from."""
    planned: list[tuple[float, float]] = []

    def fake_solver(pose, seed, **kwargs):
        planned.append((float(pose[0, 3]), float(pose[1, 3])))
        return IKResult(q=np.zeros(7), success=True, pos_error=0.0, iterations=1)

    monkeypatch.setattr(vla_mod, "resolve_solver", lambda: fake_solver)
    pol = PerceptionPolicy(pin_enabled=True, pin_offset_xy=[0.035, 0.0],
                           pin_press_z=0.002, pin_clear_z=0.18)
    pol.left_world_to_base = np.eye(4)
    anchor = np.eye(4)
    anchor[:3, 3] = (0.806, -0.010, 0.012)
    pol._place_goal_world = anchor
    pol.pad_world_gt = (0.760, 0.040)          # sheet is NOT at the anchor
    pol.pad_gt_half_extent = (0.020, 0.020)
    pol._plan_pin()
    assert pol._left_state == "descend"
    assert planned[0][0] == pytest.approx(0.775, abs=1e-6)   # 0.760 + 0.015
    assert planned[0][1] == pytest.approx(0.040, abs=1e-6)


# ---- startup cadence check ------------------------------------------------
def _vla_cfg(rate: float, *, ckpt, dataset_fps=None) -> Task2Config:
    cfg = _cfg(enabled=True)
    cfg.control.publish_rate_hz = rate
    cfg.vla.backend = "lerobot"
    cfg.vla.checkpoint = str(ckpt)
    cfg.vla.dataset_fps = dataset_fps
    return cfg


def test_cadence_check_uses_declared_fps_when_the_dataset_is_unreachable(tmp_path):
    """A deployed checkpoint records its dataset by the TRAINING host's path
    (a container mount), so on the deployment host that path does not exist.
    `vla.dataset_fps` is what keeps the check a hard failure there instead
    of a warning — a mismatched publish rate replays the demonstrated lay
    at the wrong speed."""
    from ebim_task2.runner import _assert_cadence_matches_training

    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "train_config.json").write_text(
        '{"dataset": {"root": "/data/does_not_exist_here"}}'
    )

    with pytest.raises(SystemExit, match="1/3|0.333"):
        _assert_cadence_matches_training(_vla_cfg(10.0, ckpt=ckpt, dataset_fps=30.0))

    # matching rate passes
    _assert_cadence_matches_training(_vla_cfg(30.0, ckpt=ckpt, dataset_fps=30.0))


def test_cadence_check_prefers_the_checkpoints_own_dataset(tmp_path):
    """When the training dataset IS reachable its fps wins over the declared
    value, so a stale `dataset_fps` cannot mask a real mismatch."""
    from ebim_task2.runner import _assert_cadence_matches_training

    ds = tmp_path / "ds"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text('{"fps": 30}')
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "train_config.json").write_text(
        '{"dataset": {"root": "%s"}}' % ds.as_posix()
    )

    # declared fps is wrong (15) but the dataset says 30 -> 30 is used
    with pytest.raises(SystemExit):
        _assert_cadence_matches_training(_vla_cfg(15.0, ckpt=ckpt, dataset_fps=15.0))
    _assert_cadence_matches_training(_vla_cfg(30.0, ckpt=ckpt, dataset_fps=15.0))


def test_cadence_check_is_a_no_op_without_a_checkpoint():
    from ebim_task2.runner import _assert_cadence_matches_training

    _assert_cadence_matches_training(_vla_cfg(10.0, ckpt="", dataset_fps=None))
