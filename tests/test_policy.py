"""Tests for the fixed-waypoint FSM policy (scaffold contract)."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebim_task2.policy import Phase, Task2WaypointPolicy  # noqa: E402
from ebim_task2.perception import PlacementObservation  # noqa: E402


def _policy():
    return Task2WaypointPolicy(
        {
            "pregrasp": [0.0] * 7,
            "grasp": [0.1] * 7,
            "lift": [0.2] * 7,
            "preplace": [0.3] * 7,
            "place": [0.4] * 7,
        },
        joint_tolerance_rad=0.05,
        waypoint_timeout_s=5.0,
        verification_timeout_s=3.0,
        min_iou=0.85,
        liner_dominance_ratio=0.90,
    )


def test_waits_for_state_when_no_joints():
    p = _policy()
    p.start(0.0)
    d = p.step(0.1, None, None)
    assert d.phase == Phase.WAIT_FOR_STATE
    assert d.arm_target is None


def test_starts_on_first_joint_state():
    p = _policy()
    p.start(0.0)
    # First joint state enters the FSM at PREGRASP whose target is [0]*7; since
    # we are already at [0]*7 the policy auto-advances to GRASP.
    d = p.step(0.1, [0.0] * 7, None)
    assert d.phase == Phase.GRASP
    assert d.arm_target == tuple([0.1] * 7)


def test_advances_when_waypoint_reached():
    p = _policy()
    p.start(0.0)
    p.step(0.1, [0.0] * 7, None)  # -> PREGRASP (reached)
    d = p.step(0.2, [0.0] * 7, None)
    assert d.phase == Phase.GRASP


def test_fails_open_on_waypoint_timeout():
    p = _policy()
    p.start(0.0)
    # Start at home, then never move; pregrasp target is [0]*7 which IS reached,
    # so jump to grasp target [0.1]*7 which we will not reach.
    p.step(0.1, [0.0] * 7, None)  # PREGRASP reached
    p.step(0.2, [0.0] * 7, None)  # -> GRASP, target [0.1]*7
    # now wait past timeout (5s)
    d = p.step(6.0, [0.0] * 7, None)
    assert d.phase == Phase.FAILED
    assert d.gripper_open_fraction == 1.0


def test_gripper_closed_during_transport():
    p = _policy()
    p.start(0.0)
    p.step(0.1, [0.0] * 7, None)  # PREGRASP
    p.step(0.2, [0.0] * 7, None)  # -> GRASP target [0.1]*7
    d = p.step(0.3, [0.1] * 7, None)  # reach grasp -> advance to LIFT
    assert d.phase == Phase.LIFT
    assert d.gripper_open_fraction == 0.0


def test_verify_succeeds_on_good_observation():
    """End-to-end: place waypoint reached -> RELEASE -> (settle) -> VERIFY -> SUCCEEDED.

    Regression for the dead-path bug where VERIFY was never reached without a
    manual ``enter_verify`` call, so every episode timed out at FAILED.
    """
    p = _policy()
    p.start(0.0)
    # Land exactly on each waypoint so the FSM advances through PLACE -> RELEASE.
    seq = [
        ([0.0] * 7, Phase.PREGRASP),  # enter FSM at pregrasp, reached
        ([0.0] * 7, Phase.GRASP),     # -> grasp
        ([0.1] * 7, Phase.LIFT),      # -> lift
        ([0.2] * 7, Phase.PREPLACE),  # -> preplace
        ([0.3] * 7, Phase.PLACE),     # -> place
        ([0.4] * 7, Phase.RELEASE),   # -> release
    ]
    t = 0.0
    last = None
    for joints, _ in seq:
        last = p.step(t, joints, None)
        t += 0.1
    assert last is not None and last.phase == Phase.RELEASE
    # Advance past the release settle delay (default 1.0s) -> enters VERIFY,
    # then present a passing observation. NO manual enter_verify call.
    good = PlacementObservation(iou=0.9, liner_dominance_ratio=0.95,
                                pad_pixels=100, target_pixels=100)
    d = p.step(t + 1.2, [0.4] * 7, good)
    assert d.phase == Phase.SUCCEEDED


def test_release_transitions_to_verify_without_manual_call():
    """Even with no observation yet, RELEASE must move to VERIFY after settle."""
    p = _policy()
    p.start(0.0)
    for joints, _ in [
        ([0.0] * 7, Phase.PREGRASP),
        ([0.0] * 7, Phase.GRASP),
        ([0.1] * 7, Phase.LIFT),
        ([0.2] * 7, Phase.PREPLACE),
        ([0.3] * 7, Phase.PLACE),
        ([0.4] * 7, Phase.RELEASE),
    ]:
        p.step(0.0, joints, None)
    # still in RELEASE immediately
    d0 = p.step(0.2, [0.4] * 7, None)
    assert d0.phase == Phase.RELEASE
    # after settle -> VERIFY (awaits mask since obs is None)
    d1 = p.step(1.5, [0.4] * 7, None)
    assert d1.phase == Phase.VERIFY
