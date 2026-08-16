"""Tests for the whole-body observation feed used by LeRobot-trained policies.

A fine-tuned checkpoint is only as good as the observation it is handed: the
37-D state row must carry the same channels, in the same order and units, as
``services/recording/record_task2.py`` wrote during training. A silent
off-by-one here trains fine and fails only at deployment, so the layout is
pinned by test.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebim_task2.observation import (  # noqa: E402
    GRIPPER_CLOSED_RAD,
    LEFT_JOINTS,
    RIGHT_JOINTS,
    STATE_DIM,
    VLAObservationCollector,
    gripper_open_fraction,
    quaternion_yaw,
    resolve_joint,
)


def test_state_row_matches_the_recorder_layout():
    col = VLAObservationCollector()
    col.set_ee_pose("left", (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0))
    col.set_ee_pose("right", (0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0))
    col.set_joints(
        LEFT_JOINTS + RIGHT_JOINTS
        + ["franka_spine_vertical_joint",
           "left_right_finger_joint", "right_right_finger_joint"],
        [1.0] * 7 + [2.0] * 7 + [0.35, 0.0, GRIPPER_CLOSED_RAD],
    )
    col.set_odom(7.0, 8.0, 0.0, 0.0, 0.0, 1.0, 0.11, 0.22, 0.33)

    s = col.state()

    assert s.shape == (STATE_DIM,)
    assert s.dtype == np.float32
    assert list(s[0:3]) == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert list(s[7:10]) == [pytest.approx(0.4), pytest.approx(0.5), pytest.approx(0.6)]
    assert np.allclose(s[14:21], 1.0)          # left arm
    assert np.allclose(s[21:28], 2.0)          # right arm
    assert s[28] == pytest.approx(0.35)        # spine
    assert s[29] == pytest.approx(1.0)         # left gripper fully OPEN at 0 rad
    assert s[30] == pytest.approx(0.0)         # right gripper closed at 0.8 rad
    assert list(s[31:33]) == [pytest.approx(7.0), pytest.approx(8.0)]
    assert s[33] == pytest.approx(0.0)         # identity quaternion -> yaw 0
    assert np.allclose(s[34:37], [0.11, 0.22, 0.33])


def test_ready_waits_for_the_active_arm_joints():
    """A zeroed arm row is a real (wrong) configuration, not a missing one —
    the collector must withhold the state until the joints have arrived."""
    col = VLAObservationCollector()
    assert not col.ready
    col.set_joints(RIGHT_JOINTS, [0.0] * 7)
    assert col.ready


def test_joint_name_variants_resolve_like_the_sim_bridge():
    """The robot USD publishes `*_fr3v2_1_joint*` on some builds; the recorder
    resolves both spellings and so must we, or the whole arm reads 0."""
    col = VLAObservationCollector()
    col.set_joints([f"right_fr3v2_1_joint{i}" for i in range(1, 8)], [0.5] * 7)
    assert col.ready
    assert np.allclose(col.state()[21:28], 0.5)
    assert resolve_joint({"left_fr3v2_finger_joint1": 0.4},
                         "left_right_finger_joint") == pytest.approx(0.4)


@pytest.mark.parametrize(
    "rad,expected", [(0.0, 1.0), (GRIPPER_CLOSED_RAD, 0.0), (0.4, 0.5),
                     (float("nan"), 0.0), (2.0, 0.0), (-1.0, 1.0)]
)
def test_gripper_open_fraction_matches_recorder_and_clamps(rad, expected):
    assert gripper_open_fraction(rad) == pytest.approx(expected)


def test_quaternion_yaw_matches_zyx_convention():
    half = math.pi / 4          # yaw = pi/2
    assert quaternion_yaw(0.0, 0.0, math.sin(half), math.cos(half)) == pytest.approx(
        math.pi / 2
    )
