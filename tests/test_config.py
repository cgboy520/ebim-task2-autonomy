"""Tests for config loading and fail-closed validation."""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebim_task2.config import load_config  # noqa: E402

CONFIG_PATH = ROOT / "config" / "task2.example.yaml"


def test_example_config_loads_with_control_disabled():
    cfg = load_config(CONFIG_PATH)
    assert cfg.control.enabled is False
    assert cfg.control.active_arm == "right"
    # empty waypoints tolerated while disabled
    assert cfg.waypoints.pregrasp == []
    assert cfg.waypoint_table["pregrasp"] == ()
    # new sections fall back to safe defaults
    assert cfg.camera.flip_y is False
    assert cfg.gripper.open_rad == 0.0 and cfg.gripper.closed_rad == 0.8
    assert cfg.policy.grasp_height_m == 0.02 and cfg.policy.clearance_height_m == 0.18
    assert cfg.calibration.enabled is True and cfg.calibration.samples == 20


def test_local_config_loads_enabled_with_real_scene_sections():
    cfg = load_config(ROOT / "config" / "task2.local.yaml")
    assert cfg.control.enabled is True
    assert cfg.control.active_arm == "right"
    assert len(cfg.control.safe_joint_limits_rad) == 7
    # USD joint limits, not the FR3 datasheet ones (j6 lower bound 0.44 > 0
    # catches an accidental datasheet regression).
    assert cfg.control.safe_joint_limits_rad[5][0] == 0.440
    assert cfg.control.safe_joint_limits_rad[3] == (-3.077, -0.117)
    assert cfg.control.travel_pose_on_start is True
    # five valid 7-DOF waypoints (placeholders for the perception policy)
    for name in ("pregrasp", "grasp", "lift", "preplace", "place"):
        assert len(cfg.waypoint_table[name]) == 7
    assert cfg.topics.arm_command == "/isaac/right_joint_commands"
    assert cfg.topics.gripper_command == "/isaac/right_robotiq_joint_commands"
    assert cfg.camera.cx == 640.0 and cfg.camera.cy == 360.0
    assert cfg.camera.flip_y is True
    assert cfg.gripper.open_rad == 0.0 and cfg.gripper.closed_rad == 0.8
    assert cfg.policy.grasp_height_m == 0.77
    assert cfg.policy.clearance_height_m == 0.95
    assert cfg.calibration.enabled is True
    assert cfg.calibration.samples == 20
    assert cfg.calibration.timeout_s == 10.0
    assert cfg.calibration.fallback_translation == [4.75, 2.60, 0.51]
    assert cfg.calibration.fallback_rotation_rpy == [0.0, 0.0, -1.5708]


def test_enabled_requires_limits_and_waypoints(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
        control:
          enabled: true
          active_arm: right
        topics:
          arm_state: /a
          arm_command: /b
          gripper_command: /c
          semantic_image: /d
        semantic_ids: {thermalpad: 3, target: 4, liner: 5}
        """,
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        load_config(p)


def test_valid_enabled_config_loads(tmp_path):
    p = tmp_path / "good.yaml"
    p.write_text(
        """
        control:
          enabled: true
          active_arm: left
          safe_joint_limits_rad: [[-2.0, 2.0],[-2.0,2.0],[-2.0,2.0],[-2.0,2.0],
                                   [-2.0,2.0],[-2.0,2.0],[-2.0,2.0]]
        topics:
          arm_state: /isaac/left_joint_states
          arm_command: /bridge/left_joint_commands
          gripper_command: /bridge/left_robotiq_joint_commands
          semantic_image: /isaac/eval_camera/semantic_segmentation
        semantic_ids: {thermalpad: 3, target: 4, liner: 5}
        waypoints:
          pregrasp: [0,0,0,0,0,0,0]
          grasp:    [0.1,0.1,0.1,0.1,0.1,0.1,0.1]
          lift:     [0.2,0.2,0.2,0.2,0.2,0.2,0.2]
          preplace: [0.3,0.3,0.3,0.3,0.3,0.3,0.3]
          place:    [0.4,0.4,0.4,0.4,0.4,0.4,0.4]
        """,
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.control.enabled is True
    assert cfg.control.active_arm == "left"
    assert len(cfg.control.safe_joint_limits_rad) == 7
    assert cfg.waypoint_table["grasp"] == (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)


def test_invalid_active_arm_rejected(tmp_path):
    p = tmp_path / "arm.yaml"
    p.write_text(
        """
        control: {enabled: false, active_arm: center}
        topics:
          arm_state: /a
          arm_command: /b
          gripper_command: /c
          semantic_image: /d
        semantic_ids: {thermalpad: 3, target: 4, liner: 5}
        """,
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        load_config(p)
