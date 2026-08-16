"""Tests for runtime arm-base self-calibration (calibration.py). Pure numpy."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebim_task2.calibration import (  # noqa: E402
    estimate_world_to_arm_base,
    pose_msg_to_matrix,
    rpy_to_rotation,
)
from ebim_task2.motion import franka_fk  # noqa: E402


def test_rpy_to_rotation_yaw_only():
    r = rpy_to_rotation(0.0, 0.0, -1.5708)
    c, s = np.cos(-1.5708), np.sin(-1.5708)
    assert np.allclose(r, [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], atol=1e-12)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)


def test_pose_msg_to_matrix_identity():
    t = pose_msg_to_matrix((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert np.allclose(t[:3, :3], np.eye(3), atol=1e-12)
    assert np.allclose(t[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(t[3], [0, 0, 0, 1])


def test_pose_msg_to_matrix_yaw_90():
    # Quaternion for +90 deg about Z: (x, y, z, w) = (0, 0, sin45, cos45).
    s = np.sin(np.pi / 4)
    t = pose_msg_to_matrix((0.0, 0.0, 0.0), (0.0, 0.0, s, np.cos(np.pi / 4)))
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(t[:3, :3], expected, atol=1e-12)


def test_estimate_world_to_arm_base_recovers_known_transform():
    # Synthesize a fixed mount transform: translation (0.4, 0.3, 0.5) + yaw 30.
    t_world_armbase = np.eye(4)
    t_world_armbase[:3, 3] = [0.4, 0.3, 0.5]
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    t_world_armbase[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]

    rng = np.random.default_rng(42)
    samples = []
    for _ in range(10):
        q = rng.uniform(-0.5, 0.5, size=7)
        samples.append((q, t_world_armbase @ franka_fk(q, tool_offset=0.0)))

    t_est, spread = estimate_world_to_arm_base(samples)
    assert np.allclose(t_est[:3, 3], t_world_armbase[:3, 3], atol=1e-9)
    assert np.allclose(t_est[:3, :3], t_world_armbase[:3, :3], atol=1e-9)
    assert spread == pytest.approx(0.0, abs=1e-9)


def test_estimate_world_to_arm_base_reports_spread():
    # One sample perturbed by +1 cm in x -> spread reflects the inconsistency.
    t_mount = np.eye(4)
    t_mount[:3, 3] = [0.4, 0.3, 0.5]
    q = np.zeros(7)
    good = t_mount @ franka_fk(q, tool_offset=0.0)
    bad = good.copy()
    bad[0, 3] += 0.01
    _, spread = estimate_world_to_arm_base([(q, good), (q, good.copy()), (q, bad)])
    assert spread == pytest.approx(0.01 / 1.5, abs=1e-9)


def test_estimate_world_to_arm_base_rejects_empty():
    with pytest.raises(ValueError):
        estimate_world_to_arm_base([])
