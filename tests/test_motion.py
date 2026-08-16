"""Tests for motion: forward kinematics, IK and trajectory interpolation."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebim_task2.motion import (  # noqa: E402
    franka_fk,
    interpolate_waypoints,
    solve_ik,
)


def test_fk_returns_homogeneous_pose():
    q = np.zeros(7)
    t = franka_fk(q)
    assert t.shape == (4, 4)
    # bottom row is homogeneous
    assert np.allclose(t[3], [0, 0, 0, 1])
    # at q=0 the d1 offset places the base link above the origin
    assert t[2, 3] > 0.0


def test_solve_ik_recovers_a_reachable_pose():
    home = np.array([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    target = franka_fk(home)
    # Solve from a perturbed seed; the DLS solver on the DH + numerical-jacobian
    # model should reduce the error well below the perturbation magnitude.
    seed = home + np.array([0.1, -0.1, 0.05, 0.05, 0.0, -0.05, 0.0])
    seed_err = float(np.linalg.norm(franka_fk(seed)[:3, 3] - target[:3, 3]))
    res = solve_ik(target, seed, max_iters=300, tol=1e-3)
    assert res.pos_error < seed_err          # strictly improved
    assert res.pos_error < 0.05              # within 5 cm (offline analytic model)
    rec = franka_fk(res.q)
    assert np.linalg.norm(rec[:3, 3] - target[:3, 3]) < 0.05


def test_solve_ik_respects_joint_limits():
    home = np.array([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
    target = franka_fk(home)
    limits = [(-0.05, 0.05)] * 7
    res = solve_ik(target, home, safe_joint_limits=limits, max_iters=50)
    assert all(lo <= v <= hi for v, (lo, hi) in zip(res.q, limits))


def test_interpolate_waypoints_scales_with_distance():
    start = np.zeros(7)
    goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    pts = interpolate_waypoints(start, goal, max_joint_step=0.05)
    assert len(pts) == 4  # ceil(0.2/0.05) = 4
    assert np.allclose(pts[-1], goal)
    # no single step exceeds the max joint step
    prev = start
    for p in pts:
        assert np.max(np.abs(p - prev)) <= 0.05 + 1e-9
        prev = p


def test_interpolate_waypoints_handles_zero_distance():
    q = np.array([0.1] * 7)
    pts = interpolate_waypoints(q, q, max_joint_step=0.05)
    assert len(pts) == 1
    assert np.allclose(pts[0], q)
