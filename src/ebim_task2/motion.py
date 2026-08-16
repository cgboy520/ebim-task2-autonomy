"""Motion generation: inverse kinematics and joint-space trajectory helpers.

The policy layer needs to turn a Cartesian placement target into joint angles.
On the official Ubuntu+GPU runtime a motion planner such as CuRobo or Isaac
Lab's IK is available; offline we cannot assume those, so this module ships a
self-contained damped-least-squares (DLS) IK on a Franka FR3 7-DOF kinematic
chain plus a linear joint-space interpolator.

The forward-kinematics model is EXACT for the EBiM Mobile FR3 Duo: it is built
from the official robot USD (``Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd``,
right arm ``right_fr3v2_joint1..7``) — every joint contributes
``Tr(p0) @ Rot(quat0) @ RotZ(q)`` per the USD Physics RevoluteJoint authoring,
followed by the link7→link8 (flange) offset from the USD's authored
zero-config poses. The flange→grasp-point tool offset is a calibrated
constant (see ``_TCP_OFFSET_Z``).

The real deployment may still wire :func:`solve_ik` to a simulator-backed
solver through :func:`load_solver`, keeping the call sites unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Exact FR3 kinematic chain (right arm), extracted from the official EBiM robot
# USD. Each entry: (p0, quat0) with quat0 = (w, x, y, z); the joint transform is
# Tr(p0) @ Rot(quat0) @ RotZ(q). Child-side frames are identity in the USD.
# ---------------------------------------------------------------------------
_FR3_USD_JOINTS: tuple[tuple[tuple[float, float, float], tuple[float, float, float, float]], ...] = (
    ((0.0, 0.0, 0.333), (1.0, 0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.70710677, -0.70710677, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (0.70710677, 0.70710677, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (0.70710677, 0.70710677, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (0.70710677, -0.70710677, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.70710677, 0.70710677, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (0.70710677, 0.70710677, 0.0, 0.0)),
)

# Flange (link8) offset along the link7 z-axis, from the USD's authored
# zero-config link poses: inv(world_link7) @ world_link8 == Tr(0, 0, 0.107).
_LINK7_TO_LINK8_Z = 0.107

# Tool offset: link8 -> grasp point along the flange z-axis
# (flange->fingertip reach, metres).
_TCP_OFFSET_Z = 0.161


def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def franka_fk(q: Sequence[float], *, tool_offset: float = _TCP_OFFSET_Z) -> np.ndarray:
    """Forward kinematics for the right FR3 arm, returning a 4x4 pose in the
    link0 (arm-base) frame. Exact against the official robot USD.

    ``tool_offset`` is the extra z offset beyond the flange (link8): pass
    ``0.0`` to get the FLANGE pose — required when calibrating against
    ``/isaac/*_ee_pose`` (which reports the link8 frame).
    """
    q = np.asarray(q, dtype=np.float64)
    t = np.eye(4)
    for i in range(7):
        (px, py, pz), (qw, qx, qy, qz) = _FR3_USD_JOINTS[i]
        jt = np.eye(4)
        jt[:3, 3] = (px, py, pz)
        jt[:3, :3] = _quat_to_rot(qw, qx, qy, qz) @ _rotz(float(q[i]))
        t = t @ jt
    # flange offset + tool offset, both along the local z-axis
    off = np.eye(4)
    off[2, 3] = _LINK7_TO_LINK8_Z + tool_offset
    return t @ off



def _pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """6-D pose error: 3 translational + 3 rotational (angle-axis).

    ``current`` and ``target`` are 4x4 homogeneous matrices.
    """
    dp = target[:3, 3] - current[:3, 3]
    rc = current[:3, :3]
    rt = target[:3, :3]
    r_err = rt @ rc.T
    # angle-axis from rotation matrix
    cos_angle = (np.trace(r_err) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if abs(angle) < 1e-9:
        drot = np.zeros(3)
    else:
        axis = np.array([r_err[2, 1] - r_err[1, 2],
                         r_err[0, 2] - r_err[2, 0],
                         r_err[1, 0] - r_err[0, 1]])
        axis = axis / (2.0 * math.sin(angle))
        drot = axis * angle
    return np.concatenate([dp, drot])


def _jacobian(q: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    """Numerical geometric Jacobian (6x7) at configuration ``q``."""
    q = np.asarray(q, dtype=np.float64)
    jac = np.zeros((6, 7))
    t0 = franka_fk(q)
    p0 = t0[:3, 3]
    r0 = t0[:3, :3]
    for i in range(7):
        dq = q.copy()
        dq[i] += eps
        ti = franka_fk(dq)
        pi = ti[:3, 3]
        ri = ti[:3, :3]
        # translational part
        jac[:3, i] = (pi - p0) / eps
        # rotational part: omega column = vee((dR/dq_i) @ R^T), the standard
        # finite-difference form of the geometric angular-velocity Jacobian.
        dr = ((ri - r0) / eps) @ r0.T
        jac[3, i] = dr[2, 1]
        jac[4, i] = dr[0, 2]
        jac[5, i] = dr[1, 0]
    return jac


@dataclass(frozen=True)
class IKResult:
    success: bool
    q: np.ndarray
    pos_error: float
    iterations: int


def solve_ik(
    target_pose: np.ndarray,
    seed: Sequence[float],
    *,
    safe_joint_limits: Sequence[tuple[float, float]] | None = None,
    max_iters: int = 200,
    tol: float = 1e-3,
    damping: float = 0.08,
    max_step: float = 0.2,
    ori_tol: float | None = None,
) -> IKResult:
    """Damped-least-squares IK to a 4x4 target pose.

    ``safe_joint_limits`` (7 ``(min, max)`` pairs) are *clipped* into after every
    step so the solution never violates the calibrated envelope. This is the
    analytic offline solver; production replaces it via :func:`load_solver`.

    Convergence is position-only (``tol``) unless ``ori_tol`` is set: then the
    orientation error must also fall below it.
    """
    q = np.asarray(seed, dtype=np.float64).copy()
    target_pose = np.asarray(target_pose, dtype=np.float64)
    limits = [tuple(p) for p in (safe_joint_limits or [])]

    def _clamp(arr: np.ndarray) -> np.ndarray:
        for i, (lo, hi) in enumerate(limits):
            if i < len(arr):
                arr[i] = min(max(arr[i], lo), hi)
        return arr

    q = _clamp(q)
    best_err = math.inf
    for it in range(1, max_iters + 1):
        t = franka_fk(q)
        err = _pose_error(t, target_pose)
        pos_err = float(np.linalg.norm(err[:3]))
        best_err = min(best_err, pos_err)
        ori_err = float(np.linalg.norm(err[3:]))
        if pos_err < tol and (ori_tol is None or ori_err < ori_tol):
            return IKResult(True, _clamp(q), pos_err, it)
        jac = _jacobian(q)
        # DLS update: dq = J^T (J J^T + lambda^2 I)^-1 e
        lam2 = damping * damping
        jjt = jac @ jac.T + lam2 * np.eye(6)
        dq = jac.T @ np.linalg.solve(jjt, err)
        step_norm = float(np.linalg.norm(dq))
        if step_norm > max_step:
            dq = dq * (max_step / step_norm)
        q = _clamp(q + dq)
    return IKResult(False, _clamp(q), best_err, max_iters)


# ---------------------------------------------------------------------------
# Pluggable solver backend. In the simulator env you point this at CuRobo /
# IsaacLab IK. Offline it stays None and solve_ik() is used.
# ---------------------------------------------------------------------------
SolverFn = Callable[..., IKResult]
_solver_override: SolverFn | None = None


def load_solver(solver: SolverFn | None) -> None:
    """Install a simulator-backed solver (e.g. CuRobo). Pass None to reset."""
    global _solver_override
    _solver_override = solver


def resolve_solver() -> SolverFn:
    return _solver_override if _solver_override is not None else solve_ik


# ---------------------------------------------------------------------------
# Joint-space trajectory interpolation
# ---------------------------------------------------------------------------
def interpolate_waypoints(
    start: Sequence[float],
    goal: Sequence[float],
    *,
    max_joint_step: float = 0.05,
) -> list[np.ndarray]:
    """Linear joint-space interpolation between two configurations.

    Returns a list of intermediate joint vectors (including the goal, excluding
    the start) such that no single joint moves more than ``max_joint_step``
    between consecutive samples. Used to move smoothly between waypoints.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    assert start.shape == goal.shape == (7,), "waypoints must be 7-DOF"
    delta = np.abs(goal - start)
    n = int(np.ceil(float(np.max(delta)) / max_joint_step)) if delta.size else 0
    n = max(n, 1)
    return [start + (goal - start) * (k / n) for k in range(1, n + 1)]
