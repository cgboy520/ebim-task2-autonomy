"""Target-frame geometry for the anchor-and-swing lay.

Board_Target is a FREE RIGID BODY (physics:rigidBodyEnabled, kinematic
off) resting on the table. scene_reset restores it to its spawn, and the
lay itself can push and rotate it. The evaluator scores the liner bbox
against the plate WHERE IT ENDS UP, so the lay runs in the plate's LIVE
frame: every uncommitted step re-aims at the freshly measured plate. The
plate is measured from the eval camera's STREAMED semantic segmentation —
the same render the evaluator scores from.

This module is the pure-math core: the (u, v) slot frame (u along the
plate axis, v across it) plus the semantic-image plate estimator. No rclpy
imports — everything stays unit-testable.
"""
from __future__ import annotations

import math

import numpy as np

# Slot plate half-extents (120 x 20 mm), for AABB estimates only.
HALF_U, HALF_V = 0.060, 0.010

# Eval-camera world->pixel map: px = PX0 + wx*PXS, py = PY0 - wy*PYS
# (image y is DOWN, so a world yaw +theta reads as image tilt -theta).
PX0, PXS = 114.2, 625.0
PY0, PYS = 317.2, 635.0

# Plate spawn pose after a scene_reset.
SPAWN_X, SPAWN_Y = 0.8029, -0.0028


def quat_wxyz_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Yaw about world +z of a wxyz quaternion (Isaac publishes wxyz)."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def band_lay_yaw(low) -> float | None:
    """Yaw of the axis a hanging strip will UNROLL along, from its bottom
    edge band.

    The band is the strip's WIDTH line (~20 mm across, a few mm deep), so
    the lay axis is that line's NORMAL. Returns None for a band too small
    to carry a direction. Wrapped to [-pi/2, pi/2]: the axis is 180-deg
    symmetric.
    """
    p = np.asarray(low, dtype=np.float64)
    if len(p) < 30:
        return None
    q = p[:, :2] - p[:, :2].mean(axis=0)
    evals, evecs = np.linalg.eigh(q.T @ q)
    if evals[1] <= 0.0:
        return None
    ax = evecs[:, 1]
    return wrap_slot_yaw(math.atan2(float(ax[1]), float(ax[0]))
                         + math.pi / 2.0)


def wrap_slot_yaw(theta: float) -> float:
    """Fold a yaw into [-pi/2, pi/2]: the slot is 180-deg symmetric."""
    while theta > math.pi / 2:
        theta -= math.pi
    while theta < -math.pi / 2:
        theta += math.pi
    return theta


# NOTE: /isaac/task2/object_poses is NOT a usable pose source in the
# barebone scene — its objects root is the reference root, so the only
# entry is the rigid GROUP ("task2_objects", identity). The plate must be
# measured from the eval camera's semantic stream instead (find_plate_id /
# measure_plate_id below).


class LayFrame:
    """Rotation+translation between world xy and the slot frame (u, v)."""

    def __init__(self, tx: float, ty: float, theta: float = 0.0) -> None:
        self.tx, self.ty, self.theta = float(tx), float(ty), float(theta)
        self.ca, self.sa = math.cos(self.theta), math.sin(self.theta)

    def to_world(self, u: float, v: float) -> tuple[float, float]:
        return (self.tx + self.ca * u - self.sa * v,
                self.ty + self.sa * u + self.ca * v)

    def to_uv(self, x: float, y: float) -> tuple[float, float]:
        dx, dy = x - self.tx, y - self.ty
        return (self.ca * dx + self.sa * dy, -self.sa * dx + self.ca * dy)

    def rot(self, du: float, dv: float) -> tuple[float, float]:
        """Rotate a DISPLACEMENT from the slot frame into world."""
        return (self.ca * du - self.sa * dv, self.sa * du + self.ca * dv)

    def aabb_half_extents(self, half_u: float = HALF_U,
                          half_v: float = HALF_V) -> tuple[float, float]:
        """World-axis-aligned half-extents of the rotated slot plate —
        what the official bbox IoU actually scores against."""
        return (half_u * abs(self.ca) + half_v * abs(self.sa),
                half_u * abs(self.sa) + half_v * abs(self.ca))


# Plate-estimate quality tiers (visible plate pixels).
PLATE_FULL_PX = 650     # centre + line both trustworthy
PLATE_LINE_PX = 250     # line (theta + lateral) only: the strip covers the
                        # plate 1:1 as it lays, so the visible centroid
                        # slides toward the unlaid end and the along-axis
                        # centre becomes unobservable


def find_plate_id(seg: np.ndarray,
                 *,
                 window: tuple[tuple[float, float], tuple[float, float]]
                 = ((540.0, 700.0), (240.0, 400.0)),
                 ) -> int | None:
    """Raw semantic id of the target plate, from a CLEAN view (chain
    start: the strip is still on the pedestal).

    Label-free: raw semantic ids drift between scene processes, so the
    plate is picked geometrically — the unique id whose region is
    plate-sized (550..1300 px), plate-shaped (width 55..95 px, height
    <= 45 px) and centred in the target window. Neighbouring slot rows sit
    at py 184..202 / 244..265 / 373..390 and the pedestal at py ~505, all
    outside the window; the board and chips classes span several rows and
    fail the height cap; the liner is on the pedestal in a clean view.
    Returns None when no id qualifies or two do. Once found, the id is
    stable for the scene process — later part-covered measurements track
    it with measure_plate_id and need no shape guards.
    """
    (wx0, wx1), (wy0, wy1) = window
    best = None
    for rid in np.unique(seg):
        ys, xs = np.nonzero(seg == rid)
        if not (550 <= len(xs) <= 1300):
            continue
        cx_, cy_ = float(xs.mean()), float(ys.mean())
        if not (wx0 <= cx_ <= wx1 and wy0 <= cy_ <= wy1):
            continue
        if not (55 <= xs.max() - xs.min() + 1 <= 95):
            continue
        if ys.max() - ys.min() + 1 > 45:
            continue
        if best is not None:
            return None
        best = int(rid)
    return best


def measure_plate_id(seg: np.ndarray, rid: int,
                     min_px: int = PLATE_LINE_PX
                     ) -> tuple[float, float, float, int, int] | None:
    """(x, y, yaw, n_px, n_cols) of the LOCKED plate id's visible pixels.

    The centroid is the true centre only when the plate is clean
    (n_px >= PLATE_FULL_PX); part-covered, it slides toward the unlaid
    end — callers must then keep their committed centre via
    project_center_to_line and take only the line (yaw + lateral). The
    yaw is a per-column centroid fit over the visible columns; n_cols
    tells the caller how much of the plate's length backed the fit."""
    ys, xs = np.nonzero(seg == rid)
    n = len(xs)
    if n < min_px:
        return None
    cols, counts = np.unique(xs, return_counts=True)
    if len(cols) > 8:
        cols, counts = cols[2:-2], counts[2:-2]
    # Column-height filter: the slot housing clips the plate's v-edges
    # SYMMETRICALLY (centroids unbiased), but an arm shadow clips rows on
    # ONE side and drags that column's centroid laterally. Keep only
    # columns near the frame's own healthy height.
    h_ref = float(np.percentile(counts, 85))
    keep = counts >= max(0.8 * h_ref, 4.0)
    if int(keep.sum()) < 8:
        return None
    cols, counts = cols[keep], counts[keep]
    sel = np.isin(xs, cols)
    xs_f, ys_f = xs[sel], ys[sel]
    cx_, cy_ = float(xs_f.mean()), float(ys_f.mean())
    cym = np.array([ys_f[xs_f == c].mean() for c in cols],
                   dtype=np.float64)
    slope = float(np.polyfit(cols.astype(np.float64), cym, 1)[0])
    theta = wrap_slot_yaw(-math.atan(slope))
    wx = (cx_ - PX0) / PXS
    wy = (PY0 - cy_) / PYS
    return wx, wy, theta, n, int(len(cols))


def project_center_to_line(cx_old: float, cy_old: float,
                           x_new: float, y_new: float, theta: float
                           ) -> tuple[float, float]:
    """Slide a committed plate centre onto a freshly fitted plate LINE
    (world point (x_new, y_new) + world direction theta), keeping its
    along-axis station.

    Used for the part-covered tier: theta and the lateral offset are still
    well-estimated from any visible plate segment, but the along-axis
    centre is not — so the old centre is projected onto the new line
    instead of trusting the visible centroid."""
    dx, dy = math.cos(theta), math.sin(theta)
    t = (cx_old - x_new) * dx + (cy_old - y_new) * dy
    return (x_new + t * dx, y_new + t * dy)
