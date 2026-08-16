"""Room-scene support: the barebone chain runs in a VIRTUAL frame obtained
by conjugating every world quantity.

Room layout = rigid Rz(-90 deg) image of barebone about the table origin
(2.05, 1.95), work surface at +0.75; the target plate spawns 0.103 m off
that image.

    [vx, vy, vz] = A @ [x, y, z] + b,   A = Rz(+90),  b = (2.75, -2.05, -0.75)

- pad_points, ee poses and odom are mapped in the subscription callbacks.
- odom yaw maps to yaw + pi/2 (the room spawn's -90 deg reads as ~0).
- Eval-camera rasters are rotated 90 deg CCW. det(A) = +1: chirality kept.

Constraints:
- Arm base = 0.4991 above the robot root + spine. With the spine down the
  shoulder cluster rides at z~0.71 against the 0.69-0.76 table slab; at
  mirror_lay.SPINE_TARGET it clears the slab by ~0.44. Stations stay at
  virtual vx = PARK_X.
- Every official-pattern waypoint solves from those stations at the
  official spine height (seed family in mirror_lay.Q_SEED_OFF).
- scene_reset teleports the base back to the ROBOT SPAWN:
    scored spawn (2.1, 3.05) -> virtual (-0.300, +0.050), no corridor
    scene_room.py task2 preset (4.4, 2.60) -> virtual (+0.150, +2.350),
      12 legs, 2.35 m
  Launch with `--robot-x 2.1 --robot-y 3.05 --robot-z 0.0 --robot-yaw -90.0`.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ebim_task2 import lay_frame

# ---- room geometry (authoritative spawn table, scene_robot_room_keyboard)
TABLE_X, TABLE_Y = 2.05, 1.95
LIFT_Z = 0.75
CAM_X, CAM_Y, CAM_H = 2.087, 1.885, 2.7
ROOM_IMG_MIN_W = 1000        # barebone eval cam is 640 wide, room 1280

# focal-equivalent scale: S(plane z) = F_PX / (CAM_H - z)
F_PX = 1216.0
PLATE_TOP_Z = 0.7575
STRIP_TOP_Z = 0.85

# virtual <- room rigid map
_A = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_B = np.array([TABLE_X + 0.70, -TABLE_X, -LIFT_Z])   # (2.75, -2.05, -0.75)

# base_link virtual x for both stations (clears the swinging shoulder).
PARK_X = -0.24
# corridor leg length and done threshold (virtual m).
APPROACH_LEG_VY = 0.22
APPROACH_DONE_VY = 0.12

# Minimum error (m) before an unstick pulse fires; inside this radius the
# pulse only overshoots.
BASE_UNSTICK_MIN_ERR = 0.040

# Floor on base position tolerance (m); the base cannot resolve finer.
MIN_BASE_TOL = 0.020

# Max positional error (m) left uncorrected while the strip hangs;
# relocalise absorbs it.
LOADED_REVERSAL_MAX_ERR = 0.060

ACTIVE = False
_img_w = 1280       # room eval image width, re-read at activation

# Wall-clock dwell multiplier: dwells are stated at barebone's realtime
# factor; set from the live odom-stamp measurement at activation.
BAREBONE_SIM_FACTOR = 0.29
PACE = 1.8


def pace() -> float:
    return PACE if ACTIVE else 1.0


def set_pace_from_factor(sim_factor: float) -> None:
    """Adopt a measured sim factor; only a plausibly-slow one (wall-clock
    stamps read ~1.0 and would disable pacing)."""
    global PACE
    if not sim_factor or sim_factor <= 0.0 or sim_factor > 0.5:
        print(f"  room: implausible sim factor {sim_factor}; keeping the "
              f"default pace x{PACE:.2f}", flush=True)
        return
    PACE = min(max(BAREBONE_SIM_FACTOR / sim_factor, 1.0), 3.0)


# room target plate spawn, virtual coordinates (0.103 m off the rigid
# image of the barebone spawn)
PLATE_SPAWN_X, PLATE_SPAWN_Y = 0.80, 0.10

# Place-station offset from the target's virtual y. The four slots sit at
# y -0.10/0.00/+0.10/+0.20.
PLACE_STATION_DY = 0.22

def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def scene_of(img) -> str:
    """'room' | 'barebone' from the eval camera image (room is 1280 wide),
    overridable via EBIM_SCENE=room|barebone."""
    if img is not None and getattr(img, "shape", None) is not None:
        return scene_of_width(img.shape[1])
    return scene_of_width(None)


def scene_of_width(width) -> str:
    forced = os.environ.get("EBIM_SCENE", "").strip().lower()
    if forced in ("room", "barebone"):
        return forced
    if width is not None and width >= ROOM_IMG_MIN_W:
        return "room"
    return "barebone"


def v_pt(p):
    """Room world point -> virtual point (array-friendly: (...,3))."""
    a = np.asarray(p, dtype=np.float64)
    return a @ _A.T + _B


def v_pose(t_room: np.ndarray) -> np.ndarray:
    """Room world 4x4 pose -> virtual pose."""
    m = np.eye(4)
    m[:3, :3] = _A
    m[:3, 3] = _B
    return m @ t_room


def v_odom(x: float, y: float, z: float, yaw: float):
    p = v_pt((x, y, z))
    return float(p[0]), float(p[1]), float(p[2]), _wrap(yaw + math.pi / 2)


def gt_plate_virtual(payload) -> tuple[float, float, float] | None:
    """(vx, vy, slot_yaw) of `board_target` from the room ground-truth
    object-pose stream (`/isaac/task2/object_poses`, recording mode),
    conjugated into the virtual frame.

    None outside room mode, and whenever the payload is missing,
    unparseable or carries no such object. In BAREBONE the objects root IS
    the reference root: that stream holds one identity group and carries
    nothing to read.
    """
    if not ACTIVE or not payload:
        return None
    try:
        objs = json.loads(payload).get("objects") or {}
        p = objs.get("board_target")
        if not p or len(p) < 7:
            return None
        v = v_pt((float(p[0]), float(p[1]), float(p[2])))
        yaw = lay_frame.quat_wxyz_yaw(float(p[3]), float(p[4]),
                                      float(p[5]), float(p[6]))
    except (ValueError, TypeError, AttributeError):
        return None
    # same rule as v_odom: the conjugation is Rz(+90) about the table
    return (float(v[0]), float(v[1]),
            lay_frame.wrap_slot_yaw(yaw + math.pi / 2.0))


def orient_seg(seg):
    """Semantic raster into the virtual-frame (barebone-form) orientation."""
    if seg is None or not ACTIVE:
        return seg
    return np.ascontiguousarray(np.rot90(seg, 1))


def orient_rgb(img):
    if img is None or not ACTIVE:
        return img
    return np.ascontiguousarray(np.rot90(img, 1))


def _s(plane_z: float) -> float:
    return F_PX / (CAM_H - plane_z)


def _rot_consts(plane_z: float, cx: float, cy: float):
    """(PX0, PXS, PY0, PYS) of the ROTATED image at a work plane."""
    s = _s(plane_z)
    px0 = cy - (2.75 - CAM_Y) * s
    py0 = (_img_w - 1) - cx + (CAM_X - TABLE_X) * s
    return px0, s, py0, s


def plate_window():
    """find_plate_id centroid window in the rotated image (virtual plate
    spawn (0.80, +0.10))."""
    px0, s, py0, _ = _rot_consts(PLATE_TOP_Z, _cx, _cy)
    px = px0 + 0.80 * s
    py = py0 - 0.10 * s
    return ((px - 80.0, px + 80.0), (py - 80.0, py + 80.0))


def liner_crop():
    """(row0, row1, col0, col1) of the +-60 mm liner check crop around the
    strip's rest centre (virtual (0.80, -0.30), matching the barebone
    crop's own centre), in the rotated RGB."""
    px0, s, py0, _ = _rot_consts(STRIP_TOP_Z, _cx, _cy)
    px = px0 + 0.80 * s
    py = py0 - (-0.30) * s
    d = 0.060 * s
    return (int(py - d), int(py + d), int(px - d), int(px + d))


def liner_count(crop) -> int:
    """Liner pixels in the crop. The room render lights the liner blue
    (RGB ~ (70, 146, 211)), where barebone's cyan test reads 0."""
    b_r = crop[:, :, 2] - crop[:, :, 0]
    if not ACTIVE:
        return int(((b_r > 100) & (crop[:, :, 1] > 180)).sum())
    return int(((b_r > 60) & (crop[:, :, 2] > 150)).sum())


_cx, _cy = 640.0, 360.0


def _fit_center(seg_room) -> tuple[float, float, float]:
    """Camera centre from the plate marker blob at its known spawn
    (2.15, 1.95). Returns (cx, cy, residual_px); nominal on failure."""
    if seg_room is None:
        return 640.0, 360.0, -1.0
    s = _s(PLATE_TOP_Z)
    best = None
    for rid in np.unique(seg_room):
        ys, xs = np.nonzero(seg_room == rid)
        n = len(xs)
        if not (400 <= n <= 1400):
            continue
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        if not (6 <= w <= 22 and 55 <= h <= 100):
            continue
        cxb, cyb = float(xs.mean()), float(ys.mean())
        if best is not None:
            return 640.0, 360.0, -2.0     # ambiguous: keep nominal
        best = (cxb, cyb)
    if best is None:
        return 640.0, 360.0, -3.0
    cx = best[0] - (2.15 - CAM_X) * s
    cy = best[1] + (1.95 - CAM_Y) * s
    res = math.hypot(cx - 640.0, cy - 360.0)
    return cx, cy, res


def activate(img_w: int, seg_room=None) -> None:
    """Switch the module (and lay_frame's calibration) into room mode."""
    global ACTIVE, _img_w, _cx, _cy
    _img_w = int(img_w)
    _cx, _cy, res = _fit_center(seg_room)
    ACTIVE = True
    px0, pxs, py0, pys = _rot_consts(PLATE_TOP_Z, _cx, _cy)
    lay_frame.PX0, lay_frame.PXS = px0, pxs
    lay_frame.PY0, lay_frame.PYS = py0, pys
    print(f"** ROOM MODE: virtual-frame conjugation ON "
          f"(img_w {_img_w}, centre fit ({_cx:.1f},{_cy:.1f}) "
          f"res {res:.1f}px; plate-plane consts PX0 {px0:.1f} PXS {pxs:.1f} "
          f"PY0 {py0:.1f} PYS {pys:.1f})", flush=True)


_BB_CONSTS = (lay_frame.PX0, lay_frame.PXS, lay_frame.PY0, lay_frame.PYS)


def deactivate() -> None:
    """Restore barebone state (tests only; each attempt is a fresh process)."""
    global ACTIVE, _cx, _cy
    ACTIVE = False
    _cx, _cy = 640.0, 360.0
    (lay_frame.PX0, lay_frame.PXS,
     lay_frame.PY0, lay_frame.PYS) = _BB_CONSTS


def approach_legs(vy_now: float):
    """Corridor waypoints (virtual) from the spawn down to the park's
    latitude: get onto the vx=PARK_X line while still north of the table,
    then walk vy down in short legs."""
    legs = []
    if vy_now > APPROACH_DONE_VY:
        legs.append((PARK_X, vy_now))
        vy = vy_now
        while vy - APPROACH_LEG_VY > APPROACH_DONE_VY:
            vy -= APPROACH_LEG_VY
            legs.append((PARK_X, vy))
        legs.append((PARK_X, APPROACH_DONE_VY))
    return legs
