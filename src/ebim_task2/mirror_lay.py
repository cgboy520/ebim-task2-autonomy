#!/usr/bin/env python3
"""mirror_lay.py — OFFICIAL-pattern pick, carry and place of the thermal strip.

Replicates the organizers' task2_fixpos_v1 demonstration motion (ep3),
spine included. All frames are the VIRTUAL frame (room
conjugation in :mod:`room_mode`); the TCP is the official grip point
+0.1498 m along tool z from the flange.

    SPINE  lift to the official 0.4858 m before anything else moves; that
           puts the arm base 0.2349 m ABOVE the board plane — the geometry
           every ep3 constant below was demonstrated in.
    PICK   side pinch of the strip's west drooping tab on the stand:
           tool z east (+x), pitch -5 deg, roll pi (jaws close vertically),
           TCP at (wall_west_face - 0.001, tab_y, 0.110); full close.
    PEEL   pull 19 mm west, then an up-east arc; the strip slides off the
           stand top and HANGS from the pinch (liner facing east).
    CARRY  pitch ramps -5 -> -37 deg; the hanging strip transits the
           corridor EAST of every board (TCP x ~0.874-0.885); the base
           rides to the target slot's station with the joints held.
    PLACE  descend at the target latitude until the strip's bottom edge
           contacts the board at target-centre +55 mm along the lay axis;
           press west-down at -37 deg (the official mid-sweep hold), then
           the pitch sweep -37 -> -63.7 deg along the ep3 TCP path lays
           the strip westward, liner up. FULL OPEN only once the pad
           reads flat (z_hi < flat-gate) at pitch <= open-arm.
    LEAVE  retreat at the final pitch, park via the seed family.

Everything Cartesian is closed-loop against ``/isaac/right_ee_pose``; the
arm base is re-derived from the live (q, ee) pair at every waypoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String

# runnable as a module or a bare script (package root from __file__)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from ebim_task2 import lay_frame as _lf  # noqa: E402
from ebim_task2 import room_mode  # noqa: E402
from ebim_task2.lay_frame import (  # noqa: E402
    LayFrame, PLATE_FULL_PX, band_lay_yaw, find_plate_id,
    measure_plate_id,
    project_center_to_line,
)
from ebim_task2.motion import (  # noqa: E402
    _TCP_OFFSET_Z, franka_fk, solve_ik,
)

JOINTS = 7
SIDE, OTHER = "right", "left"
SAFE = [(-2.901, 2.901), (-1.836, 1.836), (-2.901, 2.901), (-3.077, -0.117),
        (-2.876, 2.876), (0.440, 4.622), (-3.051, 3.051)]
TRAVEL = [0.0, -1.5, 0.0, -2.2, 0.0, 1.5, 0.785]

# base_step tokens that undo one another (see the loaded-reversal guard)
BASE_REVERSE = {"FWD": "BACK", "BACK": "FWD", "A": "B", "B": "A"}

OPEN_GRIP, CLOSED_GRIP = 0.0, 0.8
BOARD_Z = 0.0012

# ---- vertical spine: a commanded dof, carried in the arm command message
SPINE_JOINT = "franka_spine_vertical_joint"
SPINE_LIMITS = (0.0, 0.85)      # 0.1 m/s
# Arm base above the robot root at spine 0.
ARM_MOUNT_Z = 0.4991
# Official ep3 height. Arm base virtual z = ARM_MOUNT_Z + spine -
# room_mode.LIFT_Z (+0.2349 at 0.4858, -0.2509 at spine 0).
SPINE_TARGET = 0.4858
SPINE_TOL = 0.005
SPINE_WAIT_S = 120.0
SPINE_SAG_TRIES = 2             # the drive sags ~15 mm under the arm's load
SPINE_STALL_S = 20.0

# ---- official grasp/place geometry (task2_fixpos_v1 ep3, virtual frame) ----
# The pick stand (sticker_base.usda at table (-0.31,-0.04,0.017)): central
# wall x 0.760..0.840, 20 mm thick in y about -0.30, top edge z 0.097. The
# strip rests along x on the wall top and droops ~13 mm at each end.
WALL_W_X = 0.760
WALL_TOP_Z = 0.097
# Official grip point: +0.1498 m along tool z from the flange (the pinch
# holds the last ~4 mm of the west tab).
GRIP_REACH = 0.1498
# Official roll: tool y points DOWN (phi = pi in pinch_pose terms).
GRIP_PHI = math.pi
PITCH_GRASP = math.radians(-5.0)
PITCH_CARRY = math.radians(-37.15)
# ep3 peel: TCP path after the close, relative to the pinch point.
PEEL_WEST = -0.019
PEEL_ARC = ((-0.011, 0.016), (0.007, 0.030), (0.031, 0.039), (0.058, 0.048),
            (0.083, 0.058), (0.103, 0.069), (0.113, 0.078))
# Carry/corridor TCP (x is EAST of every board's east end; strip bottom
# clears the board tops by 25-70 mm through the transit).
CARRY_X, CARRY_Z = 0.874, 0.178
CORRIDOR_X, CORRIDOR_Z = 0.885, 0.155
# Place, in the target frame (u along the lay axis, +u = contact end):
# the strip's bottom edge lands at u = CONTACT_DU. TCP_BOTTOM_DX: ep3 hang
# offset (bottom edge west of the TCP at -37 deg).
CONTACT_DU = 0.055
TCP_BOTTOM_DX = 0.020
# Calibrated --contact-du default: CONTACT_DU plus a +24 mm along-axis
# translation.
CONTACT_DU_CAL = 0.079
# Per-slot place trims (fixed-base room mode; resolved from the target's
# virtual y; --lat-trim/--du-trim/--place-spine override).
# LAT: +v aim offset (world +y at board yaw 0), applied on top of the
# hover re-aim. PLACE_SPINE: spine height driven at the hover.
# Slots without an entry use the dv-zero defaults.
SLOT_LAT_TRIM: dict = {0.20: +0.012}
SLOT_DU_TRIM: dict = {0.20: -0.016}
SLOT_PLACE_SPINE: dict = {0.20: 0.40}
DESC_TCP_Z_FLOOR = 0.117     # bottom edge on the board when TCP reaches here
CONTACT_Z_GATE = 0.005       # strip z_lo below this = bottom edge in contact
# -37 deg press (the official inter-ramp hold, ~2.5 s): TCP (u, z).
PRESS = ((0.062, 0.130), (0.049, 0.130), (0.040, 0.105), (0.035, 0.099))
# pitch sweep rungs: (pitch deg, TCP u, TCP z) — the ep3 ramp-2 path.
SWEEP = ((-40.5, 0.034, 0.098), (-46.3, 0.035, 0.101), (-51.9, 0.027, 0.097),
         (-56.5, 0.014, 0.095), (-59.0, 0.001, 0.090), (-60.4, -0.012, 0.085),
         (-61.2, -0.026, 0.085), (-62.4, -0.041, 0.081), (-63.1, -0.057, 0.067),
         (-63.5, -0.071, 0.050), (-63.7, -0.084, 0.039))
# (u, v, z, pitch deg). The last leg re-pitches to -30 at z 0.32; from there
# the fold to the seed family and on to TRAVEL keeps every joint origin
# >= 0.181 above the board plane.
RETREAT = ((-0.050, -0.020, 0.150, -63.7), (-0.084, -0.100, 0.250, -63.7),
           (-0.100, -0.100, 0.320, -30.0))

# Joint family at the official spine height: grip point Q_SEED_TCP at
# pitch -5 / phi=pi. COMMANDED (reconf, park, bail), so it must be holdable.
# Solves every chain waypoint at every slot; joint origins >= 0.118 above
# the board plane; TRAVEL <-> family fold >= 0.167 m.
Q_SEED_TCP = (0.55, -0.299, 0.20)   # the grip point the family solves for
Q_SEED_OFF = [0.4705, -1.0845, -2.6077, -2.9749, -1.6034, 4.2992, -0.0286]
# Family reconfiguration sweep steps; the j1 reaction spins the base.
RECONF_STEPS = 180

# Castored-base yaw rate (rad/s) under an A+C / B+C pedal hold.
BASE_YAW_RATE = 0.69
BASE_YAW_MIN_PULSE = 0.06
# Achievable yaw resolution: the shortest pulse coasts ~0.08 rad.
BASE_YAW_FLOOR = 0.045
# Debug artefacts directory (override with EBIM_MIRROR_DEBUG_DIR).
DEBUG_DIR = os.environ.get("EBIM_MIRROR_DEBUG_DIR", "/root/ebim")
got: dict = {}


class N(Node):
    def __init__(self) -> None:
        super().__init__("lay_down")
        # spine target carried in every SIDE arm command (None = untouched)
        self.spine_cmd: float | None = None
        self.cmd, self.grip = {}, {}
        for s in ("left", "right"):
            self.cmd[s] = self.create_publisher(
                JointState, f"/isaac/{s}_joint_commands", 10)
            self.grip[s] = self.create_publisher(
                JointState, f"/isaac/{s}_robotiq_joint_commands", 10)
            self.create_subscription(
                JointState, f"/isaac/{s}_joint_states",
                lambda m, k=s: got.__setitem__(
                    f"{k}_q", [float(p) for p in m.position[:JOINTS]]), 10)
            self.create_subscription(
                PoseStamped, f"/isaac/{s}_ee_pose",
                lambda m, k=s: self._ee(k, m), 10)
        self.create_subscription(Float32MultiArray, "/isaac/task2/pad_points",
                                 self._pad, qos_profile_sensor_data)
        # ROOM ONLY: the benchmark's ground-truth object poses (reset-time
        # spawn record; static after a reset)
        self.create_subscription(
            String, "/isaac/task2/object_poses",
            lambda m: got.__setitem__("obj_poses", m.data), 10)
        # /pedal/state tokens: FWD/BACK = +/-x, A/B = +/-y at 0.5 m/s sim,
        # A+C/B+C = +/-yaw; 1.0 SIM-second watchdog.
        self.pedal = self.create_publisher(String, "/pedal/state", 10)
        from nav_msgs.msg import Odometry
        self.create_subscription(JointState, "/isaac/joint_states_full",
                                 self._full, 10)
        self.create_subscription(Odometry, "/isaac/odom", self._odom, 10)
        from sensor_msgs.msg import Image
        # store the RAW msg, decode at consumption (per-frame decode starves the loop)
        for key, topic in (("wrist_msg", "/isaac/right_wrist_camera/image_raw"),
                           ("eval_msg", "/isaac/eval_camera/image_raw")):
            self.create_subscription(
                Image, topic,
                lambda m, k=key: got.__setitem__(k, m),
                qos_profile_sensor_data)
        # eval-camera SEMANTIC stream: the plate is tracked in the scorer's own pixels
        self.create_subscription(
            Image, "/isaac/eval_camera/semantic_segmentation",
            lambda m: got.__setitem__("eval_sem", m),
            qos_profile_sensor_data)
        # LOOSE bbox stream: the evaluator's own target resolution, by class name
        try:
            from vision_msgs.msg import Detection2DArray
            self.create_subscription(
                Detection2DArray, "/isaac/eval_camera/bbox_2d_loose",
                lambda m: got.__setitem__("loose_bbox", m),
                qos_profile_sensor_data)
            self.create_subscription(
                String, "/isaac/eval_camera/bbox_2d_loose_labels",
                lambda m: got.__setitem__("loose_labels", m),
                qos_profile_sensor_data)
        except ImportError:      # no vision_msgs: geometric lock stands in
            print("  [no vision_msgs; target falls back to the geometric "
                  "window]", flush=True)

    def _ee(self, side: str, m) -> None:
        p, o = m.pose.position, m.pose.orientation
        w, x, y, z = o.w, o.x, o.y, o.z
        t = np.eye(4)
        t[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        t[:3, 3] = (p.x, p.y, p.z)
        if room_mode.ACTIVE:
            t = room_mode.v_pose(t)
        got[f"{side}_ee"] = t
        q = got.get(f"{side}_q")
        if q is not None:
            # pair q with the pose at arrival (independent reads tear mid-move)
            s = got.setdefault(f"{side}_samples", [])
            s.append((list(q), t.copy()))
            if len(s) > 64:
                del s[:-32]

    def _full(self, m) -> None:
        for nm, p in zip(m.name, m.position):
            if nm == "franka_spine_vertical_joint":
                got["spine"] = float(p)
                return

    def _odom(self, m) -> None:
        p, o = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2.0 * (o.w * o.z + o.x * o.y),
                         1.0 - 2.0 * (o.y * o.y + o.z * o.z))
        got["odom_stamp"] = (time.time(),
                             m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        if room_mode.ACTIVE:
            got["odom"] = room_mode.v_odom(p.x, p.y, p.z, yaw)
        else:
            got["odom"] = (float(p.x), float(p.y), float(p.z), yaw)

    def _pad(self, m) -> None:
        v = np.array(list(m.data)[2:], dtype=np.float64)
        pts = v[: (len(v) // 3) * 3].reshape(-1, 3)
        if room_mode.ACTIVE:
            pts = room_mode.v_pt(pts)
        got["pad"] = pts

    def send(self, side: str, q, grip: float) -> None:
        m = JointState()
        m.name = [f"{side}_fr3v2_joint{i + 1}" for i in range(JOINTS)]
        m.position = [float(v) for v in q]
        if side == SIDE and self.spine_cmd is not None:
            # held in every message: the bridge drops a group after a 1 s command gap
            m.name.append(SPINE_JOINT)
            m.position.append(float(self.spine_cmd))
        self.cmd[side].publish(m)
        g = JointState()
        g.name = [f"{side}_right_finger_joint"]
        g.position = [float(grip)]
        self.grip[side].publish(g)


def loose_target_px(raw_width: int | None) -> tuple[float, float] | None:
    """Centre of the evaluator's own 'target' bbox, in the frame the chain
    measures in (rotated under room mode)."""
    from ebim_task2.official_run import (
        parse_loose_label_map, rotate_bbox_for_room, select_target_bbox_px,
    )

    msg, labels = got.get("loose_bbox"), got.get("loose_labels")
    if msg is None or labels is None:
        return None
    box = select_target_bbox_px(
        msg, parse_loose_label_map(labels.data).get("target"))
    if box is None:
        return None
    y0, y1, x0, x1 = rotate_bbox_for_room(box, raw_width)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def rgb_decode(msg) -> np.ndarray | None:
    """RGB array from a raw sensor_msgs/Image, or None."""
    if msg is None:
        return None
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)[:, :, :3].copy()


def sem_decode(msg) -> np.ndarray | None:
    """Raw-id image from a semantic sensor_msgs/Image, or None."""
    if msg is None:
        return None
    dt = {"32SC1": np.int32, "32UC1": np.uint32, "16UC1": np.uint16,
          "16SC1": np.int16, "8UC1": np.uint8}.get(msg.encoding)
    if dt is None:
        return None
    a = np.frombuffer(msg.data, dtype=dt)
    if a.size != msg.height * msg.width:
        return None
    return a.reshape(msg.height, msg.width)


def pinch_pose(p, reach: float, yaw: float, phi: float = GRIP_PHI,
               pitch: float = 0.0) -> np.ndarray:
    """IK target (franka_fk's flange+TCP convention) that puts the grip
    point at world ``p`` with the tool z-axis at (``yaw``, ``pitch``) and
    the jaw-separation axis rolled ``phi`` about it (0 = separation along
    the tilted 'up'; pi = the official family, tool y down)."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    a = np.array([math.cos(yaw) * cp, math.sin(yaw) * cp, sp])
    h = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.cross(a, h)
    c, s = math.cos(phi), math.sin(phi)
    t = np.eye(4)
    t[:3, 0] = c * h + s * up
    t[:3, 1] = -s * h + c * up
    t[:3, 2] = a
    t[:3, 3] = np.asarray(p, dtype=np.float64) + a * (_TCP_OFFSET_Z - reach)
    return t


def log_so3(r):
    c = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    a = math.acos(c)
    if abs(a) < 1e-9:
        return np.zeros(3)
    return (a / (2 * math.sin(a))) * np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])


def exp_so3(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * kx + (1 - math.cos(th)) * (kx @ kx)


def pad_yaw(points) -> float:
    """Yaw of the strip's LONG axis, wrapped to (-pi/2, pi/2]. Returns 0.0
    for a cloud too small or too round to carry a direction."""
    p = np.asarray(points, dtype=np.float64)
    if p is None or len(p) < 40:
        return 0.0
    xy = p[:, :2] - p[:, :2].mean(axis=0)
    evals, evecs = np.linalg.eigh(xy.T @ xy)
    if evals[1] <= 0.0 or evals[0] / evals[1] > 0.25:
        return 0.0          # not elongated enough to trust a direction
    ax = evecs[:, 1]
    return _wrap_half(math.atan2(float(ax[1]), float(ax[0])))


def _wrap_half(a: float) -> float:
    """Wrap to (-pi/2, pi/2]. The strip is symmetric: +170 deg IS -10."""
    while a > math.pi / 2:
        a -= math.pi
    while a <= -math.pi / 2:
        a += math.pi
    return a


def strip() -> dict:
    p = got["pad"]
    return {"cx": float(p[:, 0].mean()), "cy": float(p[:, 1].mean()),
            "x_lo": float(np.percentile(p[:, 0], 1)),
            "x_hi": float(np.percentile(p[:, 0], 99)),
            "y_lo": float(np.percentile(p[:, 1], 1)),
            "y_hi": float(np.percentile(p[:, 1], 99)),
            "z_lo": float(np.percentile(p[:, 2], 1)),
            "z_hi": float(np.percentile(p[:, 2], 99))}


def _spine_settle(n: N, cmd: float, hold_q: dict,
                  grips: dict = None) -> tuple[float, bool]:
    """Hold ``cmd`` until the joint arrives or stops moving. Returns the
    live height and whether it moved at all under this command."""
    t0 = time.time()
    first = got.get("spine")
    last_move, last_val, reached = t0, (first or 0.0), 0
    n.spine_cmd = cmd
    while time.time() - t0 < SPINE_WAIT_S:
        for s_ in ("left", "right"):
            n.send(s_, hold_q[s_], (grips or {}).get(s_, OPEN_GRIP))
        rclpy.spin_once(n, timeout_sec=0.05)
        time.sleep(0.03)
        sp = got.get("spine")
        if sp is None:
            continue
        if abs(sp - last_val) > 0.001:
            last_move, last_val = time.time(), sp
        if abs(sp - cmd) <= SPINE_TOL:
            reached += 1
            if reached > 20:
                break
            continue
        reached = 0
        if time.time() - last_move > SPINE_STALL_S:
            break
    sp = float(got.get("spine", cmd))
    return sp, (first is None or abs(sp - first) > 0.002)


def raise_spine(n: N, target: float, hold_q: dict,
                grips: dict = None) -> float:
    """Drive the spine to ``target`` with the arms held, and report the
    height reached.

    The drive settles ~15 mm short of a position command, so each round
    re-commands the residual (bounded by SPINE_LIMITS and SPINE_SAG_TRIES).
    A spine that will not move costs the official geometry, not
    correctness: every solve localises against the live (q, ee_pose) pair.
    """
    lo, hi = SPINE_LIMITS
    target = max(lo, min(hi, float(target)))
    start = got.get("spine")
    if start is None:
        print(f"  spine: no /isaac/joint_states_full reading; commanding "
              f"{target:.4f} m blind", flush=True)
    elif abs(start - target) <= SPINE_TOL:
        n.spine_cmd = target
        print(f"  spine already at {start:.4f} m", flush=True)
        return float(start)
    else:
        print(f"  spine {start:.4f} -> {target:.4f} m (arm base "
              f"{ARM_MOUNT_Z + target - room_mode.LIFT_Z:+.4f} virtual z, "
              f"official {ARM_MOUNT_Z + SPINE_TARGET - room_mode.LIFT_Z:+.4f})",
              flush=True)
    t0 = time.time()
    cmd, sp = target, float(start or 0.0)
    for round_ in range(SPINE_SAG_TRIES + 1):
        sp, moved = _spine_settle(n, cmd, hold_q, grips)
        sag = target - sp
        if abs(sag) <= SPINE_TOL:
            break
        # a dropped command leaves the joint at its start height; a mere
        # steady-state offset is trimmed below
        if not moved and round_ == 0 and abs(sag) > 0.05:
            print(f"  !! spine did not move at all under a {cmd:.4f} "
                  f"command — the scene is dropping the joint (the bridge "
                  f"needs the spine in its right-arm group). Continuing "
                  f"from {sp:.4f} m; the solves follow the live base, but "
                  f"the geometry is NOT the official one", flush=True)
            break
        if round_ == SPINE_SAG_TRIES or not lo <= cmd + sag <= hi:
            print(f"  .. spine settled {sag * 1000:+.0f} mm off target and "
                  f"cannot be trimmed further; running at {sp:.4f} m",
                  flush=True)
            break
        cmd = max(lo, min(hi, cmd + sag))
        print(f"  .. spine sagged {sag * 1000:+.0f} mm under load; "
              f"re-commanding {cmd:.4f} m", flush=True)
    print(f"  spine at {sp:.4f} m after {time.time() - t0:.0f} s wall; arm "
          f"base {ARM_MOUNT_Z + sp - room_mode.LIFT_Z:+.4f} virtual z "
          f"(official {ARM_MOUNT_Z + SPINE_TARGET - room_mode.LIFT_Z:+.4f})",
          flush=True)
    # (q, ee) pairs sampled while the base was lifting are stale
    for s_ in ("left", "right"):
        got.pop(f"{s_}_samples", None)
    return sp


def fmt(s: dict) -> str:
    return (f"x {s['x_lo']:.4f}..{s['x_hi']:.4f} ({(s['x_hi'] - s['x_lo']) * 1000:5.1f}) "
            f"y {s['y_lo']:.4f}..{s['y_hi']:.4f} ({(s['y_hi'] - s['y_lo']) * 1000:4.1f}) "
            f"z {s['z_lo']:.4f}..{s['z_hi']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--stop-after", default="",
                    help="pinch | hang | carry | contact | sweep")
    ap.add_argument("--film", action="store_true",
                    help="save wrist/eval frames at the chain's key moments "
                         "to $EBIM_MIRROR_DEBUG_DIR/film/")
    ap.add_argument("--grasp-tries", type=int, default=3,
                    help="pinch attempts before the attempt is given up "
                         "(a missed pinch is retried in place)")
    ap.add_argument("--pinch-z", type=float, default=0.110,
                    help="tool-axis z of the pinch (official: wall top "
                         "+0.013; the tab bends ~7 mm up into the closing "
                         "jaw)")
    ap.add_argument("--spine", type=float, default=SPINE_TARGET,
                    help="vertical spine height (m) commanded before the "
                         "chain moves; the official demonstration ran the "
                         "whole episode at 0.4858, which puts the arm base "
                         "at the official +0.2349 above the board plane. "
                         "0 leaves the spine down")
    # ---- lay TIMING: the ep3 path at the official pace (pad unrolls
    # 85 -> 120 mm in ~0.8 s). The last sweep rungs under-track (mostly z),
    # so the final press is lighter than commanded.
    ap.add_argument("--press-settle", type=float, default=2.5,
                    help="dwell (s, x room pace) per PRESS rung")
    ap.add_argument("--press-iters", type=int, default=2,
                    help="closed-loop servo rounds per PRESS rung")
    ap.add_argument("--press-du-scale", type=float, default=1.0,
                    help="scale on the PRESS path's WESTWARD travel about "
                         "its first rung (1.0 = the ep3 path, 0.0 = press "
                         "straight down). Lower it to stop dragging the "
                         "target board out from under the pad")
    ap.add_argument("--sweep-settle", type=float, default=0.5,
                    help="dwell (s, x room pace) per SWEEP rung. The "
                         "official sweep runs -37 -> -63.7 in ~2-6 s TOTAL; "
                         "lower this to let the pad snap flat")
    ap.add_argument("--sweep-iters", type=int, default=1,
                    help="closed-loop servo rounds per SWEEP rung")
    ap.add_argument("--sweep-step", type=float, default=0.030,
                    help="streaming step (m) between SWEEP rungs")
    ap.add_argument("--sweep-wp-iters", type=int, default=1,
                    help="streaming sub-iterations per SWEEP sub-waypoint "
                         "(default: --wp-iters). These dominate the sweep's "
                         "duration, so this is the real speed knob")
    ap.add_argument("--yaw-aim-gain", type=float, default=0.0,
                    help="fraction of the measured hover heading error fed "
                         "back into the tool yaw. 0 = measure and log only "
                         "(no behaviour change)")
    ap.add_argument("--yaw-aim-cap", type=float, default=8.0,
                    help="cap on the heading correction (deg), per round")
    ap.add_argument("--yaw-aim-rounds", type=int, default=1,
                    help="measure/correct rounds at the hover. >1 re-issues "
                         "the hover pose at the corrected heading and "
                         "re-measures, so the loop converges without a "
                         "calibrated gain")
    ap.add_argument("--yaw-aim-tol", type=float, default=0.4,
                    help="stop correcting once the heading is this close "
                         "to the board axis (deg)")
    ap.add_argument("--dv-aim-cap", type=float, default=0.0,
                    help="cap on the hover re-aim's LATERAL shift (m). "
                         "Default 0 pins the lay on the tracked plate "
                         "centre")
    ap.add_argument("--lat-trim", type=float, default=None,
                    help="static lateral place trim (m, +v), added on top "
                         "of the clamped hover re-aim. Default: "
                         "SLOT_LAT_TRIM by resolved slot")
    ap.add_argument("--du-trim", type=float, default=None,
                    help="static along-axis place trim (m, +u), rides "
                         "du_off. Default: SLOT_DU_TRIM by resolved slot")
    ap.add_argument("--place-spine", type=float, default=None,
                    help="spine height (m) driven after the carry for the "
                         "place phase (grips held). Default: "
                         "SLOT_PLACE_SPINE by resolved slot")
    ap.add_argument("--flip-gate", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="refuse to pick when the pedestal crop reads no "
                         "liner (a flipped strip scores 0 and only a "
                         "scene-process reload heals it). The crop is placed "
                         "from the eval camera's own model, so a scene whose "
                         "camera differs from room_mode's constants reads a "
                         "false flip — turn it off only with the liner "
                         "confirmed another way (e.g. the semantic stream)")
    ap.add_argument("--park-spine", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="return the spine to the height the scene started "
                         "at once the arm is parked, so the scored frame "
                         "keeps the starting camera geometry")
    ap.add_argument("--contact-du", type=float, default=CONTACT_DU_CAL,
                    help="lay-axis offset from the target centre where the "
                         "hanging bottom edge first contacts (official "
                         "+0.053..0.057; the 120 mm strip then lays back "
                         "across the centre)")
    ap.add_argument("--flat-gate", type=float, default=0.75,
                    help="fraction of the pad cloud within 6 mm of the "
                         "board = the pad lies flat; the release gate")
    ap.add_argument("--open-arm-pitch", type=float, default=-60.0,
                    help="the flat-gate arms once the sweep pitch passes "
                         "this (deg); before it a flat reading is the "
                         "strip still standing on its edge")
    ap.add_argument("--target", default="auto",
                    help="'auto' locks the target plate via the evaluator's "
                         "loose stream and TRACKS it; or fixed "
                         "'x,y[,yaw_deg]'")
    ap.add_argument("--target-yaw-max", type=float, default=12.0,
                    help="exit 13 (fresh draw) if the plate yaw at chain "
                         "start exceeds this magnitude in degrees")
    ap.add_argument("--strip-len", type=float, default=0.120)
    ap.add_argument("--iters", type=int, default=18,
                    help="closed-loop servo rounds per stage")
    ap.add_argument("--max-corr", type=float, default=0.150,
                    help="cap on the closed-loop world-frame correction")
    ap.add_argument("--stream-step", type=float, default=0.010,
                    help="Cartesian spacing of streamed waypoints")
    ap.add_argument("--carry-step", type=float, default=0.005,
                    help="waypoint spacing for LOADED moves (the strip "
                         "hangs as a 116 mm pendulum off a ~4 mm bite)")
    ap.add_argument("--track-gain", type=float, default=0.35)
    ap.add_argument("--track-step", type=float, default=0.015)
    ap.add_argument("--wp-iters", type=int, default=3)
    ap.add_argument("--wp-tol", type=float, default=0.006)
    ap.add_argument("--ik-slack", type=float, default=0.060,
                    help="accept a solve this far short; the closed loop "
                         "corrects it. The real safety gate is --ik-dq")
    ap.add_argument("--ik-dq", type=float, default=0.5,
                    help="reject any solve moving a joint further than this "
                         "from the seed (branch-jump guard)")
    ap.add_argument("--corr-step", type=float, default=0.012)
    ap.add_argument("--base-x0", type=float, default=-0.02,
                    help="pick-station base x; <=-9 disables the park")
    ap.add_argument("--base-y0", type=float, default=0.0)
    ap.add_argument("--base-x", type=float, default=0.10,
                    help="place-station base x; <=-9 disables the ride")
    ap.add_argument("--base-y", type=float, default=0.0,
                    help="place-station base y (room: overridden to track "
                         "the target slot)")
    ap.add_argument("--base-tol", type=float, default=0.010)
    ap.add_argument("--fixed-base", action="store_true",
                    help="official fixed-base style: never drive the base — "
                         "all pedal output is suppressed and the arm works "
                         "from wherever the base stands (scored-spawn parity)")
    ap.add_argument("--settle", type=float, default=25.0,
                    help="WALL-second cap per closed-loop hold")
    args = ap.parse_args()
    auto_target = args.target.strip() == "auto"

    rclpy.init()
    n = N()
    for _ in range(250):
        rclpy.spin_once(n, timeout_sec=0.1)
        if all(f"{s}_q" in got for s in ("left", "right")) and "pad" in got \
                and (not auto_target or "eval_sem" in got):
            break
    if "pad" not in got or f"{SIDE}_q" not in got:
        print("no pad_points / joint states; is the scene up?")
        return 1

    # scene detection: room is served through a virtual-frame conjugation
    # (see room_mode); barebone continues in the same virtual geometry.
    if not room_mode.ACTIVE:
        for _ in range(40):     # the eval image is QoS-lossy at startup
            if "eval_msg" in got or os.environ.get("EBIM_SCENE"):
                break
            rclpy.spin_once(n, timeout_sec=0.1)
        eval_w = (got["eval_msg"].width if "eval_msg" in got
                  else got["eval_sem"].width if "eval_sem" in got else None)
        if room_mode.scene_of_width(eval_w) == "room":
            img_w = eval_w if eval_w else 1280
            room_mode.activate(img_w, sem_decode(got.get("eval_sem")))
            # purge state written before activation; callbacks re-fill it conjugated
            for k in ("pad", "odom", "left_ee", "right_ee",
                      "left_samples", "right_samples"):
                got.pop(k, None)
            # fresh-participant DDS discovery can take tens of seconds
            # per topic; allow a long fill window
            t_fill = time.time() + 90.0
            while time.time() < t_fill:
                rclpy.spin_once(n, timeout_sec=0.1)
                if "pad" in got and "odom" in got and f"{SIDE}_ee" in got:
                    break
            if "pad" not in got:
                print("no pad_points after room-mode purge; scene stalled?")
                return 1
            # stations pinned by the shoulder-vs-table-slab clearance
            if args.fixed_base:
                args.base_x0 = args.base_x = -99.0
                print("  room: FIXED-BASE mode — no stations, no pedal; "
                      "the arm works from the spawn", flush=True)
            else:
                args.base_x0 = room_mode.PARK_X
                args.base_x = room_mode.PARK_X
                print(f"  room stations: base_x0/base_x -> "
                      f"{room_mode.PARK_X}, base_y {args.base_y} (virtual)",
                      flush=True)
            if args.base_tol < room_mode.MIN_BASE_TOL:
                args.base_tol = room_mode.MIN_BASE_TOL
                print(f"  room base-tol -> {args.base_tol} (the base cannot "
                      f"resolve finer; relocalise absorbs it)", flush=True)
            # measure the sim factor that scales every wall-clock dwell
            s0 = got.get("odom_stamp")
            t_e = time.time() + 4.0
            while time.time() < t_e:
                rclpy.spin_once(n, timeout_sec=0.1)
            s1 = got.get("odom_stamp")
            if s0 and s1 and s1[0] > s0[0]:
                factor = (s1[1] - s0[1]) / (s1[0] - s0[0])
                room_mode.set_pace_from_factor(factor)
                print(f"  room sim factor {factor:.3f} -> dwell pace "
                      f"x{room_mode.pace():.2f}", flush=True)

    theta = 0.0
    plate_rid = None
    if auto_target:
        seg = room_mode.orient_seg(sem_decode(got.get("eval_sem")))
        raw_w = (got["eval_msg"].width if "eval_msg" in got
                 else got["eval_sem"].width if "eval_sem" in got else None)
        tgt_px = loose_target_px(raw_w)
        tx = ty = None
        if seg is not None and tgt_px is not None:
            win = ((tgt_px[0] - 60.0, tgt_px[0] + 60.0),
                   (tgt_px[1] - 60.0, tgt_px[1] + 60.0))
            plate_rid = find_plate_id(seg, window=win)
            print(f"  target from the evaluator's loose stream at px "
                  f"({tgt_px[0]:.0f},{tgt_px[1]:.0f})"
                  f"{'' if plate_rid is not None else ' — no semantic blob there'}",
                  flush=True)
        elif seg is not None:
            print("!! loose target stream unavailable; falling back to the "
                  "FIXED search window (assumes the stock target slot)",
                  flush=True)
            plate_rid = (find_plate_id(seg, window=room_mode.plate_window())
                         if room_mode.ACTIVE else find_plate_id(seg))
        if plate_rid is None:
            if tgt_px is not None:
                tx = (tgt_px[0] - _lf.PX0) / _lf.PXS
                ty = (_lf.PY0 - tgt_px[1]) / _lf.PYS
                print(f"  laying on the loose-stream target centre "
                      f"({tx:.4f},{ty:+.4f}) WITHOUT semantic tracking",
                      flush=True)
            else:
                tx, ty = room_mode.PLATE_SPAWN_X, room_mode.PLATE_SPAWN_Y
                print("!! plate not found in the eval semantic stream; "
                      "laying at the spawn pose WITHOUT tracking",
                      flush=True)
        else:
            tx, ty, theta, npx, _nc = measure_plate_id(seg, plate_rid)
            print(f"  plate LOCKED (id {plate_rid}, {npx}px): "
                  f"({tx:.4f},{ty:+.4f}) yaw {math.degrees(theta):+.2f} deg",
                  flush=True)
            if abs(math.degrees(theta)) > args.target_yaw_max:
                print(f"!! plate yaw {math.degrees(theta):+.1f} deg beyond "
                      f"the {args.target_yaw_max:.0f} deg guard; fresh draw")
                return 13
    else:
        parts = [float(v) for v in args.target.split(",")]
        tx, ty = parts[0], parts[1]
        theta = math.radians(parts[2]) if len(parts) > 2 else 0.0
    gp0 = (room_mode.gt_plate_virtual(got.get("obj_poses"))
           if auto_target else None)
    if gp0 is not None:
        d0 = math.hypot(gp0[0] - tx, gp0[1] - ty)
        print(f"  plate GT (room object_poses): ({gp0[0]:.4f},{gp0[1]:+.4f}) "
              f"yaw {math.degrees(gp0[2]):+.2f} deg — {d0 * 1000:.1f} mm / "
              f"{math.degrees(gp0[2] - theta):+.2f} deg from the locked "
              f"estimate; adopting it", flush=True)
        tx, ty, theta = gp0
        if abs(math.degrees(theta)) > max(args.target_yaw_max, 40.0):
            print(f"!! plate yaw {math.degrees(theta):+.1f} deg beyond the "
                  f"ground-truth guard; fresh draw")
            return 13
    frame = LayFrame(tx, ty, theta)
    if room_mode.ACTIVE and auto_target:
        # the four board slots differ only in virtual y; the place station
        # follows the target slot
        want_y = ty + room_mode.PLACE_STATION_DY
        if abs(want_y - args.base_y) > 0.005:
            print(f"  room: place station follows the target slot, "
                  f"base_y {args.base_y:+.3f} -> {want_y:+.3f} "
                  f"(target y {ty:+.4f})", flush=True)
            args.base_y = want_y
        # per-slot place trims (explicit flags win)
        slot_ = min((-0.10, 0.00, 0.10, 0.20), key=lambda s: abs(ty - s))
        if args.lat_trim is None:
            args.lat_trim = SLOT_LAT_TRIM.get(slot_)
        if args.du_trim is None:
            args.du_trim = SLOT_DU_TRIM.get(slot_)
        if args.place_spine is None:
            args.place_spine = SLOT_PLACE_SPINE.get(slot_)
        if args.lat_trim or args.du_trim or args.place_spine is not None:
            print(f"  slot {slot_:+.2f} place trims: lat "
                  f"{(args.lat_trim or 0.0) * 1000:+.1f} mm, du "
                  f"{(args.du_trim or 0.0) * 1000:+.1f} mm, spine "
                  f"{'-' if args.place_spine is None else args.place_spine}",
                  flush=True)

    # hold before sampling: bridge drops cached targets after 1 s
    hold_q = {s: np.asarray(got[f"{s}_q"], dtype=np.float64)
              for s in ("left", "right")}
    for s in ("left", "right"):
        got.pop(f"{s}_samples", None)
    t_end = time.time() + 4.0
    while time.time() < t_end:
        for s in ("left", "right"):
            n.send(s, hold_q[s], OPEN_GRIP)
        rclpy.spin_once(n, timeout_sec=0.05)
        time.sleep(0.03)

    # ---- SPINE: lift to the official demonstration height, before any
    # reach; every solve downstream localises against the moved arm base
    spine_start = float(got.get("spine", 0.0))
    if not args.dry:
        raise_spine(n, args.spine, hold_q)

    # arm base derived from the live (q, ee) pair — never fitted
    t_wb = w2b = np.eye(4)

    st0 = strip()
    print(f"strip at rest: {fmt(st0)}")
    # WEST tab: the drooping overhang past the wall's west face, the pinch target
    p = got["pad"]
    tab = p[p[:, 0] < WALL_W_X - 0.002]
    if len(tab) < 6 or st0["x_lo"] > WALL_W_X - 0.006:
        print(f"!! no west tab past the wall face x={WALL_W_X} "
              f"(cloud x_lo {st0['x_lo']:.4f}, {len(tab)} pts) — reset the "
              f"scene (park_arms.py --reset)")
        return 2
    tab_y = float(tab[:, 1].mean())
    if "eval_msg" in got and args.flip_gate:
        # liner flip gate: flips persist through resets; only a process restart
        # heals (exit 10)
        im_eval = room_mode.orient_rgb(
            rgb_decode(got["eval_msg"])).astype(np.int32)
        if room_mode.ACTIVE:
            r0, r1, c0, c1 = room_mode.liner_crop()
            crop = im_eval[r0:r1, c0:c1]
        else:
            crop = im_eval[469:546, 576:652]
        cyan = room_mode.liner_count(crop)
        if cyan < 200:
            print(f"!! strip FLIPPED on the pedestal (cyan {cyan}px < 200) "
                  f"— scene needs a PROCESS RESTART, not a reset", flush=True)
            return 10
        print(f"  liner-up confirmed on the pedestal (cyan {cyan}px)",
              flush=True)
    # roll the pinch with the strip so the jaws stay square to the tab
    syaw = pad_yaw(p)
    if abs(syaw) > math.radians(15.0):
        print(f"  strip yaw {math.degrees(syaw):+.1f} deg is past the "
              f"+-15 deg the pick is planned for; keeping the axis-aligned "
              f"plan", flush=True)
        syaw = 0.0
    print(f"  west tab: {len(tab)} pts, y centre {tab_y:+.4f}, pinch at "
          f"({WALL_W_X - 0.001:.4f},{tab_y:+.4f},{args.pinch_z:.4f}) "
          f"yaw {math.degrees(syaw):+.1f} deg", flush=True)

    seed = np.asarray(Q_SEED_OFF, dtype=np.float64)
    corr = np.zeros(3)
    cmd = seed.copy()
    grip_now = OPEN_GRIP
    cur_yaw = syaw          # tool heading; becomes the lay heading at place
    cur_pitch = PITCH_GRASP
    last_ike = [0.0]
    reloc: list = [None, 0]  # last accepted local base, consecutive refusals
    short_warned: set = set()

    def ik(pt, reach: float) -> tuple[np.ndarray, float]:
        nonlocal seed
        goal = w2b @ pinch_pose(pt, reach, cur_yaw, GRIP_PHI, cur_pitch)
        q = seed.copy()
        cur = franka_fk(q)
        dp = goal[:3, 3] - cur[:3, 3]
        dr = log_so3(goal[:3, :3] @ cur[:3, :3].T)
        steps = max(int(np.linalg.norm(dp) / 0.04),
                    int(np.linalg.norm(dr) / 0.30), 1)
        e = 9.9
        for k in range(1, steps + 1):
            t = np.eye(4)
            t[:3, 3] = cur[:3, 3] + dp * (k / steps)
            t[:3, :3] = exp_so3(dr * (k / steps)) @ cur[:3, :3]
            r = solve_ik(t, q, safe_joint_limits=SAFE, ori_tol=0.05)
            q = np.asarray(r.q, dtype=np.float64)
            e = 0.0 if r.success else float(r.pos_error)
        return q, e

    class Unreachable(Exception):
        pass

    def ik_guarded(name: str, target, reach: float, where: str) -> np.ndarray:
        """IK that tolerates the WORKSPACE EDGE but never a branch jump."""
        q, e = ik(target, reach)
        last_ike[0] = e
        if e <= 0.005:
            return q
        dq = float(np.max(np.abs(q - seed)))
        if e < args.ik_slack and dq < args.ik_dq:
            if name not in short_warned:
                short_warned.add(name)
                print(f"  .. {name}: IK short by {e * 1000:.0f} mm {where} "
                      f"(|dq|={dq:.2f} rad); taking the closest reachable pose",
                      flush=True)
            return q
        raise Unreachable(f"{name}: IK failed ({e * 1000:.0f} mm) {where} on "
                          f"{np.round(np.asarray(target), 4)} (|dq|={dq:.2f})")

    def pump(secs: float, tol: float = 0.012) -> float:
        """Hold ``cmd`` until convergence or steady state."""
        pace_ = room_mode.pace()
        t_end, err = time.time() + secs * pace_, 9.9
        stable = 0
        need = int(round(12 * pace_))
        hist: list[tuple[float, float]] = []
        while time.time() < t_end:
            n.send(SIDE, cmd, grip_now)
            n.send(OTHER, TRAVEL, OPEN_GRIP)
            rclpy.spin_once(n, timeout_sec=0.05)
            time.sleep(0.03)
            now = time.time()
            err = float(np.max(np.abs(
                np.asarray(got[f"{SIDE}_q"], dtype=np.float64) - cmd)))
            stable = stable + 1 if err < tol else 0
            if stable > need:
                break
            hist.append((now, err))
            while hist and hist[0][0] < now - 1.3 * pace_:
                hist.pop(0)
            if (len(hist) > 8 * pace_ and now - hist[0][0] > 1.1 * pace_
                    and max(e for _, e in hist) - min(e for _, e in hist) < 0.004):
                break
        return err

    def bail(code: int) -> int:
        nonlocal cmd, seed, cur_pitch
        dump_place_log()
        print("  parking before exit", flush=True)
        # lift and re-pitch first: a joint fold from a pitched-down low pose
        # dives the TCP under the table
        try:
            here = tip(GRIP_REACH)
            cur_pitch = math.radians(-30.0)
            q_up, e_up = ik((float(here[0]) - 0.02, float(here[1]),
                             max(float(here[2]) + 0.10, 0.30)), GRIP_REACH)
            if e_up < 0.02:
                a0 = np.asarray(cmd, dtype=np.float64)
                for k in range(1, 16):
                    step = a0 + (q_up - a0) * (k / 15.0)
                    t_end = time.time() + 0.30
                    while time.time() < t_end:
                        n.send(SIDE, step, OPEN_GRIP)
                        n.send(OTHER, TRAVEL, OPEN_GRIP)
                        rclpy.spin_once(n, timeout_sec=0.05)
                        time.sleep(0.03)
                cmd = q_up
        except (KeyError, ValueError):
            pass
        a = np.asarray(cmd, dtype=np.float64)
        b = np.asarray(TRAVEL, dtype=np.float64)
        # park via the seed family first: the direct TRAVEL fold sweeps
        # low across the board field
        qs_ = np.asarray(Q_SEED_OFF, dtype=np.float64)
        for k in range(1, 21):
            step = a + (qs_ - a) * (k / 20.0)
            t_end = time.time() + 0.40
            while time.time() < t_end:
                n.send(SIDE, step, OPEN_GRIP)
                n.send(OTHER, TRAVEL, OPEN_GRIP)
                rclpy.spin_once(n, timeout_sec=0.05)
                time.sleep(0.03)
        a = qs_
        for k in range(1, 41):
            step = a + (b - a) * (k / 40.0)
            t_end = time.time() + 0.50
            while time.time() < t_end:
                n.send(SIDE, step, OPEN_GRIP)
                n.send(OTHER, TRAVEL, OPEN_GRIP)
                rclpy.spin_once(n, timeout_sec=0.05)
                time.sleep(0.03)
        # a bail's frame can still be read by the self-verdict: park the
        # spine the same way the success path does
        if args.park_spine and abs(spine_start - args.spine) > SPINE_TOL:
            raise_spine(n, spine_start, {SIDE: b, OTHER: np.asarray(TRAVEL)})
        return code

    def tip(reach: float) -> np.ndarray:
        f = got[f"{SIDE}_ee"]
        return f[:3, 3] + f[:3, 2] * reach

    def relocalise() -> None:
        """Re-derive world<-armbase = ``T_ee @ franka_fk(q)^-1`` from the
        live pair (the base rolls up to 0.37 m under a reach)."""
        nonlocal t_wb, w2b
        s = got.get(f"{SIDE}_samples") or []
        if len(s) < 2:
            return
        q, f = s[-1]
        if float(np.max(np.abs(np.asarray(q) - np.asarray(s[-2][0])))) > 0.01:
            return
        t_new = np.asarray(f, dtype=np.float64) @ np.linalg.inv(
            franka_fk(np.asarray(q, dtype=np.float64), tool_offset=0.0))
        p = t_new[:3, 3]
        if reloc[0] is not None and float(np.linalg.norm(p - reloc[0])) > 0.25 \
                and reloc[1] < 3:
            reloc[1] += 1
            print(f"  [relocalise rejected: base jumped "
                  f"{float(np.linalg.norm(p - reloc[0])) * 1000:.0f} mm from "
                  f"the last accepted one ({reloc[1]}/3)]", flush=True)
            return
        reloc[1] = 0
        if reloc[0] is None:
            print(f"  [arm base {np.round(p, 4)}]", flush=True)
        reloc[0] = p.copy()
        t_wb, w2b = t_new, np.linalg.inv(t_new)

    # ---- mobile base via pedal tokens -----------------------------------
    def base_step(tok, wall_s: float, hold=None) -> None:
        """Publish a pedal token for wall_s, arms kept fed (1 s watchdog)."""
        if args.fixed_base:
            tok = None           # keep the arm-feeding dwell, never drive
        t_end, nxt = time.time() + wall_s, 0.0
        while time.time() < t_end:
            now = time.time()
            if tok is not None and now >= nxt:
                n.pedal.publish(String(data=tok))
                nxt = now + 0.10
            n.send(SIDE, cmd if hold is None else hold, grip_now)
            n.send(OTHER, TRAVEL, OPEN_GRIP)
            rclpy.spin_once(n, timeout_sec=0.05)
            time.sleep(0.03)
        if tok is not None:
            n.pedal.publish(String(data="STOP"))

    def base_wait_done(hold=None, max_wall_s: float = 10.0) -> None:
        """Ride out a pulse's transport delay, then wait until still."""
        pace_ = room_mode.pace()
        max_wall_s = max_wall_s * pace_
        t0 = time.time()
        prev_od = got["odom"]
        moved, still_since = False, None
        while time.time() - t0 < max_wall_s:
            base_step(None, 0.25, hold=hold)
            cur = got["odom"]
            d = (math.hypot(cur[0] - prev_od[0], cur[1] - prev_od[1])
                 + abs(cur[3] - prev_od[3]) * 0.3)
            prev_od = cur
            if d > 0.0015:
                moved, still_since = True, None
                continue
            if still_since is None:
                still_since = time.time()
            if time.time() - still_since > (1.0 if moved else 4.5) * pace_:
                return

    def base_err(bx: float, by: float):
        od = got["odom"]
        dxw, dyw = bx - od[0], by - od[1]
        c, s = math.cos(-od[3]), math.sin(-od[3])
        return (c * dxw - s * dyw, s * dxw + c * dyw, -od[3],
                math.hypot(dxw, dyw))

    def base_goto(bx: float, by: float, tol: float, hold=None,
                  tries: int = 12, cap_s: float = 0.30) -> bool:
        """Pulse-and-settle the castored base to world (bx, by, yaw 0)."""
        if args.fixed_base:
            return True          # fixed-base: stand wherever we are
        if got.get("odom") is None:
            print("  !! base_goto: no /isaac/odom; skipping", flush=True)
            return False
        settle_cap = 4.0 if (room_mode.ACTIVE and tol >= 0.05) else 10.0
        cap = cap_s
        prev_err = None
        swallowed = 0
        last_tok = None
        for _ in range(tries):
            dxb, dyb, dyaw, err = base_err(bx, by)
            if abs(dyaw) > 0.04:
                yaw_dur = min(max(abs(dyaw) / 0.35, 0.05), 0.25)
                if room_mode.ACTIVE:
                    yaw_dur = min(yaw_dur * room_mode.pace(), 0.55)
                base_step("A+C" if dyaw > 0 else "B+C", yaw_dur, hold=hold)
                base_wait_done(hold=hold, max_wall_s=settle_cap)
                continue
            if err <= tol:
                base_lock(hold=hold)
                od = got["odom"]
                print(f"  base parked at ({od[0]:+.4f},{od[1]:+.4f}) "
                      f"yaw={od[3]:+.3f}, err {err * 1000:.1f} mm (locked)",
                      flush=True)
                return True
            if prev_err is not None and err > prev_err + 0.008:
                cap = max(cap * 0.5, 0.05)
            prev_err = err
            tok = (("FWD" if dxb > 0 else "BACK") if abs(dxb) >= abs(dyb)
                   else ("A" if dyb > 0 else "B"))
            # LOADED: never flip direction to shave a small residual
            if (room_mode.ACTIVE and carrying[0] and last_tok is not None
                    and tok == BASE_REVERSE.get(last_tok)
                    and err <= room_mode.LOADED_REVERSAL_MAX_ERR):
                base_lock(hold=hold)
                od = got["odom"]
                print(f"  base stopped {err * 1000:.0f} mm out at "
                      f"({od[0]:+.4f},{od[1]:+.4f}): correcting it means "
                      f"reversing {last_tok}->{tok} with the strip hanging",
                      flush=True)
                return True
            last_tok = tok
            dur = min(max(err / 0.25, 0.05), cap)
            if swallowed >= 2:
                lunge = (min(0.22, cap_s + 0.04)
                         if room_mode.ACTIVE and carrying[0]
                         else min(0.28, cap_s + 0.10))
                dur = max(dur, lunge)
            od0 = got["odom"]
            base_step(tok, dur, hold=hold)
            base_wait_done(hold=hold, max_wall_s=settle_cap)
            od1 = got["odom"]
            moved = math.hypot(od1[0] - od0[0], od1[1] - od0[1])
            if moved < max(0.005, 0.25 * dur * 0.17):
                if (room_mode.ACTIVE and carrying[0]
                        and err <= room_mode.BASE_UNSTICK_MIN_ERR):
                    print(f"  [base_goto: {moved * 1000:.0f} mm on a "
                          f"{dur:.2f}s pulse at {err * 1000:.0f} mm out — "
                          f"too close to unstick, holding the stride]",
                          flush=True)
                else:
                    swallowed += 1
                    cap = min(max(cap * 1.6, 0.12), cap_s)
            else:
                swallowed = 0
            print(f"  [base_goto {tok} {dur:.2f}s: ({od0[0]:+.3f},{od0[1]:+.3f})"
                  f" -> ({od1[0]:+.3f},{od1[1]:+.3f}) yaw={od1[3]:+.3f}]",
                  flush=True)
        dxb, dyb, dyaw, err = base_err(bx, by)
        if err <= tol and abs(dyaw) <= 0.04:
            base_lock(hold=hold)
            od = got["odom"]
            print(f"  base parked at ({od[0]:+.4f},{od[1]:+.4f}) "
                  f"yaw={od[3]:+.3f}, err {err * 1000:.1f} mm (locked on the "
                  f"last pulse)", flush=True)
            return True
        od = got["odom"]
        print(f"  !! base_goto gave up at ({od[0]:+.4f},{od[1]:+.4f}) "
              f"yaw={od[3]:+.3f} (err "
              f"{math.hypot(bx - od[0], by - od[1]) * 1000:.0f} mm)",
              flush=True)
        return False

    def base_lock(hold=None) -> None:
        """Parking brake: a 0.05 s A+C squirt toes the swerve modules."""
        base_step("A+C", 0.05, hold=hold)

    def zero_base_yaw(hold=None, lim: float = 0.05, tries: int = 14) -> float:
        """Walk the base's yaw back to ~0 on its own, before any translation."""
        if args.fixed_base or got.get("odom") is None:
            return 0.0
        gain, prev_sign = BASE_YAW_RATE, 0
        for _ in range(tries):
            od_ = got["odom"]
            if abs(od_[3]) <= lim:
                break
            sign = 1 if od_[3] > 0 else -1
            if prev_sign and sign != prev_sign:
                gain *= 2.0          # overshot: ask for a shorter pulse
            prev_sign = sign
            base_step("A+C" if -od_[3] > 0 else "B+C",
                      min(max(abs(od_[3]) / gain, BASE_YAW_MIN_PULSE), 0.30),
                      hold=hold)
            base_wait_done(hold=hold, max_wall_s=5.0)
        return abs(got["odom"][3])

    prev = None
    place_log: list = []
    carrying = [False]        # True once the strip is confirmed hanging

    def harvest(name: str) -> None:
        q = got.get(f"{SIDE}_q")
        f = got.get(f"{SIDE}_ee")
        if q is not None and f is not None:
            place_log.append({"tag": name, "q": [round(float(v), 5) for v in q],
                              "p": [round(float(v), 5) for v in f[:3, 3]],
                              "R": [[round(float(v), 6) for v in row]
                                    for row in f[:3, :3]]})

    def dump_place_log() -> None:
        if place_log:
            try:
                json.dump({"side": SIDE, "samples":
                           [{"tag": s["tag"], "q": s["q"], "p": s["p"],
                             "R": s.get("R"), "q_err": 0.0} for s in place_log]},
                          open(os.path.join(DEBUG_DIR, "place_samples.json"), "w"))
                print(f"  [{len(place_log)} place samples dumped]", flush=True)
            except OSError as e:  # debug output must never kill the chain
                print(f"  [place-sample dump skipped: {e}]", flush=True)

    def bump(c_new: np.ndarray, cap: float) -> np.ndarray:
        c = np.asarray(c_new, dtype=np.float64)
        step = c - corr
        s = float(np.linalg.norm(step))
        if s > cap:
            c = corr + step * (cap / s)
        nrm = float(np.linalg.norm(c))
        return c * (args.max_corr / nrm) if nrm > args.max_corr else c

    def hang_state() -> tuple[float, float]:
        """(drop, span): free-end depth below the TCP and the cloud's
        vertical span — the hanging-strip health numbers."""
        st_ = strip()
        tz = float(tip(GRIP_REACH)[2])
        return tz - st_["z_lo"], st_["z_hi"] - st_["z_lo"]

    def flat_frac() -> float:
        """Fraction of the pad cloud within 6 mm of the board plane."""
        p_ = got["pad"]
        return float((p_[:, 2] < BOARD_Z + 0.006).mean())

    def loaded_ok(name: str, at) -> None:
        """Abort when the carried strip reads dropped (cloud fell away from
        the tool). Only checked while the strip should hang CLEAR — from
        the descent on (target z < 0.150) the strip is meant to touch the
        board and a low cloud is the goal, not a drop."""
        if not carrying[0]:
            return
        if at is not None and float(at[2]) < 0.150:
            return
        st_ = strip()
        if st_["z_hi"] < 0.050:
            print(f"  !! {name}: strip DROPPED mid-move (z_hi "
                  f"{st_['z_hi']:.4f})", flush=True)
            raise SystemExit(bail(7))

    def go(name: str, pt, reach: float, grip: float, stream: bool = True,
           settle: float | None = None, step: float | None = None,
           soft: bool = False, iters: int | None = None,
           wp_iters: int | None = None):
        nonlocal prev, cmd, seed, corr, grip_now
        try:
            return _go(name, pt, reach, grip, stream, settle, step, iters,
                       wp_iters)
        except Unreachable as exc:
            if not soft:
                print(f"  !! {exc}; aborting", flush=True)
                raise SystemExit(bail(5))
            print(f"  .. {exc}; stopping this stage here", flush=True)
            return None

    def _go(name: str, pt, reach: float, grip: float, stream: bool,
            settle: float | None, step: float | None,
            iters: int | None = None,
            wp_iters: int | None = None) -> np.ndarray:
        nonlocal prev, cmd, seed, corr, grip_now
        grip_now = grip
        pt = np.asarray(pt, dtype=np.float64)
        if stream and prev is not None:
            a, b = np.asarray(prev), pt
            sp = args.stream_step if step is None else step
            nstep = max(int(math.ceil(float(np.linalg.norm(b - a)) / sp)), 1)
            for k in range(1, nstep + 1):
                wp = a + (b - a) * (k / nstep)
                nsub = args.wp_iters if wp_iters is None else wp_iters
                for sub in range(max(nsub, 1)):
                    try:
                        cmd = ik_guarded(name, wp + corr, reach,
                                         f"at waypoint {k}/{nstep}")
                    except Unreachable:
                        if float(np.linalg.norm(corr)) < 1e-6:
                            raise
                        corr = corr * 0.6
                        continue
                    seed = cmd.copy()
                    t_end = time.time() + ((0.55 if sub == 0 else 0.35)
                                           * room_mode.pace())
                    while time.time() < t_end:
                        n.send(SIDE, cmd, grip_now)
                        n.send(OTHER, TRAVEL, OPEN_GRIP)
                        rclpy.spin_once(n, timeout_sec=0.05)
                        time.sleep(0.03)
                    relocalise()
                    dw = wp - tip(reach)
                    if float(np.linalg.norm(dw)) < args.wp_tol:
                        break
                    corr = bump(corr + args.track_gain * dw, args.track_step)
                loaded_ok(name, wp)
        qerr = 0.0
        gain, prev_miss = 1.0, None
        best = [9.9, 0]
        for _ in range(args.iters if iters is None else iters):
            try:
                cmd = ik_guarded(name, pt + corr, reach,
                                 "on the corrected target")
            except Unreachable:
                cur = float(np.linalg.norm(pt - tip(reach)))
                if cur < 0.015:
                    print(f"  .. {name}: corrected aim out of reach, but the "
                          f"tool is already {cur * 1000:.1f} mm off; keeping "
                          f"this pose", flush=True)
                    break
                if float(np.linalg.norm(corr)) < 1e-6:
                    raise
                corr = corr * 0.6
                print(f"  .. {name}: corrected aim out of reach; easing corr "
                      f"to ({corr[0] * 1000:+.0f},{corr[1] * 1000:+.0f},"
                      f"{corr[2] * 1000:+.0f})", flush=True)
                continue
            seed = cmd.copy()
            qerr = pump(settle if settle is not None else args.settle)
            relocalise()
            loaded_ok(name, pt)
            d = pt - tip(reach)
            miss = float(np.linalg.norm(d))
            if miss < 0.0015:
                break
            if miss < best[0] - 0.0005:
                best[0], best[1] = miss, 0
            else:
                best[1] += 1
                if best[1] >= 3:
                    print(f"  .. {name}: no longer improving at "
                          f"{miss * 1000:.1f} mm; moving on", flush=True)
                    break
            if prev_miss is not None and miss > 0.8 * prev_miss:
                gain = max(gain * 0.5, 0.15)
            prev_miss = miss
            corr = bump(corr + gain * d, args.corr_step)
        if name.startswith(("corr", "desc", "press", "sw", "ret")):
            harvest(name)
        d = pt - tip(reach)
        miss = float(np.linalg.norm(d)) * 1000
        dq = np.abs(np.asarray(got[f"{SIDE}_q"], dtype=np.float64) - cmd)
        jw = int(np.argmax(dq))
        print(f"  {name:9s} tcp=({pt[0]:.4f},{pt[1]:+.4f},{pt[2]:.4f}) "
              f"pitch={math.degrees(cur_pitch):+.1f} "
              f"miss={miss:5.1f}mm d=({d[0] * 1000:+.1f},{d[1] * 1000:+.1f},"
              f"{d[2] * 1000:+.1f}) q_err={qerr:.3f}(j{jw + 1}) "
              f"ik={last_ike[0] * 1000:.0f}mm "
              f"corr=({corr[0] * 1000:+.0f},{corr[1] * 1000:+.0f},"
              f"{corr[2] * 1000:+.0f}) base=({t_wb[0, 3]:.3f},{t_wb[1, 3]:.3f},"
              f"{t_wb[2, 3]:.3f}) | {fmt(strip())}", flush=True)
        od, sp = got.get("odom"), got.get("spine")
        if od is not None:
            print(f"            base_link=({od[0]:+.4f},{od[1]:+.4f},"
                  f"{od[2]:+.4f}) yaw={od[3]:+.5f}  spine="
                  f"{sp if sp is None else round(sp, 5)}", flush=True)
        prev = pt
        return d * 1000.0

    def hold_tick(ref, state) -> None:
        """One 10 Hz tick of continuous base hold during a glide."""
        if args.fixed_base:
            return
        if ref is None or got.get("odom") is None:
            return
        now = time.time()
        if now < state.get("next", 0.0):
            return
        state["next"] = now + 0.10
        od = got["odom"]
        dyaw = -od[3]
        if abs(dyaw) > 0.08:
            n.pedal.publish(String(data="A+C" if dyaw > 0 else "B+C"))
            return
        dxw, dyw = ref[0] - od[0], ref[1] - od[1]
        c, sn = math.cos(-od[3]), math.sin(-od[3])
        dxb, dyb = c * dxw - sn * dyw, sn * dxw + c * dyw
        on = state.get("on", False)
        lim = 0.010 if on else 0.015     # hysteresis
        if max(abs(dxb), abs(dyb)) < lim:
            if on:
                n.pedal.publish(String(data="STOP"))
            state["on"] = False
            return
        state["on"] = True
        tok = (("FWD" if dxb > 0 else "BACK") if abs(dxb) >= abs(dyb)
               else ("A" if dyb > 0 else "B"))
        n.pedal.publish(String(data=tok))

    def station_keep(pt, reach: float, state: dict,
                     gate: float = 0.012) -> None:
        """Hold a world point through base drift during a long dwell. Never
        chase the tool (the corr chase feeds the base slide); never drive
        the base (that IS the drag)."""
        nonlocal cmd, seed
        if not room_mode.ACTIVE:
            return
        now = time.time()
        if now < state.get("next", 0.0):
            return
        state["next"] = now + 0.6
        od = got.get("odom")
        if od is None:
            return
        ref = state.get("od0")
        if ref is None:
            state["od0"] = (od[0], od[1], od[3])
            return
        if (math.hypot(od[0] - ref[0], od[1] - ref[1]) < 0.004
                and abs(math.atan2(math.sin(od[3] - ref[2]),
                                   math.cos(od[3] - ref[2]))) < 0.008):
            return
        before = tip(reach)
        off = float(np.linalg.norm(np.asarray(pt, dtype=np.float64) - before))
        if off > gate:
            if not state.get("warned"):
                state["warned"] = True
                print(f"  .. station-keep stood down: the tool is "
                      f"{off * 1000:.0f} mm off the held point, which is "
                      f"not the drift it corrects", flush=True)
            return
        t_before = t_wb.copy()
        relocalise()
        if np.array_equal(t_before, t_wb):
            return
        try:
            qn = ik_guarded("keep", np.asarray(pt, dtype=np.float64) + corr,
                            reach, "station-keeping the pose")
        except Unreachable:
            return
        if float(np.max(np.abs(qn - cmd))) > 0.10:
            return
        cmd = qn
        seed = cmd.copy()
        state["od0"] = (od[0], od[1], od[3])
        state["n"] = state.get("n", 0) + 1
        state["mm"] = state.get("mm", 0.0) + float(
            np.linalg.norm(np.asarray(pt, dtype=np.float64) - before)) * 1000.0

    def keep_report(tag: str, state: dict) -> None:
        if state.get("n"):
            print(f"  room: station-kept the {tag} through "
                  f"{state['n']} base-drift corrections "
                  f"({state.get('mm', 0.0) / state['n']:.1f} mm mean)",
                  flush=True)

    def glide(name: str, pt, reach: float, steps: int = 30,
              dwell: float = 0.25, hold_ref=None) -> bool:
        dwell = dwell * room_mode.pace()
        """Joint-space glide: solve once, stream the joint interp (Cartesian
        streaming rocks the base; glides do not)."""
        nonlocal cmd, seed, corr
        try:
            qb = ik_guarded(name, np.asarray(pt, dtype=np.float64) + corr,
                            reach, "for the glide")
        except Unreachable as exc:
            print(f"  .. {name}: glide target unsolvable ({exc}); "
                  f"falling back", flush=True)
            return False
        qa = np.asarray(cmd, dtype=np.float64)
        hstate: dict = {}
        t_wb_g = t_wb.copy()
        od0 = got.get("odom")
        left = steps
        while left > 0:
            od = got.get("odom")
            if od is not None and od0 is not None:
                ddx, ddy = od[0] - od0[0], od[1] - od0[1]
                if math.hypot(ddx, ddy) > 0.004:
                    t_wb_g[0, 3] += ddx
                    t_wb_g[1, 3] += ddy
                    od0 = od
                    goal = np.linalg.inv(t_wb_g) @ pinch_pose(
                        np.asarray(pt, dtype=np.float64) + corr, reach,
                        cur_yaw, GRIP_PHI, cur_pitch)
                    r = solve_ik(goal, cmd, safe_joint_limits=SAFE,
                                 ori_tol=0.05)
                    qn = np.asarray(r.q, dtype=np.float64)
                    dq_lim = 0.15 if room_mode.ACTIVE else 0.5
                    if (r.success or float(r.pos_error) < 0.010) and \
                            float(np.max(np.abs(qn - qb))) < dq_lim:
                        qb = qn
                        qa = np.asarray(cmd, dtype=np.float64)
            cmd = qa + (qb - qa) * (1.0 / left) if left > 1 else qb.copy()
            qa = cmd.copy()
            left -= 1
            t_end = time.time() + dwell
            while time.time() < t_end:
                n.send(SIDE, cmd, grip_now)
                n.send(OTHER, TRAVEL, OPEN_GRIP)
                hold_tick(hold_ref, hstate)
                rclpy.spin_once(n, timeout_sec=0.05)
                time.sleep(0.03)
        if hold_ref is not None and not args.fixed_base:
            n.pedal.publish(String(data="STOP"))
        seed = cmd.copy()
        relocalise()
        loaded_ok(name, pt)
        return True

    def glide_converge(name: str, pt, reach: float, rounds: int = 3,
                       tol: float = 0.008, hold_ref=None,
                       steps: int | None = None, dwell: float = 0.22) -> bool:
        """Re-solve-and-glide chase of a world target with corr FROZEN."""
        nonlocal corr, prev
        corr = np.zeros(3)
        pt = np.asarray(pt, dtype=np.float64)
        ok = False
        for r in range(rounds):
            relocalise()
            d = pt - tip(reach)
            miss = float(np.linalg.norm(d))
            if miss < tol:
                ok = True
                break
            nsteps = steps if steps is not None \
                else max(6, min(30, int(miss / 0.01)))
            if not glide(f"{name}-g{r}", pt, reach, steps=nsteps, dwell=dwell,
                         hold_ref=hold_ref):
                break
            ok = True
            base_step(None, 1.5)   # strain-free coast; the base stops
        relocalise()
        d = pt - tip(reach)
        od = got.get("odom") or (9.9, 9.9, 9.9, 9.9)
        print(f"  {name:9s} glide-converged miss="
              f"{float(np.linalg.norm(d)) * 1000:5.1f}mm "
              f"d=({d[0] * 1000:+.1f},{d[1] * 1000:+.1f},{d[2] * 1000:+.1f}) "
              f"base_link=({od[0]:+.4f},{od[1]:+.4f}) yaw={od[3]:+.3f} | "
              f"{fmt(strip())}", flush=True)
        runaway = 0.060 if room_mode.ACTIVE else 0.120
        if ok and float(np.linalg.norm(d)) > runaway:
            print(f"  .. {name}: still {float(np.linalg.norm(d)) * 1000:.0f}"
                  f" mm off after the glides (runaway guard "
                  f"{runaway * 1000:.0f} mm); stopping here", flush=True)
            ok = False
        if ok:
            prev = pt
            if name.startswith(("corr", "desc", "press", "sw", "ret")):
                harvest(name)
        return ok

    def save_frame(tag: str) -> None:
        if not args.film:
            return
        try:
            film_dir = os.path.join(DEBUG_DIR, "film")
            os.makedirs(film_dir, exist_ok=True)
            for k in ("wrist", "eval"):
                im = rgb_decode(got.get(f"{k}_msg"))
                if im is not None:
                    np.save(f"{film_dir}/{tag}_{k}.npy", im)
        except OSError as e:  # debug output must never kill the chain
            print(f"  [film skipped: {e}]", flush=True)
        st_ = strip()
        print(f"  [film] {tag}: strip {fmt(st_)}", flush=True)

    def refresh_frame(where: str) -> None:
        """Re-measure the plate and re-aim every uncommitted step; covered
        views freeze the last good frame, part-covered update the LINE only."""
        nonlocal frame, theta, cur_yaw

        def adopt(nx_, ny_, mth_, note: str) -> None:
            nonlocal frame, theta, cur_yaw
            moved_ = math.hypot(nx_ - frame.tx, ny_ - frame.ty)
            dth_ = mth_ - frame.theta
            if moved_ < 0.0015 and abs(dth_) < math.radians(0.8):
                return
            frame = LayFrame(nx_, ny_, mth_)
            theta = mth_
            cur_yaw = mth_        # the lay heading follows the plate yaw
            print(f"  [{where}] plate moved {moved_ * 1000:.1f} mm / dyaw "
                  f"{math.degrees(dth_):+.1f} deg -> re-aim ({nx_:.4f},"
                  f"{ny_:+.4f}) yaw {math.degrees(mth_):+.1f} ({note})",
                  flush=True)

        if room_mode.ACTIVE:
            raw_w_ = (got["eval_msg"].width if "eval_msg" in got
                      else got["eval_sem"].width if "eval_sem" in got
                      else None)
            tp_ = loose_target_px(raw_w_)
            if tp_ is not None:
                adopt((tp_[0] - _lf.PX0) / _lf.PXS,
                      (_lf.PY0 - tp_[1]) / _lf.PYS,
                      frame.theta, "live loose bbox")
                return
        gp_ = room_mode.gt_plate_virtual(got.get("obj_poses"))
        if gp_ is not None:
            cap_ = math.radians(3.0)
            th_ = frame.theta + max(-cap_, min(cap_, gp_[2] - frame.theta))
            adopt(gp_[0], gp_[1], th_,
                  "gt" if abs(th_ - gp_[2]) < 1e-9
                  else f"gt yaw-limited from {math.degrees(gp_[2]):+.1f}")
            return
        if plate_rid is None:
            return
        seg_ = room_mode.orient_seg(sem_decode(got.get("eval_sem")))
        if seg_ is None:
            return
        m_ = measure_plate_id(seg_, plate_rid)
        if m_ is None:
            return
        mx_, my_, mth_, npx_, ncols_ = m_
        if npx_ < PLATE_FULL_PX:
            if ncols_ < 45:
                return
            dth_cap = math.radians(1.5)
            mth_ = frame.theta + max(-dth_cap,
                                     min(dth_cap, mth_ - frame.theta))
            nx_, ny_ = project_center_to_line(frame.tx, frame.ty,
                                              mx_, my_, mth_)
        else:
            nx_, ny_ = mx_, my_
        adopt(nx_, ny_, mth_, f"{npx_}px/{ncols_}c")

    def strip_settle(cap_s: float = 15.0) -> None:
        """Wait until the hanging strip stops swinging."""
        pace_ = room_mode.pace()
        t_cap = time.time() + cap_s * pace_
        need = int(round(12 * pace_))
        stable = 0
        last = None
        while time.time() < t_cap:
            n.send(SIDE, cmd, grip_now)
            n.send(OTHER, TRAVEL, OPEN_GRIP)
            rclpy.spin_once(n, timeout_sec=0.05)
            time.sleep(0.03)
            st_ = strip()
            ext = max(st_["x_hi"] - st_["x_lo"], st_["y_hi"] - st_["y_lo"])
            if last is not None and ext < 0.045 \
                    and abs(ext - last) < 0.004:
                stable += 1
                if stable >= need:
                    return
            else:
                stable = 0
            last = ext

    # ---- plan dump -------------------------------------------------------
    pinch_pt = (WALL_W_X - 0.001, tab_y, args.pinch_z)
    if args.dry:
        print(f"  pinch at {tuple(round(v, 4) for v in pinch_pt)} pitch "
              f"{math.degrees(PITCH_GRASP):+.0f} yaw "
              f"{math.degrees(syaw):+.1f}; peel arc {len(PEEL_ARC)} pts; "
              f"carry pitch {math.degrees(PITCH_CARRY):+.1f}")
        print(f"  target ({tx:.4f},{ty:+.4f}) yaw "
              f"{math.degrees(theta):+.1f}; contact edge u="
              f"{args.contact_du:+.4f}; sweep {len(SWEEP)} rungs to "
              f"{SWEEP[-1][0]:.1f} deg; open gate z_hi<{args.flat_gate}")
        return 0

    # park the base canonically FIRST (fixed-base: stand at the spawn)
    if room_mode.ACTIVE and not args.fixed_base \
            and got.get("odom") is not None:
        od0 = got["odom"]
        for lx, ly in room_mode.approach_legs(od0[1]):
            base_goto(lx, ly, 0.06, hold=TRAVEL, tries=25)
    if args.base_x0 > -9 and not args.fixed_base:
        base_goto(args.base_x0, args.base_y0, args.base_tol, hold=TRAVEL,
                  tries=25 if room_mode.ACTIVE else 12)
    if args.fixed_base:
        # the park dwells normally give the (q, ee) sampler its settle
        # window; swing to TRAVEL and stand until it settles (post-reset
        # the arm starts at the spawn config, so the swing itself takes
        # several paced seconds before samples can settle)
        base_step(None, 6.0 * room_mode.pace(), hold=TRAVEL)

    # localise before the first solve (t_wb still identity)
    relocalise()
    for _try in range(3):
        if reloc[0] is not None:
            break
        print("  .. no settled (q, ee) pair yet; standing still to settle",
              flush=True)
        base_step(None, 5.0 * room_mode.pace(), hold=TRAVEL)
        relocalise()
    if reloc[0] is None:
        print("no settled (q, ee) pair to localise the arm base from")
        return 1

    def joint_sweep(qb, steps: int, dwell: float = 0.35) -> None:
        nonlocal cmd
        qa_ = np.asarray(got[f"{SIDE}_q"], dtype=np.float64)
        qb = np.asarray(qb, dtype=np.float64)
        for k_ in range(1, steps + 1):
            cmd = qa_ + (qb - qa_) * (k_ / float(steps))
            t_e = time.time() + dwell
            while time.time() < t_e:
                n.send(SIDE, cmd, grip_now)
                n.send(OTHER, TRAVEL, OPEN_GRIP)
                rclpy.spin_once(n, timeout_sec=0.05)
                time.sleep(0.03)
        # no castor-brake squirts here (each A+C squirt injects yaw); the
        # reach probe absorbs the sweep's base drift

    def reconf_sweep() -> None:
        """TWO-STAGE quasi-static reconfiguration into the official family:
        wrist joints first with the arm still folded (their ~2 rad swings
        barely load the castors), then shoulder/elbow."""
        q_mid = list(TRAVEL[:4]) + list(Q_SEED_OFF[4:])
        joint_sweep(q_mid, RECONF_STEPS // 2)
        joint_sweep(Q_SEED_OFF, RECONF_STEPS // 2)
        err_ = float(np.max(np.abs(
            np.asarray(got[f"{SIDE}_q"], dtype=np.float64) - cmd)))
        od_ = got.get("odom")
        print(f"  family reconfiguration done (err {err_:.3f} rad"
              + (f", base yaw {od_[3]:+.3f})" if od_ else ")"), flush=True)

    def pick_reachable():
        """Probe (never send) the pick block from the LIVE arm base with
        the same sub-stepped continuation the live ik() uses: pre1 entry,
        pinch, east hang, -37 carry. Returns the ENTRY solution q_pre1 on
        success (None on failure): the chain sweeps to it and seeds from
        it, so the live solves stay on the probe's branch."""
        nonlocal cur_yaw, cur_pitch, seed
        saved = (cur_yaw, cur_pitch, seed.copy())
        probes = ((0.55, -0.299, args.pinch_z, PITCH_GRASP),
                  (WALL_W_X - 0.001, -0.299, args.pinch_z, PITCH_GRASP),
                  (0.872, -0.299, 0.188, PITCH_GRASP),
                  (CARRY_X, -0.200, CARRY_Z, PITCH_CARRY))
        q_entry = None
        try:
            cur_yaw = syaw
            for i, (px, py, pz, pp) in enumerate(probes):
                cur_pitch = pp
                q, e = ik((px, py, pz), GRIP_REACH)
                if e > 0.008:
                    print(f"  .. reach probe ({px:.3f},{py:+.3f},{pz:.3f}) "
                          f"pitch {math.degrees(pp):+.0f}: "
                          f"{e * 1000:.0f} mm short from the live base",
                          flush=True)
                    return None
                if i == 0:
                    q_entry = q.copy()
                seed = q      # continue the family into the next probe
        finally:
            cur_yaw, cur_pitch, seed = saved[0], saved[1], saved[2]
        return q_entry

    reconf_sweep()
    if room_mode.ACTIVE:
        # a re-park is only forced when the pick block does not SOLVE from
        # here; forced corrections happen FOLDED at TRAVEL (a stretched family
        # hold sweeps the gripper past the table edge while the base wanders)
        for cycle in range(3):
            reloc[0] = None
            relocalise()
            _, _, _, err_now = base_err(args.base_x0, args.base_y0)
            od_ = got.get("odom")
            yaw_now = abs(od_[3]) if od_ is not None else 0.0
            # no yaw precondition: relocalise feeds the yawed base into every
            # solve, so the probe alone decides
            q_entry = pick_reachable()
            if q_entry is not None:
                print(f"  room: pick block solves from {err_now * 1000:.0f}"
                      f" mm / {math.degrees(yaw_now):.1f} deg off station; "
                      f"proceeding without a re-park", flush=True)
                # enter on the probe's own branch: sweep to its pre1
                # solution and seed the chain from it
                joint_sweep(q_entry, max(
                    20, int(float(np.max(np.abs(
                        q_entry - np.asarray(got[f"{SIDE}_q"],
                                             dtype=np.float64)))) / 0.03)))
                seed = np.asarray(q_entry, dtype=np.float64)
                break
            if cycle == 2:
                print(f"  !! pick block unreachable after the recovery "
                      f"cycles ({err_now * 1000:.0f} mm off); fresh draw",
                      flush=True)
                return bail(13)
            print(f"  room: pick unreachable from {err_now * 1000:.0f} mm /"
                  f" {math.degrees(yaw_now):.1f} deg off station; folding "
                  f"to TRAVEL to drive home light", flush=True)
            joint_sweep(TRAVEL, 30)
            if got.get("odom") is not None and abs(got["odom"][3]) > 0.04:
                zero_base_yaw(hold=TRAVEL, lim=BASE_YAW_FLOOR)
            base_goto(args.base_x0, args.base_y0, args.base_tol,
                      hold=TRAVEL, tries=25, cap_s=0.45)
            reconf_sweep()
        reloc[0] = None
        relocalise()

    # ---- PICK: official side pinch of the west tab -----------------------
    hang = args.strip_len - 0.004      # refined after the peel
    got_it = False
    for attempt in range(1, max(args.grasp_tries, 1) + 1):
        if attempt > 1:
            print(f"  pinch retry {attempt}/{args.grasp_tries}", flush=True)
            strip_settle()
            st_r = strip()
            p_r = got["pad"]
            tab_r = p_r[p_r[:, 0] < WALL_W_X - 0.002]
            if len(tab_r) < 6:
                print("  !! west tab gone after the failed pinch; fresh draw",
                      flush=True)
                return bail(13)
            pinch_pt = (WALL_W_X - 0.001, float(tab_r[:, 1].mean()),
                        args.pinch_z)
        cur_pitch = PITCH_GRASP
        go("pre1", (0.55, pinch_pt[1], args.pinch_z), GRIP_REACH, OPEN_GRIP)
        go("pre2", (0.66, pinch_pt[1], args.pinch_z), GRIP_REACH, OPEN_GRIP,
           settle=8.0)
        go("approach", (WALL_W_X - 0.021, pinch_pt[1], args.pinch_z),
           GRIP_REACH, OPEN_GRIP, step=0.006, settle=8.0)
        go("pinch", pinch_pt, GRIP_REACH, OPEN_GRIP, step=0.004, settle=10.0,
           iters=6)
        # full close where we stand (official g -> 0 in one step)
        grip_now = CLOSED_GRIP
        pump(6.0)
        save_frame(f"close{attempt}")
        if args.stop_after == "pinch":
            return bail(0)
        # peel: 19 mm west at constant z — an attached tab drags the cloud
        st_before = strip()
        go("peel_w", (pinch_pt[0] + PEEL_WEST, pinch_pt[1],
                      args.pinch_z + 0.001), GRIP_REACH, CLOSED_GRIP,
           step=0.004, settle=6.0, iters=4)
        st_after = strip()
        if st_after["x_lo"] > st_before["x_lo"] - 0.006 \
                and st_after["z_hi"] < st_before["z_hi"] + 0.004:
            print(f"  .. pinch {attempt} missed (cloud did not follow the "
                  f"west pull: x_lo {st_before['x_lo']:.4f} -> "
                  f"{st_after['x_lo']:.4f}); reopening", flush=True)
            grip_now = OPEN_GRIP
            pump(3.0)
            go("back-out", (0.66, pinch_pt[1], args.pinch_z), GRIP_REACH,
               OPEN_GRIP, step=0.006, settle=4.0, iters=3)
            continue
        # peel arc: the strip slides off the stand top and swings to a hang
        for i, (du, dz) in enumerate(PEEL_ARC):
            go(f"peel{i}", (pinch_pt[0] + du, pinch_pt[1],
                            args.pinch_z + dz), GRIP_REACH, CLOSED_GRIP,
               step=args.carry_step, settle=3.0, iters=2)
        strip_settle()
        drop, span = hang_state()
        if span >= 0.090 and drop <= args.strip_len + 0.035:
            hang = drop
            carrying[0] = True
            got_it = True
            print(f"  strip HANGING: {span * 1000:.0f} mm span, free end "
                  f"{drop * 1000:.0f} mm below the grip point", flush=True)
            save_frame("hang")
            break
        print(f"  .. pinch {attempt}: bad hang (span {span * 1000:.0f} mm, "
              f"drop {drop * 1000:.0f} mm)", flush=True)
        if span < 0.030 and strip()["z_hi"] < 0.115:
            # the strip fell back onto the stand: reopen and retry
            grip_now = OPEN_GRIP
            pump(3.0)
            go("back-out", (0.60, pinch_pt[1], args.pinch_z + 0.04),
               GRIP_REACH, OPEN_GRIP, step=0.008, settle=4.0, iters=3)
            continue
        return bail(7)
    if not got_it:
        print(f"  !! no pinch after {args.grasp_tries} tries; fresh draw",
              flush=True)
        return bail(13)
    if args.stop_after == "hang":
        return bail(0)

    # ---- pitch ramp to the carry attitude --------------------------------
    ramp_from = math.degrees(PITCH_GRASP)
    ramp_to = math.degrees(PITCH_CARRY)
    hang_pt = np.asarray(tip(GRIP_REACH), dtype=np.float64)
    prev = hang_pt.copy()
    rungs = 8
    for i in range(1, rungs + 1):
        cur_pitch = math.radians(ramp_from + (ramp_to - ramp_from) * i / rungs)
        wy = pinch_pt[1] + 0.012 * i       # drift north with ep3's ramp
        go(f"ramp{i}", (CARRY_X, wy, CARRY_Z + (0.188 - CARRY_Z)
                        * (rungs - i) / rungs), GRIP_REACH, CLOSED_GRIP,
           step=args.carry_step, settle=2.5, iters=2)
    strip_settle()
    loaded_ok("carry", None)
    if args.stop_after == "carry":
        return bail(0)

    # ---- RIDE the base to the place station (fixed-base: skip) -----------
    if args.base_x > -9 and not args.fixed_base:
        print(f"  advancing the base to ({args.base_x:+.3f},"
              f"{args.base_y:+.3f}) for the place reach", flush=True)
        legs = [(args.base_x0 if args.base_x0 > -9 else 0.0,
                 args.base_y0 + (args.base_y - args.base_y0) * f)
                for f in (0.4, 0.8)] \
            + [(args.base_x, args.base_y)]
        for lx, ly in legs:
            base_goto(lx, ly, max(args.base_tol, 0.025), cap_s=0.18,
                      tries=25 if room_mode.ACTIVE else 12)
            if strip()["z_hi"] < 0.050:
                print("  !! strip DROPPED mid-leg during the base ride",
                      flush=True)
                return bail(7)
        if room_mode.ACTIVE:
            yl_ = zero_base_yaw(hold=cmd, lim=BASE_YAW_FLOOR)
            od_ = got.get("odom")
            print(f"  room: place station at "
                  f"({od_[0]:+.4f},{od_[1]:+.4f}) yaw "
                  f"{math.degrees(od_[3]):+.2f} deg" if od_ else
                  f"  room: place-station yaw {math.degrees(yl_):+.2f} deg",
                  flush=True)
            if yl_ > BASE_YAW_FLOOR:
                base_goto(args.base_x, args.base_y,
                          max(args.base_tol, 0.025), cap_s=0.18, tries=20)
            strip_settle()
            if strip()["z_hi"] < 0.050:
                print("  !! strip DROPPED during the base ride", flush=True)
                return bail(7)
        base_goto(args.base_x, args.base_y, max(args.base_tol, 0.020),
                  cap_s=0.18, tries=25 if room_mode.ACTIVE else 12)
        base_step(None, 4.0)     # let the strip stop swinging again
        strip_settle()
        # intentional move => unguarded relocalise
        reloc[0] = None
        relocalise()
        prev = np.asarray(tip(GRIP_REACH), dtype=np.float64)
        print(f"  streaming origin reset to the ridden tool position "
              f"({prev[0]:.3f},{prev[1]:+.3f},{prev[2]:.3f})", flush=True)
        refresh_frame("pre-place")
        drop, span = hang_state()
        if span < 0.090 or drop < hang - 0.035:
            print(f"  !! strip FOLDED during the base ride (span "
                  f"{span * 1000:.0f} mm, free end {drop * 1000:.0f} mm "
                  f"below the grip, hang was {hang * 1000:.0f}); aborting",
                  flush=True)
            return bail(7)
        if drop > hang + 0.035:
            print(f"  !! strip DROPPED during the base ride (free end "
                  f"{drop * 1000:.0f} mm below the grip, hang only "
                  f"{hang * 1000:.0f}); aborting", flush=True)
            return bail(7)

    # ---- PLACE: descend, contact, press, pitch sweep, open ---------------
    def u_pt(u: float, v: float, z: float):
        w = frame.to_world(u, v)
        return (float(w[0]), float(w[1]), float(z))

    # corridor entry + hover over the contact latitude, still at -37;
    # the tool heading follows the plate yaw from here (the lay axis)
    cur_yaw = frame.theta
    cur_pitch = PITCH_CARRY
    # --contact-du translates the whole lay along the board axis: descent,
    # press and sweep ride the same offset, so the path keeps its shape.
    # --du-trim rides the same translation (per-slot residual).
    du_off = args.contact_du - CONTACT_DU + (args.du_trim or 0.0)
    # far-slot glide clearance: ride the corridor 50 mm high in
    # place-spine mode (hover2 descends to the true hover before the
    # descent).
    hover_lift = 0.05 if args.place_spine is not None else 0.0
    glide_converge("corr_in", u_pt(CONTACT_DU + du_off + TCP_BOTTOM_DX + 0.010,
                                   -0.050, CORRIDOR_Z + hover_lift),
                   GRIP_REACH, hold_ref=None)
    hover_pt = u_pt(CONTACT_DU + du_off + TCP_BOTTOM_DX, 0.0,
                    CORRIDOR_Z + hover_lift)
    glide_converge("hover", hover_pt, GRIP_REACH)
    # hover re-aim: measure where the hanging bottom edge plumbs and along
    # which line it will unroll (the bottom edge is the strip's WIDTH line;
    # the pad lays along that line's normal). The strip does not hang square
    # to the tool. CLOSED loop: rotating the tool rotates the held strip, so
    # re-issue the hover pose at the new heading and re-measure — no
    # calibrated gain (the hanging reading over-reads the laid yaw).
    lat_trim = args.lat_trim or 0.0
    dv_aim = lat_trim
    du_aim = 0.0
    for _rnd in range(max(1, args.yaw_aim_rounds)):
        strip_settle()
        st_h = strip()
        p_h = got["pad"]
        low = p_h[p_h[:, 2] < st_h["z_lo"] + 0.012]
        if len(low) < 4:
            print("  hover re-aim: bottom band too small to read; "
                  "laying on the tracked plate centre", flush=True)
            break
        bx, by = float(low[:, 0].mean()), float(low[:, 1].mean())
        eu = (bx - frame.tx) * frame.ca + (by - frame.ty) * frame.sa
        ev = -(bx - frame.tx) * frame.sa + (by - frame.ty) * frame.ca
        # the bottom-edge AIM is contact_du minus half the low-band depth
        du_aim = max(min(args.contact_du - 0.006 - eu, 0.015), -0.015)
        # static trim rides OUTSIDE the clamp; the cap guards only the
        # live hover estimate
        dv_aim = lat_trim + max(min(0.0 - ev, args.dv_aim_cap),
                                -args.dv_aim_cap)
        print(f"  hover re-aim: bottom edge at (u {eu * 1000:+.0f}, "
              f"v {ev * 1000:+.0f}) mm -> shift (du {du_aim * 1000:+.0f}, "
              f"dv {dv_aim * 1000:+.0f}) mm", flush=True)
        lay_yaw = band_lay_yaw(low)
        if lay_yaw is None:
            break
        dyaw = _wrap_half(lay_yaw - frame.theta)
        print(f"  hover heading [{_rnd}]: bottom edge lays along "
              f"{math.degrees(lay_yaw):+.2f} deg vs board "
              f"{math.degrees(frame.theta):+.2f} -> off "
              f"{math.degrees(dyaw):+.2f} deg (tool yaw "
              f"{math.degrees(cur_yaw):+.2f}, {len(low)} band pts)",
              flush=True)
        if args.yaw_aim_gain <= 0.0:
            break                            # measure-and-log only
        if abs(math.degrees(dyaw)) <= args.yaw_aim_tol:
            break
        cap_ = math.radians(args.yaw_aim_cap)
        corr_y = max(min(args.yaw_aim_gain * dyaw, cap_), -cap_)
        cur_yaw = _wrap_half(cur_yaw - corr_y)
        print(f"  hover heading: tool yaw -> {math.degrees(cur_yaw):+.2f} "
              f"deg (applied {math.degrees(-corr_y):+.2f})", flush=True)
        if _rnd + 1 < max(1, args.yaw_aim_rounds):
            # re-issue the SAME hover point at the new heading, then loop
            go("hov_yaw", hover_pt, GRIP_REACH, CLOSED_GRIP,
               step=args.stream_step, settle=2.0, iters=3)

    if args.place_spine is not None and not args.dry:
        # place-phase spine, driven AT THE HOVER. The drive holds JOINTS
        # (TCP and hanging strip ride the base by the spine delta):
        # interleave <=0.08 m pre-lifts with drops so the strip keeps
        # ground clearance; sp_lift soft (an eased lift still clears with
        # the glide-lift margin); grips stay CLOSED through every settle.
        sp_now = float(got.get("spine", args.spine))
        while sp_now - args.place_spine > 0.005:
            step_ = min(0.08, sp_now - args.place_spine)
            go("sp_lift", (hover_pt[0], hover_pt[1], hover_pt[2] + step_),
               GRIP_REACH, CLOSED_GRIP, step=args.stream_step, settle=2.0,
               iters=2, soft=True)
            sp_now = raise_spine(n, sp_now - step_, {SIDE: cmd, OTHER: TRAVEL},
                                 grips={SIDE: CLOSED_GRIP, OTHER: OPEN_GRIP})
            strip_settle()
            if strip()["z_hi"] < 0.050:
                print("  !! strip DROPPED during the place-spine drive",
                      flush=True)
                return bail(7)
        if args.place_spine - sp_now > 0.005:
            # up-moves ride the strip upward — no lift needed
            raise_spine(n, args.place_spine, {SIDE: cmd, OTHER: TRAVEL},
                        grips={SIDE: CLOSED_GRIP, OTHER: OPEN_GRIP})
        # descend to the TRUE hover at the new base height (drops the
        # glide-clearance lift; short vertical move, no drag exposure)
        hover_pt = u_pt(CONTACT_DU + du_off + TCP_BOTTOM_DX, 0.0, CORRIDOR_Z)
        glide_converge("hover2", hover_pt, GRIP_REACH)

    # descent: lower until the bottom edge contacts the board
    contact = False
    z_now = CORRIDOR_Z
    while z_now > DESC_TCP_Z_FLOOR:
        z_now = max(z_now - 0.006, DESC_TCP_Z_FLOOR)
        go("desc", u_pt(CONTACT_DU + du_off + TCP_BOTTOM_DX + du_aim,
                        dv_aim, z_now),
           GRIP_REACH, CLOSED_GRIP, step=0.004, settle=2.0, iters=2)
        if strip()["z_lo"] <= CONTACT_Z_GATE:
            contact = True
            break
    st_c = strip()
    print(f"  bottom-edge contact: {'yes' if contact else 'NO (floor hit)'} "
          f"at TCP z {z_now:.4f}, strip z_lo {st_c['z_lo']:.4f}", flush=True)
    save_frame("contact")
    if args.stop_after == "contact":
        return bail(0)

    # press west-down at -37 (the official inter-ramp hold, ~2.5 s of it)
    kstate: dict = {}
    # the ep3 press walks the TCP 27 mm WEST while pressed (0.062 -> 0.035);
    # --press-du-scale scales this travel toward a vertical press
    press_u0 = PRESS[0][0]
    for i, (pu, pz) in enumerate(PRESS):
        pu = press_u0 + (pu - press_u0) * args.press_du_scale
        pt_ = u_pt(pu + du_off + du_aim, dv_aim, pz)
        go(f"press{i}", pt_, GRIP_REACH, CLOSED_GRIP, step=0.004,
           settle=args.press_settle, iters=args.press_iters)
        station_keep(pt_, GRIP_REACH, kstate, gate=0.030)
    keep_report("press", kstate)

    # the pitch sweep: -37 -> -63.7 along the ep3 TCP path
    opened = False
    for pd, su, sz in SWEEP:
        cur_pitch = math.radians(pd)
        pt_ = u_pt(su + du_off + du_aim, dv_aim, sz)
        go(f"sw{abs(pd):.0f}", pt_, GRIP_REACH, CLOSED_GRIP,
           step=args.sweep_step, settle=args.sweep_settle,
           iters=args.sweep_iters, wp_iters=args.sweep_wp_iters)
        ff = flat_frac()
        if pd <= args.open_arm_pitch and ff >= args.flat_gate:
            print(f"  pad FLAT at pitch {pd:+.1f} (flat fraction "
                  f"{ff:.2f}); opening", flush=True)
            opened = True
            break
    if not opened:
        # ep3-style short wait at the last rung, then open regardless: the
        # pad is within a few deg of flat by construction here
        t_wait = time.time() + 4.0 * room_mode.pace()
        while time.time() < t_wait:
            n.send(SIDE, cmd, grip_now)
            n.send(OTHER, TRAVEL, OPEN_GRIP)
            rclpy.spin_once(n, timeout_sec=0.05)
            time.sleep(0.03)
            if flat_frac() >= args.flat_gate:
                break
        print(f"  opening at the final rung (flat fraction "
              f"{flat_frac():.2f})", flush=True)
    if args.stop_after == "sweep":
        return bail(0)
    carrying[0] = False
    grip_now = OPEN_GRIP
    pump(4.0)
    save_frame("open")

    # ---- retreat at the final pitch, then park ----------------------------
    # soft: the strip is RELEASED — an unreachable retreat waypoint must
    # never abort the episode; the joint-interpolated park below clears the
    # camera regardless
    for i, (ru, rv, rz, rp) in enumerate(RETREAT):
        cur_pitch = math.radians(rp)
        go(f"ret{i}", u_pt(ru, rv, rz), GRIP_REACH, OPEN_GRIP,
           step=args.stream_step, settle=2.0, iters=2, soft=True)

    # park via INTERPOLATION through the seed family: a direct TRAVEL fold
    # sweeps the elbow across the board; an unparked arm occludes the eval
    # camera (no_target_bbox)
    qs_ = np.asarray(Q_SEED_OFF, dtype=np.float64)
    qa_ = np.asarray(cmd, dtype=np.float64)
    for k in range(1, 21):
        pstep = qa_ + (qs_ - qa_) * (k / 20.0)
        t_end = time.time() + 0.40 * room_mode.pace()
        while time.time() < t_end:
            n.send(SIDE, pstep, OPEN_GRIP)
            n.send(OTHER, TRAVEL, OPEN_GRIP)
            rclpy.spin_once(n, timeout_sec=0.05)
            time.sleep(0.03)
    pa = qs_
    pb = np.asarray(TRAVEL, dtype=np.float64)
    for k in range(1, 41):
        pstep = pa + (pb - pa) * (k / 40.0)
        t_end = time.time() + 0.15 * room_mode.pace()
        while time.time() < t_end:
            n.send(SIDE, pstep, OPEN_GRIP)
            n.send(OTHER, TRAVEL, OPEN_GRIP)
            rclpy.spin_once(n, timeout_sec=0.05)
            time.sleep(0.03)
    cmd = pb.copy()

    # the verdict is read with the arm parked: put the spine back where
    # the scene started it
    if args.park_spine and abs(spine_start - args.spine) > SPINE_TOL:
        raise_spine(n, spine_start, {SIDE: cmd, OTHER: np.asarray(TRAVEL)})

    dump_place_log()
    refresh_frame("final")   # score the plate where the parked arm left it
    st = strip()
    print(f"\nfinal strip: {fmt(st)}")
    # TRACKED frame: the evaluator scores the plate where it ENDS UP
    hx, hy = frame.aabb_half_extents()
    ftx, fty = frame.tx, frame.ty
    print(f"target AABB x {ftx - hx:.4f}..{ftx + hx:.4f}, "
          f"y {fty - hy:+.4f}..{fty + hy:+.4f} "
          f"(yaw {math.degrees(frame.theta):+.1f} deg)")
    # loose-analog target bbox in eval-camera px; parsed by official_run's
    # self-verdict
    print(f"final target AABB px "
          f"({_lf.PX0 + (ftx - hx) * _lf.PXS:.1f},"
          f"{_lf.PY0 - (fty + hy) * _lf.PYS:.1f},"
          f"{_lf.PX0 + (ftx + hx) * _lf.PXS:.1f},"
          f"{_lf.PY0 - (fty - hy) * _lf.PYS:.1f})")
    ix = max(0.0, min(st["x_hi"], ftx + hx) - max(st["x_lo"], ftx - hx))
    iy = max(0.0, min(st["y_hi"], fty + hy) - max(st["y_lo"], fty - hy))
    ua = ((st["x_hi"] - st["x_lo"]) * (st["y_hi"] - st["y_lo"])
          + 4.0 * hx * hy - ix * iy)
    print(f"crude bbox IoU estimate {ix * iy / ua if ua > 0 else 0:.3f}")
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
