"""Rot guards for the scoring mechanism (mirror_lay.py, official pattern).

The module needs rclpy so it cannot be imported here; these checks read the
source instead. They pin the cross-module contract official_run relies on
(stdout markers, flags) and the official-pattern constants, and ensure no
host-specific debug path is hardcoded (the container's runtime user cannot
write /root/ebim).
"""

from __future__ import annotations

import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ebim_task2" / "mirror_lay.py"
SOURCE = SRC.read_text(encoding="utf-8")


def test_official_pattern_flags_exist():
    flags = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', SOURCE))
    for needed in ("--pinch-z", "--contact-du", "--flat-gate",
                   "--open-arm-pitch", "--grasp-tries", "--base-x",
                   "--base-y", "--film", "--target"):
        assert needed in flags, f"official-pattern flag {needed} vanished"


def test_official_run_stdout_markers_survive():
    """official_run parses these exact stdout strings; renaming one silently
    breaks the accept loop's completed/flip/self-verdict detection."""
    for marker in ("crude bbox IoU estimate",
                   "FLIPPED on the pedestal",
                   "final target AABB px "):
        assert marker in SOURCE, f"stdout marker {marker!r} vanished"


def test_official_pattern_constants():
    """The ep3-derived geometry: pitch -5 grasp / -37 carry / sweep ending
    ~-64 with monotonic pitch; roll pi (the phi=0 family cannot reach the
    sweep); the grip point is the official +0.1498 m."""
    import math
    ns: dict = {"math": math}
    lines = SOURCE.splitlines()
    for name in ("GRIP_REACH", "GRIP_PHI", "PITCH_GRASP", "PITCH_CARRY",
                 "WALL_W_X", "WALL_TOP_Z", "CONTACT_DU", "SWEEP", "PRESS",
                 "Q_SEED_OFF"):
        i = next(k for k, ln in enumerate(lines)
                 if re.match(rf"{name}\s*=", ln))
        stmt = lines[i]
        while (stmt.count("(") > stmt.count(")")
               or stmt.count("[") > stmt.count("]")):
            i += 1
            stmt += "\n" + lines[i]
        exec(stmt, ns)
    assert ns["GRIP_REACH"] == pytest.approx(0.1498)
    assert ns["GRIP_PHI"] == pytest.approx(math.pi)
    assert math.degrees(ns["PITCH_GRASP"]) == pytest.approx(-5.0)
    assert math.degrees(ns["PITCH_CARRY"]) == pytest.approx(-37.15)
    assert ns["WALL_W_X"] == pytest.approx(0.760)
    pitches = [r[0] for r in ns["SWEEP"]]
    assert pitches == sorted(pitches, reverse=True)
    assert -64.5 < pitches[-1] < -63.0
    # the sweep TCP walks west (u decreasing) as the pad unrolls
    us = [r[1] for r in ns["SWEEP"][2:]]
    assert us == sorted(us, reverse=True)
    # seed family stays inside the SAFE envelope
    safe = [(-2.901, 2.901), (-1.836, 1.836), (-2.901, 2.901),
            (-3.077, -0.117), (-2.876, 2.876), (0.440, 4.622),
            (-3.051, 3.051)]
    for q, (lo, hi) in zip(ns["Q_SEED_OFF"], safe):
        assert lo + 0.05 < q < hi - 0.05


def test_debug_outputs_are_relocatable_and_nonfatal():
    # Every debug artefact path must go through DEBUG_DIR (env-overridable);
    # a bare /root/ebim write would crash the chain in the official container.
    assert "EBIM_MIRROR_DEBUG_DIR" in SOURCE
    writes = re.findall(r'"(/root/ebim[^"]*)"', SOURCE)
    assert writes == ["/root/ebim"], f"hardcoded rig paths beyond DEBUG_DIR: {writes}"


def _helpers():
    """mirror_lay imports rclpy at module scope, so pull the pure helpers
    out of the source the same way the rot guards above read it."""
    import math
    import numpy as np
    src = SRC.read_text(encoding="utf-8")
    ns = {"np": np, "math": math}
    for fn in ("def _wrap_half", "def pad_yaw"):
        i = src.index(fn)
        exec(src[i:src.index("\ndef ", i + 5)], ns)
    return ns["pad_yaw"], ns["_wrap_half"]


def test_pad_yaw_recovers_a_rotated_strip():
    """A rotated strip breaks an AXIS-ALIGNED pick, whose -x extreme is then
    a CORNER rather than the edge midpoint.

    The strip always SPAWNS nominal, but it is rotated in run by a failed
    grasp and by the tow the release cascade puts through it, and
    `remeasure_strip` replans off whatever yaw it then finds."""
    import math
    import numpy as np
    pad_yaw, wrap_half = _helpers()

    # the strip's axis has no head or tail: +170 deg IS -10 deg, and a
    # 170 deg tool swing would tear the grasp
    assert math.degrees(wrap_half(math.radians(170.0))) == pytest.approx(-10.0)

    rng = np.random.default_rng(0)
    for deg in (0.0, 3.0, -9.4, 12.0, -20.0):
        th = math.radians(deg)
        s = rng.uniform(-0.0525, 0.0525, 800)      # 105 mm long
        t = rng.uniform(-0.0103, 0.0103, 800)      # 20.6 mm wide
        pts = np.stack([s * math.cos(th) - t * math.sin(th) + 0.80,
                        s * math.sin(th) + t * math.cos(th) - 0.30,
                        np.full(800, 0.09)], axis=1)
        assert math.degrees(pad_yaw(pts)) == pytest.approx(deg, abs=0.8)

    # no usable direction must degenerate to the axis-aligned plan, never
    # to an arbitrary tool roll
    round_ = np.stack([rng.normal(0, 0.01, 500), rng.normal(0, 0.01, 500),
                       np.zeros(500)], axis=1)
    assert pad_yaw(round_) == 0.0
    assert pad_yaw(np.zeros((10, 3))) == 0.0


def _consts(*names: str) -> dict:
    """Exec named module-level constants out of the source (mirror_lay needs
    rclpy, so it cannot be imported here)."""
    import math
    ns: dict = {"math": math}
    lines = SOURCE.splitlines()
    for name in names:
        i = next(k for k, ln in enumerate(lines)
                 if re.match(rf"{name}\s*=", ln))
        stmt = lines[i]
        while (stmt.count("(") > stmt.count(")")
               or stmt.count("[") > stmt.count("]")):
            i += 1
            stmt += "\n" + lines[i]
        exec(stmt, ns)
    return ns


def test_spine_is_commanded_at_the_official_height():
    """The height is the one value that puts the arm base where the 22
    official episodes had it relative to the board plane."""
    from ebim_task2 import room_mode
    ns = _consts("SPINE_JOINT", "SPINE_LIMITS", "SPINE_TARGET", "ARM_MOUNT_Z")
    assert ns["SPINE_JOINT"] == "franka_spine_vertical_joint"
    lo, hi = ns["SPINE_LIMITS"]
    assert (lo, hi) == (0.0, 0.85)
    assert lo <= ns["SPINE_TARGET"] <= hi
    base_z = ns["ARM_MOUNT_Z"] + ns["SPINE_TARGET"] - room_mode.LIFT_Z
    assert base_z == pytest.approx(0.2349, abs=0.002)
    # ...and the chain drives it, before any reach
    assert '"--spine"' in SOURCE
    assert "raise_spine(n, args.spine" in SOURCE
    assert SOURCE.index("raise_spine(n, args.spine") < SOURCE.index("relocalise()")
    # the target rides in the arm group's command message; the bridge
    # resolves a group's commands by joint name
    send = SOURCE[SOURCE.index("    def send("):SOURCE.index("def loose_target_px")]
    assert "m.name.append(SPINE_JOINT)" in send


def test_seed_family_is_a_pose_that_can_be_held():
    """Q_SEED_OFF is COMMANDED (reconf, park, bail), so it must clear the
    board plane at the commanded spine height. Barebone and room arm-base
    geometry differ by 0.75 m: +0.501 above its objects vs -0.2509 below
    the boards."""
    import numpy as np
    from ebim_task2 import room_mode
    from ebim_task2.motion import franka_fk
    ns = _consts("Q_SEED_OFF", "Q_SEED_TCP", "SPINE_TARGET", "ARM_MOUNT_Z",
                 "GRIP_REACH")
    # arm-base mount rotation in the body frame (task2_fixpos_v1,
    # ee_pose x FK^-1)
    mount_r = np.array([[+0.880980, -0.401150, +0.250905],
                        [+0.440171, +0.500328, -0.745602],
                        [+0.173563, +0.767301, +0.617353]])
    base_z = ns["ARM_MOUNT_Z"] + ns["SPINE_TARGET"] - room_mode.LIFT_Z
    flange = franka_fk(np.asarray(ns["Q_SEED_OFF"], dtype=float),
                       tool_offset=0.0)
    grip = flange[:3, 3] + flange[:3, 2] * ns["GRIP_REACH"]
    grip_z = base_z + float((mount_r @ grip)[2])      # station is x/y only
    assert grip_z == pytest.approx(ns["Q_SEED_TCP"][2], abs=0.01)
    assert grip_z > 0.10, "the held family must not park the tool on the boards"

