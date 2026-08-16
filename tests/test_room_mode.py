"""room_mode: the rigid room->virtual conjugation, pinned against live
room-scene measurements."""
import json
import math

import numpy as np
import pytest

from ebim_task2 import lay_frame, room_mode


@pytest.fixture
def room_active():
    room_mode.activate(1280, None)
    yield
    room_mode.deactivate()


def test_scene_of_by_image_width(monkeypatch):
    monkeypatch.delenv("EBIM_SCENE", raising=False)
    assert room_mode.scene_of(np.zeros((480, 640, 3))) == "barebone"
    assert room_mode.scene_of(np.zeros((720, 1280, 3))) == "room"
    assert room_mode.scene_of(None) == "barebone"


def test_scene_of_env_override(monkeypatch):
    monkeypatch.setenv("EBIM_SCENE", "room")
    assert room_mode.scene_of(np.zeros((480, 640, 3))) == "room"
    monkeypatch.setenv("EBIM_SCENE", "barebone")
    assert room_mode.scene_of(np.zeros((720, 1280, 3))) == "barebone"


def test_strip_rest_maps_to_barebone_rest():
    # live room pad_points: x 1.740..1.760 y 1.898..2.003 z .834...852
    # live barebone rest:        x .746...851  y -.310..-.290  z .083...102
    v = room_mode.v_pt((1.75, 1.95, 0.85))
    assert v == pytest.approx((0.80, -0.30, 0.10), abs=1e-9)
    lo = room_mode.v_pt((1.740, 1.898, 0.834))
    hi = room_mode.v_pt((1.760, 2.003, 0.852))
    xs = sorted((lo[0], hi[0]))
    ys = sorted((lo[1], hi[1]))
    assert xs[0] == pytest.approx(0.747, abs=0.002)
    assert xs[1] == pytest.approx(0.852, abs=0.002)
    assert ys[0] == pytest.approx(-0.310, abs=0.002)
    assert ys[1] == pytest.approx(-0.290, abs=0.002)


def test_plate_spawn_maps_off_rigid_image():
    # the room target spawns 0.103 m off the rigid image: virtual (0.80, +0.10)
    v = room_mode.v_pt((2.15, 1.95, 0.75))
    assert v == pytest.approx((0.80, 0.10, 0.0), abs=1e-9)


def test_odom_spawn_maps_to_zero_yaw():
    vx, vy, _vz, vyaw = room_mode.v_odom(4.4, 2.6, 0.0, -math.pi / 2)
    assert (vx, vy) == pytest.approx((0.15, 2.35), abs=1e-9)
    assert vyaw == pytest.approx(0.0, abs=1e-9)


def test_pose_conjugation_rotates_orientation():
    t = np.eye(4)
    t[:3, 3] = (2.05, 1.95, 0.75)
    v = room_mode.v_pose(t)
    assert v[:3, 3] == pytest.approx((0.80, 0.0, 0.0), abs=1e-9)
    # room +x maps to virtual +y (Rz(+90))
    assert v[:3, 0] == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_rotated_mapping_matches_measured_plate(room_active):
    # live: plate marker centroid at room px (679.0, 319.0); rot90 CCW
    # puts it at (319, 600); the patched lay_frame constants must predict
    # the virtual plate spawn (0.80, +0.10) there within a few px.
    px = lay_frame.PX0 + 0.80 * lay_frame.PXS
    py = lay_frame.PY0 - 0.10 * lay_frame.PYS
    assert px == pytest.approx(319.0, abs=6.0)
    assert py == pytest.approx(600.0, abs=6.0)


def test_liner_crop_covers_measured_liner_blob(room_active):
    # live: the liner blob sits at rotated rows 854..868,
    # cols 283..351 — the crop must fully contain it
    r0, r1, c0, c1 = room_mode.liner_crop()
    assert 0 <= r0 < r1 <= 1280
    assert 0 <= c0 < c1 <= 720
    assert r0 <= 854 and r1 >= 868
    assert c0 <= 283 and c1 >= 351


def test_liner_count_room_blue_vs_barebone_cyan(room_active):
    # live liner sample mean RGB (70, 146, 211): the barebone cyan test
    # reads 0 in the room render; the room blue test must read it
    crop = np.zeros((10, 10, 3), dtype=np.int32)
    crop[:, :, 0] = 70
    crop[:, :, 1] = 146
    crop[:, :, 2] = 211
    assert room_mode.liner_count(crop) == 100
    # a flipped strip shows the pad's dark top: nothing blue-dominant
    dark = np.full((10, 10, 3), 60, dtype=np.int32)
    assert room_mode.liner_count(dark) == 0


def test_liner_count_barebone_formula_when_inactive():
    crop = np.zeros((4, 4, 3), dtype=np.int32)
    crop[:, :, 1] = 200
    crop[:, :, 2] = 180
    assert room_mode.liner_count(crop) == 16
    crop[:, :, 1] = 150     # cyan gate needs G > 180
    assert room_mode.liner_count(crop) == 0


def test_orient_seg_rotation_and_inactive_passthrough(room_active):
    seg = np.zeros((720, 1280), dtype=np.int32)
    seg[319, 679] = 7
    rot = room_mode.orient_seg(seg)
    assert rot.shape == (1280, 720)
    ys, xs = np.nonzero(rot == 7)
    assert (xs[0], ys[0]) == (319, 1279 - 679)


def test_orient_passthrough_when_inactive():
    seg = np.zeros((720, 1280), dtype=np.int32)
    assert room_mode.orient_seg(seg) is seg
    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert room_mode.orient_rgb(rgb) is rgb


def test_deactivate_restores_lay_frame():
    before = (lay_frame.PX0, lay_frame.PXS, lay_frame.PY0, lay_frame.PYS)
    room_mode.activate(1280, None)
    assert (lay_frame.PX0, lay_frame.PXS) != (before[0], before[1])
    room_mode.deactivate()
    assert (lay_frame.PX0, lay_frame.PXS,
            lay_frame.PY0, lay_frame.PYS) == before


def test_live_bbox_rotates_into_the_masks_frame(monkeypatch):
    """The evaluator streams raw camera pixels while _grab_mask serves the
    room mask rotated; feeding the two together scores IoU 0 for ANY lay."""
    from ebim_task2.official_run import rotate_bbox_for_room

    monkeypatch.setenv("EBIM_SCENE", "room")
    # the room plate marker occupies raw cols 673..685,
    # rows 282..356 -> rotated rows 594..606, cols 282..356
    raw = (282, 356, 673, 685)
    got = rotate_bbox_for_room(raw, 1280)
    assert got == (1279 - 685, 1279 - 673, 282, 356)
    assert got == (594, 606, 282, 356)
    # and that is where the chain's own rotated calibration puts the plate
    room_mode.activate(1280, None)
    try:
        px = lay_frame.PX0 + 0.80 * lay_frame.PXS
        py = lay_frame.PY0 - 0.10 * lay_frame.PYS
        # box is (y0, y1, x0, x1); py is a ROW, px a COLUMN
        assert got[0] <= py <= got[1]
        assert got[2] <= px <= got[3]
    finally:
        room_mode.deactivate()


def test_live_bbox_untouched_on_barebone(monkeypatch):
    from ebim_task2.official_run import rotate_bbox_for_room

    monkeypatch.setenv("EBIM_SCENE", "barebone")
    raw = (300, 340, 100, 180)
    assert rotate_bbox_for_room(raw, 640) == raw
    assert rotate_bbox_for_room(None, 1280) is None


def test_pace_rejects_an_implausible_factor():
    room_mode.activate(1280, None)
    try:
        room_mode.set_pace_from_factor(0.16)
        assert room_mode.pace() == pytest.approx(0.29 / 0.16, abs=0.01)
        # a wall-clock stamp reads ~1.0 and must NOT silently disable pacing
        keep = room_mode.pace()
        room_mode.set_pace_from_factor(1.0)
        assert room_mode.pace() == keep
        room_mode.set_pace_from_factor(0.0)
        assert room_mode.pace() == keep
    finally:
        room_mode.deactivate()
        room_mode.PACE = 1.8


def test_approach_legs_walk_the_corridor():
    legs = room_mode.approach_legs(2.35)
    assert legs[0] == (room_mode.PARK_X, 2.35)
    assert legs[-1] == (room_mode.PARK_X, room_mode.APPROACH_DONE_VY)
    vys = [ly for _, ly in legs]
    assert all(a >= b for a, b in zip(vys, vys[1:]))
    assert all(lx == room_mode.PARK_X for lx, _ in legs)
    assert room_mode.approach_legs(0.1) == []


def _poses(name="board_target", x=2.15, y=1.95, quat=(0.70711, 0.0, 0.0, 0.70711)):
    return json.dumps({"sim_time": 1.0,
                       "objects": {name: [x, y, 0.75, *quat]}})


def test_gt_plate_lands_on_the_measured_room_lock():
    """The room lock measured (0.7994, +0.0993) yaw -0.00 from the semantic
    raster; the ground-truth stream must agree, or the conjugation is wrong."""
    room_mode.activate(1280, None)
    try:
        vx, vy, th = room_mode.gt_plate_virtual(_poses())
        assert (vx, vy) == pytest.approx((0.80, 0.10), abs=0.001)
        assert math.degrees(th) == pytest.approx(0.0, abs=0.01)
    finally:
        room_mode.deactivate()


def test_gt_plate_follows_a_yaw_jitter():
    """A +100 deg world yaw is a +10 deg slot yaw, not a fold to -80.

    Every board spawns at the nominal +90 deg, but they get shoved and
    rotated in run (15/17 disturbed lays moved the target), so the slot-yaw
    conversion has to hold away from the spawn."""
    room_mode.activate(1280, None)
    try:
        c, s = math.cos(math.radians(50.0)), math.sin(math.radians(50.0))
        _, _, th = room_mode.gt_plate_virtual(_poses(quat=(c, 0.0, 0.0, s)))
        assert math.degrees(th) == pytest.approx(10.0, abs=1e-3)
    finally:
        room_mode.deactivate()


def test_gt_plate_is_none_without_a_usable_payload():
    """Barebone publishes one identity GROUP entry, not the plate; and the
    stream is absent entirely when the scene runs without --record."""
    assert room_mode.gt_plate_virtual(_poses()) is None      # not room mode
    room_mode.activate(1280, None)
    try:
        assert room_mode.gt_plate_virtual(None) is None
        assert room_mode.gt_plate_virtual("") is None
        assert room_mode.gt_plate_virtual("{not json") is None
        assert room_mode.gt_plate_virtual(_poses(name="task2_objects")) is None
        assert room_mode.gt_plate_virtual(
            json.dumps({"objects": {"board_target": [1.0, 2.0]}})) is None
    finally:
        room_mode.deactivate()


def test_room_only_constants_are_ordered_as_the_evidence_requires():
    """The base-drive constants encode measured thresholds; a silent edit
    that inverts them would quietly re-enable the failures they close."""
    # the reversal guard has to cover a ~40 mm overshoot, and must stay
    # well inside the spare arm reach at the place station
    assert 0.042 < room_mode.LOADED_REVERSAL_MAX_ERR < 0.100
    # the unstick pulse moves tens of mm, so it must be barred at least out
    # to the place-station tolerance it would otherwise overshoot
    assert room_mode.BASE_UNSTICK_MIN_ERR >= 0.025

