"""Tests for the semantic-stream plate estimator (lay_frame.py).

Fixtures are SYNTHESIZED rasters mimicking the eval camera's semantic
image (720x1280 int32): a rotated 120x20 mm plate plus the distractor
regions that share the frame (slot rows, chips cluster, pedestal liner).
Geometry mirrors the live scene: plate ~950 px, slot rows at py 184..390,
pedestal at py ~505.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ebim_task2.lay_frame import (
    PLATE_FULL_PX, PLATE_LINE_PX, PX0, PXS, PY0, PYS, SPAWN_X, SPAWN_Y,
    find_plate_id, measure_plate_id, project_center_to_line,
)


def draw_plate(seg, rid, wx, wy, theta_world, half_u=0.060, half_v=0.010,
               u_cover=None):
    """Rasterise a rotated plate at world (wx, wy, theta); u_cover=(u0,u1)
    blanks that plate-frame span (the laid strip covering it)."""
    cx, cy = PX0 + wx * PXS, PY0 - wy * PYS
    th_img = -theta_world
    ca, sa = math.cos(th_img), math.sin(th_img)
    ys, xs = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    dx, dy = xs - cx, ys - cy
    u = (ca * dx + sa * dy) / PXS          # back to metres, along axis
    v = (-sa * dx + ca * dy) / PYS
    m = (np.abs(u) <= half_u) & (np.abs(v) <= half_v)
    if u_cover is not None:
        m &= ~((u >= u_cover[0]) & (u <= u_cover[1]))
    seg[m] = rid


def scene(plate=(SPAWN_X, SPAWN_Y, 0.0), u_cover=None):
    seg = np.zeros((720, 1280), dtype=np.int32)
    # slot rows (board class, one shared id): x 575..657, four rows
    for y0, y1 in ((184, 202), (244, 265), (373, 390)):
        seg[y0:y1, 575:657] = 4
    # chips cluster (own id, tall union spanning rows)
    for y0 in (190, 250, 380):
        seg[y0:y0 + 8, 585:640:6] = 6
    # pedestal liner (plate-shaped but at py ~505)
    seg[498:512, 580:655] = 2
    draw_plate(seg, 5, *plate, u_cover=u_cover)
    return seg


class TestFindPlateId:
    def test_clean_flat_scene(self):
        assert find_plate_id(scene()) == 5

    @pytest.mark.parametrize("theta_deg", [-10, -4, 3, 10])
    def test_clean_tilted_scene(self, theta_deg):
        assert find_plate_id(
            scene((SPAWN_X, SPAWN_Y, math.radians(theta_deg)))) == 5

    def test_shifted_plate_still_found(self):
        assert find_plate_id(scene((SPAWN_X + 0.020, SPAWN_Y - 0.018,
                                    0.0))) == 5

    def test_covered_plate_rejected(self):
        # Half-covered at chain start would be an anomaly - the lock
        # requires a clean plate.
        assert find_plate_id(scene(u_cover=(-0.06, 0.0))) is None

    def test_no_plate(self):
        seg = scene()
        seg[seg == 5] = 0
        assert find_plate_id(seg) is None


class TestMeasurePlateId:
    @pytest.mark.parametrize("wx,wy,theta_deg", [
        (SPAWN_X, SPAWN_Y, 0.0),
        (0.8129, 0.0152, -8.0),
        (0.7929, -0.0208, 6.5),
    ])
    def test_recovers_pose(self, wx, wy, theta_deg):
        seg = scene((wx, wy, math.radians(theta_deg)))
        m = measure_plate_id(seg, 5)
        assert m is not None
        mx, my, mth, npx, _nc = m
        assert mx == pytest.approx(wx, abs=0.002)
        assert my == pytest.approx(wy, abs=0.002)
        assert math.degrees(mth) == pytest.approx(theta_deg, abs=0.6)
        assert npx >= PLATE_FULL_PX

    def test_part_covered_line_still_good(self):
        # Strip laid over the +u half: centroid slides -u, but the line
        # (theta) must hold and the count must land in the line tier.
        th = math.radians(-7.0)
        seg = scene((SPAWN_X, SPAWN_Y, th), u_cover=(0.0, 0.061))
        m = measure_plate_id(seg, 5)
        assert m is not None
        mx, my, mth, npx, _nc = m
        assert PLATE_LINE_PX <= npx < PLATE_FULL_PX
        assert math.degrees(mth) == pytest.approx(-7.0, abs=0.8)
        # centroid must NOT be trusted as centre here:
        assert abs(mx - SPAWN_X) > 0.02

    def test_fully_covered_returns_none(self):
        seg = scene(u_cover=(-0.061, 0.061))
        assert measure_plate_id(seg, 5) is None

    def test_arm_shadow_cannot_fake_lateral(self):
        # An arm shadow clips rows on ONE side of a run of columns; their
        # centroids shift laterally and an unfiltered fit re-aims the lay
        # off-axis (smoke4-a2: -5.6 mm fake, killed by the anchor gate).
        # The column-height filter must drop those columns.
        seg = scene((SPAWN_X, SPAWN_Y, 0.0))
        ys, xs = np.nonzero(seg == 5)
        cy_mid = ys.mean()
        cols = np.unique(xs)
        for c in cols[len(cols) // 3: 2 * len(cols) // 3]:
            rows = np.nonzero(seg[:, c] == 5)[0]
            seg[rows[rows <= cy_mid], c] = 0
        m = measure_plate_id(seg, 5)
        assert m is not None
        mx, my, mth, _n, _c = m
        assert my == pytest.approx(SPAWN_Y, abs=0.0015)
        assert math.degrees(mth) == pytest.approx(0.0, abs=0.8)


class TestProjectCenterToLine:
    def test_lateral_snap_axial_keep(self):
        # Plate slid +2 cm laterally (world y), committed centre keeps its
        # x station but adopts the new lateral line.
        cx, cy = project_center_to_line(0.80, 0.00, 0.83, 0.02, 0.0)
        assert (cx, cy) == pytest.approx((0.80, 0.02))

    def test_rotated_line(self):
        th = math.radians(30)
        # New line passes through the old centre: projection is identity.
        cx, cy = project_center_to_line(0.80, 0.00, 0.80, 0.00, th)
        assert (cx, cy) == pytest.approx((0.80, 0.00))

    def test_estimator_roundtrip(self):
        # End-to-end part-covered flow: measure a covered plate, project
        # the committed centre onto the measured line - the result must
        # recover the true centre to ~1 mm.
        true = (0.8079, 0.0092, math.radians(-5.0))
        seg = scene(true, u_cover=(0.005, 0.061))
        mx, my, mth, _n, _c = measure_plate_id(seg, 5)
        cx, cy = project_center_to_line(true[0], true[1], mx, my, mth)
        assert cx == pytest.approx(true[0], abs=0.0012)
        assert cy == pytest.approx(true[1], abs=0.0012)
