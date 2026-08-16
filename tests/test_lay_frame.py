"""Unit tests for the target-frame lay geometry (lay_frame.py).

The theta=0 identities pin backward-equivalence: with a zero measured yaw
the rotated-frame mechanism must reproduce the exact axis-aligned world
points.
"""

from __future__ import annotations


import math

import pytest

from ebim_task2.lay_frame import LayFrame, quat_wxyz_yaw, wrap_slot_yaw


def _quat_about_z(theta: float) -> tuple[float, float, float, float]:
    return (math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2))


class TestQuatYaw:
    def test_identity(self):
        assert quat_wxyz_yaw(1, 0, 0, 0) == pytest.approx(0.0)

    @pytest.mark.parametrize("deg", [-10, -3.7, 0.5, 8, 45])
    def test_pure_z_rotation_roundtrip(self, deg):
        th = math.radians(deg)
        assert quat_wxyz_yaw(*_quat_about_z(th)) == pytest.approx(th)


class TestWrapSlotYaw:
    @pytest.mark.parametrize("deg,expect", [
        (0, 0), (10, 10), (-10, -10), (170, -10), (-170, 10), (95, -85),
    ])
    def test_folds_to_half_pi(self, deg, expect):
        assert wrap_slot_yaw(math.radians(deg)) == pytest.approx(
            math.radians(expect))


class TestLayFrameZeroYawIsV12:
    """theta=0 must reproduce the world-x-aligned geometry exactly."""

    FR = LayFrame(0.8005, -0.0029, 0.0)

    def test_anchor_point(self):
        # ax = tx + sgn*(total/2 - inset); anchor u is the same offset.
        sgn, total, inset = 1.0, 0.120, 0.008
        au = sgn * (total / 2 - inset)
        assert self.FR.to_world(au, 0.0) == pytest.approx(
            (0.8005 + 0.052, -0.0029))

    def test_arc_point(self):
        sgn, hang, pad_half_w, au = 1.0, 0.1074, 0.0135, 0.052
        th = math.radians(32)
        u = au - sgn * (hang * math.cos(th) + pad_half_w)
        assert self.FR.to_world(u, 0.0)[0] == pytest.approx(
            0.8005 + au - (hang * math.cos(th) + pad_half_w))
        assert self.FR.to_world(u, 0.0)[1] == pytest.approx(-0.0029)

    def test_uv_roundtrip(self):
        u, v = self.FR.to_uv(0.836, 0.004)
        assert self.FR.to_world(u, v) == pytest.approx((0.836, 0.004))

    def test_rot_is_identity(self):
        assert self.FR.rot(0.004, 0.0) == pytest.approx((0.004, 0.0))

    def test_aabb_is_plate(self):
        assert self.FR.aabb_half_extents() == pytest.approx((0.060, 0.010))


class TestLayFrameRotated:
    TH = math.radians(10)
    FR = LayFrame(0.81, 0.01, TH)

    def test_u_axis_follows_yaw(self):
        x, y = self.FR.to_world(0.052, 0.0)
        assert x == pytest.approx(0.81 + 0.052 * math.cos(self.TH))
        assert y == pytest.approx(0.01 + 0.052 * math.sin(self.TH))

    def test_v_axis_is_lateral(self):
        x, y = self.FR.to_world(0.0, 0.004)
        assert x == pytest.approx(0.81 - 0.004 * math.sin(self.TH))
        assert y == pytest.approx(0.01 + 0.004 * math.cos(self.TH))

    def test_uv_roundtrip(self):
        for wx, wy in ((0.75, -0.04), (0.86, 0.03)):
            u, v = self.FR.to_uv(wx, wy)
            assert self.FR.to_world(u, v) == pytest.approx((wx, wy))

    def test_aabb_matches_observed_jitter(self):
        # A 10-deg draw reads ~41 mm tall on the eval camera (75x26 px
        # observed live) - the AABB estimate must agree.
        hx, hy = self.FR.aabb_half_extents()
        assert 2 * hy == pytest.approx(0.0405, abs=0.002)
        assert 2 * hx == pytest.approx(0.1217, abs=0.002)

    def test_lateral_error_projection(self):
        # A point sitting exactly ON the slot axis has v == 0 however far
        # along u it lies; a 4 mm lateral offset reads v == 4 mm.
        far = self.FR.to_world(0.055, 0.0)
        assert self.FR.to_uv(*far)[1] == pytest.approx(0.0, abs=1e-12)
        off = self.FR.to_world(0.055, 0.004)
        assert self.FR.to_uv(*off)[1] == pytest.approx(0.004)


import numpy as np

from ebim_task2 import lay_frame

# ---- band_lay_yaw: the hover heading measurement -------------------------
# The bottom-edge band is the strip's WIDTH line (~20 mm across, a few mm
# deep), so the axis the pad unrolls along is that line's NORMAL. Getting
# this 90-deg relationship backwards would aim the lay across the board.

def _band(yaw_deg, n=400, width=0.020, depth=0.003, seed=0):
    """A synthetic bottom-edge band whose strip lays along `yaw_deg`."""
    rng = np.random.default_rng(seed)
    # points along the WIDTH line, i.e. perpendicular to the lay axis
    t = rng.uniform(-width / 2, width / 2, n)
    d = rng.uniform(-depth / 2, depth / 2, n)
    lay = math.radians(yaw_deg)
    wx, wy = -math.sin(lay), math.cos(lay)      # width dir = lay normal
    lx, ly = math.cos(lay), math.sin(lay)
    x = t * wx + d * lx
    y = t * wy + d * ly
    return np.stack([x, y, np.zeros(n)], axis=1)


def test_band_lay_yaw_recovers_the_lay_axis():
    for want in (0.0, 1.5, -3.0, 4.7, -8.0, 12.0):
        got_ = lay_frame.band_lay_yaw(_band(want))
        assert got_ is not None
        assert abs(math.degrees(got_) - want) < 0.5, (want, math.degrees(got_))


def test_band_lay_yaw_is_180_symmetric():
    a = lay_frame.band_lay_yaw(_band(5.0))
    b = lay_frame.band_lay_yaw(_band(185.0))
    assert a is not None and b is not None
    assert abs(math.degrees(a) - math.degrees(b)) < 0.5


def test_band_lay_yaw_refuses_a_band_too_small_to_read():
    assert lay_frame.band_lay_yaw(_band(3.0, n=10)) is None
    assert lay_frame.band_lay_yaw(np.zeros((0, 3))) is None
