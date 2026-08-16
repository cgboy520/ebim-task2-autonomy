"""Tests for the ground-truth-free pad geometry estimator.

The four numbers the scripted policy needs about the pad — where it is,
where its top cluster is, its crest height, and the surface it rests on —
recovered from the eval camera alone, without the ground-truth topic.
These tests pin the arithmetic against the scene's reference values
(tray 0.017, pad crest 0.102, camera 1.95 m, fx 1221.665).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebim_task2.perception import estimate_pad_geometry_from_depth  # noqa: E402

CAM_Z, FX = 1.95, 1221.665
KW = dict(thermalpad_id=2, liner_id=5, camera_height_m=CAM_Z, focal_px=FX,
          cx=640.0, cy=360.0, origin_xy=(0.837, -0.065), flip_y=True)


def _scene(pad_z=0.102, tray_z=0.017, pad_px=(507, 521, 580, 648)):
    """A 720p top-down frame: floor, an 11 cm tray, and the pad on it."""
    mask = np.ones((720, 1280), dtype=np.int32)
    depth = np.full((720, 1280), CAM_Z, dtype=np.float64)
    # tray footprint, generously around the pad
    depth[470:560, 555:675] = CAM_Z - tray_z
    y0, y1, x0, x1 = pad_px
    mask[y0:y1, x0:x1] = 5
    depth[y0:y1, x0:x1] = CAM_Z - pad_z
    return mask, depth


def test_heights_match_the_measured_scene():
    g = estimate_pad_geometry_from_depth(*_scene(), **KW)
    assert g is not None
    assert g.z_top == pytest.approx(0.102, abs=1e-4)
    # The support surface is the ring OUTSIDE the pad: the tray, not the floor.
    assert g.z_bottom == pytest.approx(0.017, abs=1e-4)


def test_world_xy_uses_the_scale_at_the_pad_height_not_a_flat_one():
    """The pinhole scale is fx/(camera_z - object_z). Using a scale calibrated
    at another height is a radial error about the principal point."""
    mask, depth = _scene()
    g = estimate_pad_geometry_from_depth(mask, depth, **KW)
    assert g is not None

    scale = FX / (CAM_Z - 0.102)
    px = (580 + 648 - 1) / 2.0
    py = (507 + 521 - 1) / 2.0
    assert g.centroid_xy[0] == pytest.approx(0.837 + (px - 640.0) / scale, abs=2e-4)
    assert g.centroid_xy[1] == pytest.approx(-0.065 - (py - 360.0) / scale, abs=2e-4)
    # A flat 660.6 px/m would land somewhere else entirely.
    assert abs(g.centroid_xy[1] - (-0.065 - (py - 360.0) / 660.6)) > 1e-4


def test_crest_ignores_a_single_noisy_pixel():
    """A high percentile, not max: one bad depth sample must not re-price the
    press depth (the GT feed read 0.116 on a settled 0.102 crest)."""
    mask, depth = _scene()
    depth[510, 600] = CAM_Z - 0.30          # one absurd spike inside the blob
    g = estimate_pad_geometry_from_depth(mask, depth, **KW)
    assert g is not None
    assert g.z_top == pytest.approx(0.102, abs=2e-3)


def test_returns_none_when_the_pad_is_not_visible():
    mask, depth = _scene()
    mask[:] = 1                              # arm occludes everything
    assert estimate_pad_geometry_from_depth(mask, depth, **KW) is None


def test_returns_none_on_unusable_depth():
    mask, depth = _scene()
    depth[:] = np.nan
    assert estimate_pad_geometry_from_depth(mask, depth, **KW) is None


def test_floor_pad_reports_the_floor_as_its_support():
    """Once the pad is off the tray the clamp must follow it down, or the
    grasp keeps aiming 8 cm too high."""
    mask, depth = _scene(pad_z=0.019, tray_z=0.0, pad_px=(300, 314, 580, 648))
    g = estimate_pad_geometry_from_depth(mask, depth, **KW)
    assert g is not None
    assert g.z_top == pytest.approx(0.019, abs=1e-4)
    assert g.z_bottom == pytest.approx(0.0, abs=1e-4)


# ---- runner wiring -------------------------------------------------------
def test_runner_installs_depth_geometry_onto_the_policy():
    """The vision path has to land on the same policy attributes the GT feed
    writes, or switching sources silently changes nothing."""

    from ebim_task2.config import (
        CameraConfig, ControlConfig, PadSourceConfig, SemanticIds, Task2Config,
        TopicConfig,
    )
    from ebim_task2.runner import Task2AutonomyNode
    from ebim_task2.vla_policy import PerceptionPolicy

    cfg = Task2Config(
        control=ControlConfig(enabled=False),
        topics=TopicConfig(arm_state="/a", arm_command="/b",
                           gripper_command="/c", semantic_image="/d"),
        semantic_ids=SemanticIds(thermalpad=2, target=3, liner=5),
        camera=CameraConfig(cx=640.0, cy=360.0, pixel_scale=660.6,
                            origin_x_m=0.837, origin_y_m=-0.065, flip_y=True,
                            camera_height_m=CAM_Z, focal_px=FX),
        pad_source=PadSourceConfig(source="depth"),
    )
    pol = PerceptionPolicy()
    node = Task2AutonomyNode(cfg, pol)
    mask, depth = _scene()
    node._depth = depth

    node._apply_depth_pad_geometry(
        mask, {"thermalpad": 2, "target": 3, "liner": 5})

    assert pol.pad_gt_z_top == pytest.approx(0.102, abs=1e-4)
    assert pol.pad_gt_z_bottom == pytest.approx(0.017, abs=1e-4)
    assert pol.pad_world_gt is not None
    assert pol.pad_world_gt_top is not None
    # The grasp depth clamps against the camera-measured tray.
    pol._grasp_support_margin = 0.006
    pol._grasp_depth = 0.010
    assert pol._grasp_z() == pytest.approx(0.092, abs=1e-4)


def test_depth_path_is_inert_when_focal_length_is_unset():
    """Rescaling per height needs fx; without it the estimator must not run on
    a flat scale and quietly produce plausible-but-wrong metres."""
    from ebim_task2.config import (
        CameraConfig, ControlConfig, PadSourceConfig, SemanticIds, Task2Config,
        TopicConfig,
    )
    from ebim_task2.runner import Task2AutonomyNode
    from ebim_task2.vla_policy import PerceptionPolicy

    cfg = Task2Config(
        control=ControlConfig(enabled=False),
        topics=TopicConfig(arm_state="/a", arm_command="/b",
                           gripper_command="/c", semantic_image="/d"),
        semantic_ids=SemanticIds(thermalpad=2, target=3, liner=5),
        camera=CameraConfig(camera_height_m=CAM_Z, focal_px=None),
        pad_source=PadSourceConfig(source="depth"),
    )
    pol = PerceptionPolicy()
    node = Task2AutonomyNode(cfg, pol)
    mask, depth = _scene()
    node._depth = depth
    node._apply_depth_pad_geometry(mask, {"thermalpad": 2, "target": 3, "liner": 5})
    assert pol.pad_gt_z_top is None
