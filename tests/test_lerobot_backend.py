"""Action-space bridge between a LeRobot checkpoint and the single-arm runner.

The dataset action row is the recorder's 20-D whole-body vector; the runner
drives one 7-joint arm plus its gripper. Slicing the wrong window publishes
the OTHER arm's trajectory to this arm — plausible-looking motion, entirely
wrong — so the mapping is pinned here with a fake policy (no torch weights,
no GPU).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch", reason="LeRobotBackend needs torch")

from ebim_task2.backends import VLAConfig, VLAInput  # noqa: E402
from ebim_task2.backends.vla_runtime import LeRobotBackend  # noqa: E402


class _FakePolicy:
    """Returns a fixed 20-D action whose every channel is distinguishable."""

    def __init__(self, image_keys=("observation.images.eval_camera",)):
        self.config = type("C", (), {"input_features": {k: None for k in image_keys}})()
        self.seen: dict | None = None

    def select_action(self, batch):
        self.seen = batch
        act = torch.arange(20, dtype=torch.float32) / 100.0
        return act.unsqueeze(0)


def _backend(image_keys=("observation.images.eval_camera",), arm="right"):
    pol = _FakePolicy(image_keys)
    ident = lambda x: x  # noqa: E731 - processors are identity in this test
    return LeRobotBackend(pol, ident, ident, "cpu",
                          VLAConfig(backend="lerobot", active_arm=arm)), pol


def _inp(images):
    return VLAInput(joints=np.zeros(7), gripper_open=1.0, image=None,
                    instruction=None, state=np.zeros(37, dtype=np.float32),
                    images=images)


def test_right_arm_slice_and_gripper_channel():
    backend, _ = _backend()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    out = backend.propose(_inp({"eval_camera": frame}), horizon=8)

    # action[10:17] is the RIGHT arm; action[18] its gripper.
    assert np.allclose(out.chunk[0], np.arange(10, 17) / 100.0)
    assert out.gripper_seq == [pytest.approx(0.18)]
    assert out.is_delta is False


def test_base_twist_channels_are_forwarded():
    """action[0:3] is the base twist; the winning demos ride the base, so
    dropping these channels would leave a fine-tuned policy unable to reach
    the place lane."""
    backend, _ = _backend()
    out = backend.propose(_inp({"eval_camera": np.zeros((8, 8, 3), np.uint8)}),
                          horizon=8)
    assert out.base_seq is not None
    assert np.allclose(out.base_seq[0], np.arange(0, 3) / 100.0)


def test_spine_channel_is_forwarded():
    """action[19] is the vertical spine, and it is NOT optional.

    Every demo ramps it 0 -> ~0.486 m within the first second and holds; at
    spine 0 the arm base sits 0.2509 m BELOW the board plane, so a policy
    trained on spine-inclusive demos cannot reach any board without it. This
    channel was silently dropped here once — the docstring named it while the
    slice table did not — which makes a correct-looking motion that never
    touches the boards."""
    backend, _ = _backend()
    out = backend.propose(_inp({"eval_camera": np.zeros((8, 8, 3), np.uint8)}),
                          horizon=8)
    assert out.spine_seq is not None
    assert out.spine_seq[0] == pytest.approx(0.19)


def test_left_arm_selects_the_other_window():
    backend, _ = _backend(arm="left")
    out = backend.propose(_inp({"eval_camera": np.zeros((8, 8, 3), np.uint8)}), horizon=8)
    assert np.allclose(out.chunk[0], np.arange(3, 10) / 100.0)
    assert out.gripper_seq == [pytest.approx(0.17)]


def test_images_are_batched_chw_floats_in_unit_range():
    backend, pol = _backend()
    frame = np.full((4, 6, 3), 255, dtype=np.uint8)

    backend.propose(_inp({"eval_camera": frame}), horizon=1)

    img = pol.seen["observation.images.eval_camera"]
    assert img.shape == (1, 3, 4, 6)          # B, C, H, W
    assert float(img.max()) == pytest.approx(1.0)
    assert pol.seen["observation.state"].shape == (1, 37)
    assert isinstance(pol.seen["task"], list)


def test_missing_camera_or_state_holds_instead_of_guessing():
    """A missing stream would otherwise be zero-filled into a confidently
    wrong action; an empty chunk makes the policy hold."""
    backend, _ = _backend()
    assert backend.propose(_inp({}), horizon=8).chunk == []

    no_state = VLAInput(joints=np.zeros(7), gripper_open=1.0, image=None,
                        instruction=None, state=None,
                        images={"eval_camera": np.zeros((4, 4, 3), np.uint8)})
    assert backend.propose(no_state, horizon=8).chunk == []


def test_only_trained_cameras_are_required():
    """The checkpoint's own feature list drives which streams are needed —
    a recorder default of four cameras must not block a one-camera model."""
    backend, pol = _backend(image_keys=("observation.images.wrist_right",))
    out = backend.propose(_inp({"wrist_right": np.zeros((4, 4, 3), np.uint8),
                                "head": np.zeros((4, 4, 3), np.uint8)}), horizon=1)
    assert out.chunk
    assert "observation.images.head" not in pol.seen
