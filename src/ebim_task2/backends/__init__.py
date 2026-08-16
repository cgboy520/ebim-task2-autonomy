"""VLA inference backend skeletons for the EBiM Task 2 autonomy node.

A real deployment loads a Foundation VLA (Pi0.5 / OpenVLA-OFT family) here. The
skeletons below define the contract the :class:`VLAPolicy` expects and ship a
deterministic stub so the autonomy loop runs offline without torch.

To activate a real model on a GPU host, implement
:meth:`VLABackend.propose` against your trained checkpoint (see
``vla_runtime.py``) and ensure the checkpoint + dataset action stats are
downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class VLAInput:
    """Normalized inputs handed to a VLA backend."""

    joints: np.ndarray              # (7,) current joint positions
    gripper_open: float             # current gripper open fraction
    image: np.ndarray | None        # (H,W,3) uint8 RGB of the eval camera
    instruction: str | None         # language task string
    # Full recorded observation, for backends fine-tuned on the benchmark's
    # LeRobot datasets rather than on a 7-joint abstraction. ``state`` is the
    # recorder's 37-D observation.state row and ``images`` is keyed by the
    # recorder's camera names (head / wrist_left / wrist_right / eval_camera),
    # each (H,W,3) uint8 RGB. Both stay None for the 7-joint backends.
    state: np.ndarray | None = None
    images: dict[str, np.ndarray] | None = None


@dataclass
class VLAOutput:
    """A proposed action chunk from the VLA."""

    chunk: list[np.ndarray]         # list of (7,) joint targets / deltas
    is_delta: bool = False
    gripper_seq: list[float] | None = None  # optional per-step gripper cmds
    # Optional per-step base twist (vx, vy, omega) for policies trained on
    # the recorder's whole-body action row; the runner reduces each to a
    # /pedal/state token. None entries (or None overall) command no base.
    base_seq: list[np.ndarray | None] | None = None
    # Optional per-step vertical-spine height (metres, absolute). The demos
    # ramp it 0 -> ~0.486 in the first second and hold.
    spine_seq: list[float] | None = None
    # Optional per-step LEFT-arm joint targets + left gripper open fraction.
    # The official demos command the left arm every episode and its joints
    # are 7 of the 29 kept state dims — it must be executed at deploy.
    left_seq: list[np.ndarray] | None = None
    left_gripper_seq: list[float] | None = None


@dataclass
class VLAConfig:
    """Deployment knobs for the foundation VLA backend (mirrors vla_policy)."""

    backend: str = "stub"
    checkpoint: str | None = None
    action_chunk_size: int = 8
    delta_actions: bool = True
    max_joint_delta: float = 0.15
    active_arm: str = "right"
    instruction: str = (
        "Pick up the thermal pad and place it on the target RAM board."
    )


class VLABackend:
    """Interface a real Foundation-VLA backend implements."""

    backend_name: str = "base"

    def propose(self, inp: VLAInput, *, horizon: int) -> VLAOutput:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class StubBackend(VLABackend):
    """Deterministic hold backend. Used offline and as a safe fallback.

    Returns the current configuration as a (horizon,) chunk of joint targets,
    so the policy loop is exercisable without torch or weights.
    """

    backend_name = "stub"

    def propose(self, inp: VLAInput, *, horizon: int) -> VLAOutput:
        q = np.asarray(inp.joints, dtype=np.float64).copy()
        return VLAOutput(chunk=[q.copy() for _ in range(horizon)], is_delta=False)
