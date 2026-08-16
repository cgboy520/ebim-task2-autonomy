"""Runtime entry point for VLA backends.

``VLAPolicy.load()`` imports :func:`build_backend` from this module and expects
it to return a :class:`VLABackend` (or None if no real model is available).

Two real backends are sketched (Pi0.5 / OpenVLA-OFT). They are gated behind
lazy imports of ``torch`` / ``transformers`` / ``openpi`` so the module imports
cleanly offline; on a GPU host, install the runtime extras and point
``checkpoint`` at the downloaded weights + dataset action stats.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import StubBackend, VLABackend, VLAConfig


def build_backend(cfg: "VLAConfig | None" = None) -> VLABackend | None:
    """Build the VLA backend named by ``cfg.backend``.

    Returns None when the requested backend's heavy deps or checkpoint are not
    available, so the caller falls back to the deterministic stub. This never
    raises on a missing environment — it logs and returns None.
    """
    if cfg is None:
        cfg = VLAConfig()

    backend = (cfg.backend or "stub").lower()
    ckpt = cfg.checkpoint or os.environ.get("EBIM_VLA_CHECKPOINT")

    if backend in ("stub", "", "none"):
        return StubBackend()

    if backend in ("pi05", "pi0.5", "pi_0.5"):
        return _try_pi05(ckpt)
    if backend in ("openvla", "openvla_oft", "openvla-oft"):
        return _try_openvla(ckpt, cfg)
    if backend in ("lerobot", "smolvla", "act"):
        return _try_lerobot(ckpt, cfg)

    # Unknown backend -> stub with a clear signal.
    return None


def _try_lerobot(ckpt: str | None, cfg) -> VLABackend | None:
    """Load a policy fine-tuned by ``lerobot-train`` on the benchmark's own
    Task 2 recordings.

    The checkpoint directory is what lerobot writes under
    ``outputs/train/<run>/checkpoints/<step>/pretrained_model``: policy
    weights + ``config.json`` + the saved pre/post processor pipelines that
    carry the dataset normalization statistics. The processors are loaded
    from the checkpoint, not rebuilt."""
    try:  # pragma: no cover - requires torch + lerobot + GPU
        import torch
        from lerobot.policies.factory import (
            get_policy_class,
            make_pre_post_processors,
        )

        if not ckpt:
            return None
        path = Path(ckpt)
        if not path.exists():
            return None
        from lerobot.configs.policies import PreTrainedConfig

        policy_cfg = PreTrainedConfig.from_pretrained(str(path))
        policy = get_policy_class(policy_cfg.type).from_pretrained(str(path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy.to(device)
        policy.eval()
        pre, post = make_pre_post_processors(policy_cfg, pretrained_path=str(path))
        return LeRobotBackend(policy, pre, post, device, cfg)
    except Exception:
        return None


def _try_pi05(ckpt: str | None) -> VLABackend | None:
    try:  # pragma: no cover - requires openpi + GPU
        from openpi.models.pi0 import PI0Policy  # type: ignore
        from openpi.serving import PolicyClient

        if not ckpt:
            return None
        # Load norm_stats from the checkpoint's assets/<repo_id> directory.
        # The Pi0.5 flow-matching expert returns already-denormalized actions.
        policy = PI0Policy.load(ckpt)
        return Pi05Backend(policy)
    except Exception:
        return None


def _try_openvla(ckpt: str | None, cfg) -> VLABackend | None:
    try:  # pragma: no cover - requires torch + transformers + GPU
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        if not ckpt:
            return None
        processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        return OpenVLABackend(model, processor, ckpt)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Real backend implementations (only importable with the heavy deps present)
# ---------------------------------------------------------------------------
class Pi05Backend(VLABackend):  # pragma: no cover - requires openpi + GPU
    """Pi0.5 flow-matching VLA. Returns denormalized joint action chunks."""

    backend_name = "pi05"

    def __init__(self, policy) -> None:
        self.policy = policy

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def propose(self, inp, *, horizon):  # noqa: D401
        import numpy as np

        payload = {
            "state": inp.joints.astype(np.float32),
            "images": {"cam_high": inp.image} if inp.image is not None else {},
            "prompt": inp.instruction or "",
        }
        out = self.policy.infer(payload)
        actions = np.asarray(out["actions"])  # (horizon, dim)
        chunk = [actions[i, :7].astype(np.float64) for i in range(min(len(actions), horizon))]
        from . import VLAOutput

        return VLAOutput(chunk=chunk, is_delta=False)


class OpenVLABackend(VLABackend):  # pragma: no cover - requires torch + GPU
    """OpenVLA-OFT backend. Decodes parallel action tokens, denormalizes via
    ``dataset_statistics.json`` beside the checkpoint."""

    backend_name = "openvla_oft"

    def __init__(self, model, processor, ckpt_path: str) -> None:
        import json
        from pathlib import Path

        self.model = model
        self.processor = processor
        stats_path = Path(ckpt_path) / "dataset_statistics.json"
        self.stats = json.loads(stats_path.read_text()) if stats_path.exists() else None

    def reset(self) -> None:
        if hasattr(self.model, "reset"):
            self.model.reset()

    def propose(self, inp, *, horizon):  # noqa: D401
        import numpy as np
        import torch
        from PIL import Image

        pil = Image.fromarray(inp.image) if inp.image is not None else None
        prompt = inp.instruction or "place the thermal pad on the target"
        inputs = self.processor(prompt, pil, inp.joints, unnorm_key=None)
        with torch.no_grad():
            out = self.model.predict_action(**inputs, do_sample=False)
        actions = np.asarray(out)  # (chunk, dim)
        from . import VLAOutput

        chunk = [actions[i, :7].astype(np.float64) for i in range(min(len(actions), horizon))]
        return VLAOutput(chunk=chunk, is_delta=False)


class LeRobotBackend(VLABackend):  # pragma: no cover - requires torch + lerobot
    """LeRobot-trained policy (SmolVLA / ACT) on the Task 2 recordings.

    Bridges two different action spaces. The dataset's action row is the
    recorder's 20-D whole-body vector (base twist, both arms, both grippers,
    spine); this policy drives ONE 7-joint arm plus its gripper, so only the
    active arm's slice is handed back. The unused channels are still fed to
    the model.

    Action layout (services/recording/record_task2.py):
        [0:3] base twist | [3:10] left arm | [10:17] right arm
        [17:19] gripper open fractions (left, right) | [19] spine
    """

    backend_name = "lerobot"

    BASE_TWIST = slice(0, 3)
    LEFT_ARM = slice(3, 10)
    RIGHT_ARM = slice(10, 17)
    LEFT_GRIPPER = 17
    RIGHT_GRIPPER = 18
    SPINE = 19

    # 29-D state convention: 37 -> 29 (drop left_ee 0-6 + left_gripper 29);
    # camera topics resolved in observation.CAMERA_TOPICS.
    # Slots absent from the training data: the policy zero-pads + masks
    # absent keys; never feed real frames.
    EMPTY_OK_CAMS = {"right_wrist_0_rgb"}
    S29_KEEP = [i for i in range(37) if i not in {0, 1, 2, 3, 4, 5, 6, 29}]

    def __init__(self, policy, preprocessor, postprocessor, device: str,
                 cfg: "VLAConfig") -> None:
        self.policy = policy
        self.pre = preprocessor
        self.post = postprocessor
        self.device = device
        self.cfg = cfg
        self.arm = (getattr(cfg, "active_arm", None) or "right").lower()
        self.instruction = (
            getattr(cfg, "instruction", None)
            or "Pick up the thermal pad and place it on the target RAM board."
        )
        self._image_keys = self._discover_image_keys(policy)
        feats = getattr(getattr(policy, "config", None), "input_features",
                        None) or {}
        st_ = feats.get("observation.state")
        st_shape = getattr(st_, "shape", None) or (
            st_.get("shape") if isinstance(st_, dict) else None)
        self._state_dim = int(st_shape[0]) if st_shape else None

    @staticmethod
    def _discover_image_keys(policy) -> list[str]:
        """Camera feature names the checkpoint was trained with, in order.

        The checkpoint's own feature list is the authority — not the
        recorder default."""
        features = getattr(getattr(policy, "config", None), "input_features", None) or {}
        return [k for k in features if k.startswith("observation.images.")]

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def propose(self, inp, *, horizon):  # noqa: D401
        import numpy as np
        import torch

        from . import VLAOutput

        if inp.state is None:
            # No state row -> empty chunk (the runner holds position).
            return VLAOutput(chunk=[], is_delta=False)

        state = np.asarray(inp.state, dtype=np.float32)
        if os.environ.get("EBIM_VLA_DEBUG_IO"):
            import sys
            print("VLA_DBG state37:" if state.shape[-1] == 37 else
                  f"VLA_DBG state{state.shape[-1]}:",
                  np.array2string(state, precision=3, max_line_width=220),
                  file=sys.stderr, flush=True)
        if self._state_dim in (29, 32) and state.shape[-1] == 37:
            # pi05 configs declare the padded dim (max_state_dim 32); the
            # batch carries the dataset's 29-D row
            state = state[self.S29_KEEP]
        elif self._state_dim not in (None, state.shape[-1]):
            raise RuntimeError(
                f"checkpoint wants a {self._state_dim}-D state; recorder "
                f"row is {state.shape[-1]}-D and no slice rule matches")
        obs: dict = {
            "observation.state": torch.from_numpy(
                state
            ).unsqueeze(0).to(self.device),
            "task": [self.instruction],
        }
        images = inp.images or {}
        for key in self._image_keys:
            name = key.removeprefix("observation.images.")
            frame = images.get(name)
            if frame is None:
                if name in self.EMPTY_OK_CAMS:
                    continue        # policy zero-pads + masks this slot
                return VLAOutput(chunk=[], is_delta=False)
            # uint8 HWC -> float CHW in [0,1], batched: lerobot's own
            # preprocess_observation contract.
            t = torch.from_numpy(np.asarray(frame, dtype=np.uint8)).to(self.device)
            obs[key] = (t.permute(2, 0, 1).float() / 255.0).unsqueeze(0)

        with torch.inference_mode():
            batch = self.pre(obs)
            # Ask for the FULL chunk and let the runner pace it open-loop;
            # `horizon` (action_chunk_size) caps how much of it is used.
            if hasattr(self.policy, "predict_action_chunk"):
                actions = self.policy.predict_action_chunk(batch)
                if actions.ndim == 3:
                    # The saved postprocessor (unnormalizer) is only ever fed
                    # (batch, dim) rows on lerobot's own select_action path —
                    # fold the chunk axis into the batch axis so its stats
                    # broadcast per action row, identically to queue pops.
                    actions = actions[0]
                actions = self.post(actions)
            else:
                actions = self.post(self.policy.select_action(batch))
        act = np.asarray(actions.to("cpu").numpy(), dtype=np.float64)
        if os.environ.get("EBIM_VLA_DEBUG_IO"):
            import sys
            a0 = act[0] if act.ndim > 1 else act
            print("VLA_DBG action0:",
                  np.array2string(np.asarray(a0), precision=3,
                                  max_line_width=220),
                  file=sys.stderr, flush=True)
        if act.ndim == 3:
            act = act[0]          # (chunk, dim)
        elif act.ndim == 1:
            act = act[None, :]    # single action -> (1, dim)

        arm_slice = self.RIGHT_ARM if self.arm == "right" else self.LEFT_ARM
        grip_idx = self.RIGHT_GRIPPER if self.arm == "right" else self.LEFT_GRIPPER
        if act.shape[1] <= grip_idx:
            return VLAOutput(chunk=[], is_delta=False)
        n = max(1, min(len(act), int(horizon or 1)))
        joints_seq = [act[i, arm_slice].astype(np.float64) for i in range(n)]
        # Recorded gripper channel is binary {0.0, 1.0}; snap to it.
        grip_seq = [0.0 if float(act[i, grip_idx]) < 0.5 else 1.0
                    for i in range(n)]
        # Demos command the other arm every episode; execute it.
        other_slice = self.LEFT_ARM if self.arm == "right" else self.RIGHT_ARM
        other_grip_idx = (self.LEFT_GRIPPER if self.arm == "right"
                          else self.RIGHT_GRIPPER)
        left_seq = [act[i, other_slice].astype(np.float64) for i in range(n)]
        left_grip_seq = [0.0 if float(act[i, other_grip_idx]) < 0.5 else 1.0
                         for i in range(n)]
        # base action ≡ 0.0 in official fixpos episodes (carry is arm-driven);
        # channel threaded through for mobile-trained checkpoints.
        base_seq = [act[i, self.BASE_TWIST].astype(np.float64) for i in range(n)]
        # The spine is a COMMANDED dof; the demos ramp it 0 -> ~0.486 in
        # the first second.
        spine_seq = (
            [float(act[i, self.SPINE]) for i in range(n)]
            if act.shape[1] > self.SPINE
            else None
        )
        return VLAOutput(chunk=joints_seq, is_delta=False, gripper_seq=grip_seq,
                         base_seq=base_seq, spine_seq=spine_seq,
                         left_seq=left_seq, left_gripper_seq=left_grip_seq)
