"""GR00T N1.7's torch side: checkpoint -> 46 ONNX graphs, calibration, fixtures.

This is the only groot17 module that imports torch. It runs in an environment
with the `n1.7-release` Isaac-GR00T package importable (torch 2.7.1,
transformers >= 4.57 for Qwen3-VL); `polima.policies.groot17` and `.runtime`
stay importable in the compiler venv and on the board.

## Loading without the gated VLM weights

`Gr00tN1d7.__init__` calls `Qwen3VLForConditionalGeneration.from_pretrained`
on Cosmos-Reason2-2B. Every tensor that call would download is also inside the
GR00T checkpoint, so `load_policy` redirects it to `_from_config` for the
duration of the load and lets the checkpoint fill the module. A load that left
anything at random initialization would be caught by `trace`, which compares
the unrolled pipeline against NVIDIA's own `get_action_with_features`.

## The reference is NVIDIA's code, not ours

`trace` runs the real backbone and records the tensors crossing every boundary
the cut introduces, then re-runs the action head *both* as NVIDIA wrote it and
as the plan unrolls it, and refuses to export if the two disagree. `export_all`
likewise re-runs each image through the exported vision modules and requires
them to reproduce the backbone's own image tokens and deepstack features. So
every structural claim in `runtime.py` -- per-image attention, the deepstack
taps, the token runs, the mask parity -- is tested against the model on every
export, not just once.

## Rewrites that are not cosmetic

* `VisionPatch` is the Conv3d patch embedding as a linear layer. Its kernel
  equals its stride over a pre-flattened patch, so the two are the same
  function; Model Compiler has no 3D convolution.
* `VisionPair` computes attention itself, with real-valued rotary embeddings
  baked for the fixed 16x16 grid. Qwen3-VL's own path splits by `cu_seqlens`
  and applies RoPE through complex-number helpers, neither of which exports.
* `ActionProject` takes a precomputed tau encoding instead of a scalar
  timestep, because the sinusoidal path emits scalar Cast/Expand nodes the
  compiler cannot place -- the same reason as N1.6.
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from polima.policies.groot17 import runtime as rt

#: One representative observation is traced: every graph downstream of the
#: processor sees the same fixed 282-token sequence.
TRACE_FIRST_ONLY = True

DEFAULT_MODEL = "nvidia/GR00T-N1.7-3B"
DEFAULT_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
IMAGE_TOKEN_ID = 151655

#: Tolerances for the export-time structural checks (float32, CPU).
STRUCTURE_ATOL = 1e-3


# ------------------------------------------------------------------ vision


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class VisionPatch(nn.Module):
    """Conv3d(kernel == stride) over flattened patches, as the linear layer it is."""

    def __init__(self, vision, position_embeddings: torch.Tensor):
        super().__init__()
        conv = vision.patch_embed.proj
        out_channels = int(conv.weight.shape[0])
        self.linear = nn.Linear(int(conv.weight[0].numel()), out_channels, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(conv.weight.detach().reshape(out_channels, -1))
            self.linear.bias.copy_(conv.bias.detach())
        self.register_buffer("position", position_embeddings.detach().float().clone().unsqueeze(0))

    def forward(self, patches):
        return self.linear(patches) + self.position


class VisionPair(nn.Module):
    """Two Qwen3-VL ViT blocks over one image's 256 tokens, attention made explicit."""

    def __init__(self, blocks: Sequence[nn.Module], cos: torch.Tensor, sin: torch.Tensor,
                 final_norm: nn.Module | None = None):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        # (seq, head_dim) -> (1, 1, seq, head_dim), broadcasting over heads.
        self.register_buffer("cos", cos.detach().float().clone()[None, None])
        self.register_buffer("sin", sin.detach().float().clone()[None, None])
        # The last pair carries the final merger's pre-shuffle LayerNorm. Inside
        # vit_merger that norm is followed directly by the channel-changing fold,
        # and ModelSDK 2.1 then refuses it for the MLA (a reduction over >= 128
        # channels) and crashes calibrating the APU fallback. Ending this graph,
        # after a residual add, it compiles. Block 23 feeds only the merger.
        self.final_norm = final_norm

    def _attention(self, attn, hidden):
        batch, sequence, width = hidden.shape
        heads = int(attn.num_heads)
        qkv = attn.qkv(hidden).reshape(batch, sequence, 3, heads, width // heads)
        query = qkv[:, :, 0].transpose(1, 2)
        key = qkv[:, :, 1].transpose(1, 2)
        value = qkv[:, :, 2].transpose(1, 2)
        query = query * self.cos + _rotate_half(query) * self.sin
        key = key * self.cos + _rotate_half(key) * self.sin
        weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * attn.scaling, dim=-1)
        output = torch.matmul(weights, value).transpose(1, 2).reshape(batch, sequence, width)
        return attn.proj(output)

    def forward(self, hidden):
        for block in self.blocks:
            hidden = hidden + self._attention(block.attn, block.norm1(hidden))
            hidden = hidden + block.mlp(block.norm2(hidden))
        return hidden if self.final_norm is None else self.final_norm(hidden)


class Merger(nn.Module):
    """Qwen3-VL patch merger: 256x1024 -> 64x4096 -> 64x2048.

    The processor already orders patches so every 2x2 merge window is four
    consecutive tokens, which is what makes the fold a plain reshape. The final
    merger normalizes before folding, the deepstack mergers after. With
    `norm_upstream` the final merger's norm has already run at the end of the
    last ViT pair (see `VisionPair`), so this graph starts at the fold.
    """

    def __init__(self, merger, *, norm_upstream: bool = False):
        super().__init__()
        self.merger = merger
        self.post_shuffle = bool(merger.use_postshuffle_norm)
        if norm_upstream and self.post_shuffle:
            raise ValueError("only a pre-shuffle norm can move to the previous graph")
        self.norm_upstream = norm_upstream

    def forward(self, hidden):
        batch = hidden.shape[0]
        if not self.post_shuffle and not self.norm_upstream:
            hidden = self.merger.norm(hidden)
        hidden = hidden.reshape(batch, rt.MERGED_TOKENS, rt.MERGED_CHANNELS)
        if self.post_shuffle:
            hidden = self.merger.norm(hidden)
        return self.merger.linear_fc2(self.merger.act_fn(self.merger.linear_fc1(hidden)))


# ---------------------------------------------------------------- language


def causal_mask(sequence: int) -> torch.Tensor:
    mask = torch.zeros((1, 1, sequence, sequence), dtype=torch.float32)
    mask.masked_fill_(
        torch.triu(torch.ones((sequence, sequence), dtype=torch.bool), diagonal=1),
        -10000.0,
    )
    return mask


class LlmPair(nn.Module):
    """Two Qwen3-VL text layers with baked M-RoPE, plus any deepstack additions.

    Deepstack adds a visual feature at the image-token positions after each of
    the first three layers. The runtime pre-scatters each feature into a full
    282x2048 tensor that is zero elsewhere, so the addition is a plain `+`.
    """

    def __init__(self, layers: Sequence[nn.Module], start: int, cos: torch.Tensor,
                 sin: torch.Tensor):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.start = start
        self.register_buffer("cos", cos.detach().float().clone())
        self.register_buffer("sin", sin.detach().float().clone())
        self.register_buffer("causal_mask", causal_mask(int(cos.shape[1])))

    def forward(self, hidden, *deepstack):
        taps = iter(deepstack)
        for offset, layer in enumerate(self.layers):
            hidden = layer(hidden, position_embeddings=(self.cos, self.sin),
                           attention_mask=self.causal_mask)
            if self.start + offset < len(rt.DEEPSTACK_BLOCKS):
                hidden = hidden + next(taps)
        return hidden


class VlSelfAttentionPair(nn.Module):
    """Two of the action head's VL self-attention blocks; the first fuses `vlln`."""

    def __init__(self, blocks: Sequence[nn.Module], vlln: nn.Module | None):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.vlln = vlln

    def forward(self, hidden):
        if self.vlln is not None:
            hidden = self.vlln(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


# ------------------------------------------------------------------ action


class StateProject(nn.Module):
    """Normalized state lane -> the DiT's state token, clipping outliers first.

    GR00T's processor clips min/max-normalized state to [-1, 1]
    (`clip_outliers`). The plan's `normalize` opcode is a pure affine map, so the
    clip lives here instead of as a new opcode. Clipping the whole 132-wide lane
    is exact: the padding is zero, and stays zero.
    """

    def __init__(self, head):
        super().__init__()
        self.encoder = head.state_encoder
        self.register_buffer("embodiment", torch.tensor([rt.EMBODIMENT_ID], dtype=torch.long))

    def forward(self, state):
        return self.encoder(torch.clamp(state, -1.0, 1.0), self.embodiment)


class ActionProject(nn.Module):
    """The action lane plus this step's precomputed tau encoding."""

    def __init__(self, head):
        super().__init__()
        self.encoder = head.action_encoder
        self.position_embedding = head.position_embedding
        self.register_buffer("embodiment", torch.tensor([rt.EMBODIMENT_ID], dtype=torch.long))
        self.register_buffer("positions", torch.arange(rt.CHUNK, dtype=torch.long))

    def forward(self, actions, tau_embedding):
        features = self.encoder.W1(actions, self.embodiment)
        features = torch.cat((features, tau_embedding), dim=-1)
        features = self.encoder.W2(features, self.embodiment)
        features = features * torch.sigmoid(features)
        features = self.encoder.W3(features, self.embodiment)
        return features + self.position_embedding(self.positions).unsqueeze(0)


class DiTBlockPair(nn.Module):
    """One cross-attention block then one self-attention block.

    Arguments are in the spec's declared order -- hidden, temb, backbone, mask --
    which is also the order the plan feeds them and the ONNX is exported in.
    """

    def __init__(self, dit, start: int):
        super().__init__()
        if start % 2 or start < 0 or start >= 2 * rt.BLOCK_PAIRS:
            raise ValueError(f"pair must begin at an even block, got {start}")
        self.cross_block = dit.transformer_blocks[start]
        self.self_block = dit.transformer_blocks[start + 1]

    def forward(self, hidden, temb, backbone_features, additive_mask):
        hidden = self.cross_block(hidden, encoder_hidden_states=backbone_features,
                                  encoder_attention_mask=additive_mask, temb=temb)
        return self.self_block(hidden, temb=temb)


class ActionTail(nn.Module):
    def __init__(self, head):
        super().__init__()
        dit = head.model
        self.norm_out = dit.norm_out
        self.proj_out_1 = dit.proj_out_1
        self.proj_out_2 = dit.proj_out_2
        self.decoder = head.action_decoder
        self.register_buffer("embodiment", torch.tensor([rt.EMBODIMENT_ID], dtype=torch.long))

    def forward(self, hidden, temb):
        shift, scale = self.proj_out_1(torch.nn.functional.silu(temb)).chunk(2, dim=1)
        hidden = self.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        decoded = self.decoder(self.proj_out_2(hidden), self.embodiment)
        return decoded[:, -rt.CHUNK:]


# --------------------------------------------------------------- host helpers


def patchify_for_wire(image: np.ndarray) -> np.ndarray:
    """Client-side preprocessor: a 256x256 HWC uint8 frame -> (256, 1536) patches.

    Reproduces Qwen2VLImageProcessor for a single still image: rescale to
    [0, 1], normalize, repeat the frame to the temporal patch size, then order
    patches so each 2x2 merge window is four consecutive rows. Torch-free, so
    the robot client can run it.
    """
    array = np.asarray(image)
    if array.shape != (rt.IMAGE_SIDE, rt.IMAGE_SIDE, 3):
        raise ValueError(
            f"expected a {rt.IMAGE_SIDE}x{rt.IMAGE_SIDE}x3 frame, got {array.shape}"
        )
    pixels = array.astype(np.float32) / 255.0
    pixels = (pixels - np.asarray(rt.IMAGE_MEAN, dtype=np.float32)) / np.asarray(
        rt.IMAGE_STD, dtype=np.float32)
    frames = np.repeat(pixels.transpose(2, 0, 1)[None], rt.TEMPORAL_PATCH, axis=0)
    grid, merge, patch = rt.PATCH_SIDE, rt.MERGE, rt.PATCH
    folded = frames.reshape(1, rt.TEMPORAL_PATCH, 3, grid // merge, merge, patch,
                            grid // merge, merge, patch)
    folded = folded.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(folded.reshape(rt.PATCH_TOKENS, rt.PATCH_CHANNELS))


def additive_masks(image_mask: torch.Tensor, attention_mask: torch.Tensor):
    """(image_additive, text_additive): the DiT's two cross-attention masks."""
    image_valid = image_mask & attention_mask
    text_valid = (~image_mask) & attention_mask
    zero = torch.zeros((), dtype=torch.float32)
    blocked = torch.full((), -10000.0, dtype=torch.float32)
    return (torch.where(image_valid.cpu(), zero, blocked),
            torch.where(text_valid.cpu(), zero, blocked))


def scatter_runs(tokens: np.ndarray, sequence: int = rt.SEQUENCE) -> np.ndarray:
    """(IMAGES * 64, 2048) -> (1, 282, 2048), zero outside the image runs."""
    tokens = np.asarray(tokens, dtype=np.float32).reshape(rt.IMAGES, rt.MERGED_TOKENS, -1)
    full = np.zeros((1, sequence, tokens.shape[-1]), dtype=np.float32)
    for image, (start, length) in enumerate(rt.VISUAL_RUNS):
        full[0, start:start + length] = tokens[image]
    return full


_EXPORT_TAKES_DYNAMO = "dynamo" in inspect.signature(torch.onnx.export).parameters


def _rank4_shape(shape) -> tuple[int, ...]:
    """Prepend unit axes up to rank 4: [N, W, C] -> [N, 1, W, C], [N, C] -> [N, 1, 1, C]."""
    shape = tuple(int(dim) for dim in shape)
    return (1,) * (4 - len(shape)) + shape


class Rank4(nn.Module):
    """Expose a token graph's inputs and output as rank 4. Numerically a no-op.

    ModelSDK 2.1's MPK packager unpacks every tessellated tensor as four
    dimensions and fails on [N, W, C] ("expected 4, got 3"). `promote_rank3_hwc`
    rewrites single-input graphs after export, but the DiT pairs, the
    deepstack-fed language graphs and the projectors take several inputs, so
    the same reshape happens here instead -- the rank-4 form ACT already
    compiles through PoLiMa.
    """

    def __init__(self, module: nn.Module, input_shapes):
        super().__init__()
        self.module = module
        self.input_shapes = [tuple(int(dim) for dim in shape) for shape in input_shapes]

    def forward(self, *inputs):
        restored = [item.reshape(shape) for item, shape in zip(inputs, self.input_shapes)]
        output = self.module(*restored)
        return output.reshape(_rank4_shape(output.shape))


#: Every graph not inside a device-resident chain. Chains stay rank 3 and are
#: promoted by `--promote-rank3-hwc`; these are exported rank 4 directly.
RANK4_GRAPHS = frozenset((
    *rt.deepstack_names(), "vit_merger", *rt.LLM_ENTRY,
    "state_project", "action_project", *rt.block_names(), "action_tail",
))


def _export_onnx(module: nn.Module, inputs: tuple, path: Path, names, outputs, *,
                 rank4: bool = False) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = module.cpu().float().eval()
    # Clone: inputs are usually the previous stage's output, produced under
    # inference_mode, and the ONNX tracer cannot use inference tensors.
    cpu_inputs = tuple(item.detach().cpu().float().clone() for item in inputs)
    if rank4:
        module = Rank4(module, [item.shape for item in cpu_inputs])
        cpu_inputs = tuple(item.reshape(_rank4_shape(item.shape)) for item in cpu_inputs)
    extra = {"dynamo": False} if _EXPORT_TAKES_DYNAMO else {}
    with torch.no_grad():
        expected = module(*cpu_inputs)
        torch.onnx.export(module, cpu_inputs, str(path), input_names=list(names),
                          output_names=list(outputs), opset_version=17,
                          do_constant_folding=True, **extra)
    return expected.detach().cpu().numpy().astype(np.float32)


def _save_calibration(directory: Path, name: str, rank4: bool = False, **tensors) -> None:
    """One array per input, shaped (N, *input_shape) -- rank-4 inputs included."""
    directory.mkdir(parents=True, exist_ok=True)

    def shaped(value):
        array = np.asarray(value, dtype=np.float32)
        return array.reshape((array.shape[0], *_rank4_shape(array.shape[1:]))) if rank4 else array

    np.savez(directory / f"{name}.npz", **{key: shaped(value) for key, value in tensors.items()})


def _max_abs(left, right) -> float:
    left = left.detach().cpu().float().numpy() if torch.is_tensor(left) else np.asarray(left)
    right = right.detach().cpu().float().numpy() if torch.is_tensor(right) else np.asarray(right)
    return float(np.abs(left.astype(np.float32) - right.astype(np.float32)).max())


def _require_close(label: str, left, right, atol: float = STRUCTURE_ATOL) -> float:
    difference = _max_abs(left, right)
    if not np.isfinite(difference) or difference > atol:
        raise ValueError(f"{label}: exported path diverges from the model "
                         f"(max_abs {difference:.3e} > {atol:g})")
    return difference


# ------------------------------------------------------ the driver's contract


@contextlib.contextmanager
def _backbone_from_config():
    """Build Qwen3-VL from its config while a GR00T checkpoint loads."""
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    cls = Qwen3VLForConditionalGeneration
    # `from_pretrained` is normally inherited from PreTrainedModel, so there may
    # be nothing on the class itself to put back -- only the patch to remove.
    own = cls.__dict__.get("from_pretrained")

    def from_config(klass, name, *args, **kwargs):
        config = AutoConfig.from_pretrained(name, trust_remote_code=kwargs.get("trust_remote_code", False))
        return klass._from_config(config, attn_implementation="eager")

    cls.from_pretrained = classmethod(from_config)
    try:
        yield
    finally:
        if own is None:
            del cls.from_pretrained
        else:
            cls.from_pretrained = own


def load_policy(checkpoint: str | Path, lerobot_src: str | Path | None = None):
    """(policy, image_keys) for a GR00T N1.7 checkpoint directory.

    `policy` is Isaac-GR00T's `Gr00tPolicy`, which owns both the model and the
    processor; the export needs the processor to build a faithful observation.
    The model runs in float32 on CPU so the reference and the exported graphs are
    computed in the same precision.
    """
    import gr00t.model  # noqa: F401  registers Gr00tN1d7 with AutoModel
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    with _backbone_from_config():
        policy = Gr00tPolicy(embodiment_tag=rt.EMBODIMENT_TAG, model_path=str(checkpoint),
                             device="cpu", strict=False)
    policy.model.float().eval()
    qwen = policy.model.backbone.model
    for config in (qwen.config, qwen.config.text_config, qwen.config.vision_config):
        config._attn_implementation = "eager"
    image_keys = list(policy.get_modality_config()["video"].modality_keys)
    return policy, image_keys


def build_modules(policy) -> dict[str, Any]:
    model = policy.model
    qwen = model.backbone.model
    return {
        "policy": policy,
        "model": model,
        "head": model.action_head,
        "vision": qwen.model.visual,
        "language": qwen.model.language_model,
    }


def _synthetic_frame(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """A smooth, textured frame: gradients plus noise, not pure noise.

    Calibration statistics from white noise would be unrepresentative of camera
    images; a structured frame is closer while needing no dataset.
    """
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    phase = rng.uniform(0, 2 * np.pi, size=3).astype(np.float32)
    base = 0.5 + 0.35 * np.sin(2 * np.pi * (x * rng.uniform(0.5, 3) + y * rng.uniform(0.5, 3)) + phase)
    noise = rng.normal(0.0, 0.06, size=(height, width, 3)).astype(np.float32)
    return (np.clip(base + noise, 0.0, 1.0) * 255).astype(np.uint8)


def load_samples(policy, checkpoint: str | Path, observation_keys: Sequence[str],
                 dataset_root: str | Path | None = None, count: int = 8,
                 lerobot_src: str | Path | None = None):
    """Deterministic DROID-embodiment observations, processed by GR00T's own pipeline.

    The base checkpoint has no dataset of its own to draw from, so samples are
    synthesized: seeded textured frames, small states, and a fixed instruction.
    The prompt is fixed for a deployed checkpoint, which is what makes the
    282-token sequence static.
    """
    if dataset_root:
        raise NotImplementedError(
            "groot17 calibrates from synthetic DROID observations; drawing from a "
            "LeRobot dataset is not implemented for the base checkpoint"
        )
    from gr00t.data.types import MessageType

    modality = policy.get_modality_config()
    rng = np.random.default_rng(123)
    frames = len(modality["video"].delta_indices)
    samples = []
    for _ in range(count):
        observation = {
            "video": {
                key: np.stack([_synthetic_frame(rng, *SYNTHETIC_CAPTURE)
                               for _ in range(frames)])
                for key in modality["video"].modality_keys
            },
            "state": {
                key: (rng.standard_normal((1, width)) * 0.05).astype(np.float32)
                for key, width in rt.STATE_KEYS
            },
            "language": {
                modality["language"].modality_keys[0]: [INSTRUCTION],
            },
        }
        step = policy._to_vla_step_data(observation)
        processed = policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
        collated = policy.collate_fn([processed])
        raw_state = np.concatenate([observation["state"][key][0] for key, _ in rt.STATE_KEYS])
        samples.append({"collated": collated, "observation": observation,
                        "raw_state": raw_state.astype(np.float32)})
    return samples, None, Path("synthetic-droid")


INSTRUCTION = "pick up the grey eraser and place it in the white basket"

#: DROID's native camera resolution. The processor letterboxes to square before
#: resizing, so any aspect ratio reaches the vision tower as 256x256.
SYNTHETIC_CAPTURE = (180, 320)


def _capture_backbone(model, inputs) -> dict:
    """Run the real backbone, recording every tensor the cut needs."""
    captured: dict = {}
    qwen = model.backbone.model

    def vision_hook(module, args, kwargs, output):
        captured["pixel_values"] = (args[0] if args else kwargs["hidden_states"]).detach().float()
        captured["grid_thw"] = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        captured["image_embeds"] = output[0].detach().float()
        captured["deepstack"] = [feature.detach().float() for feature in output[1]]

    def language_hook(module, args, kwargs):
        captured["inputs_embeds"] = kwargs["inputs_embeds"].detach().float().clone()
        captured["position_ids"] = kwargs["position_ids"].detach().clone()
        captured["visual_pos_masks"] = kwargs["visual_pos_masks"].detach().clone()
        captured["deepstack_visual_embeds"] = [
            feature.detach().float() for feature in kwargs["deepstack_visual_embeds"]]

    hooks = [
        qwen.model.visual.register_forward_hook(vision_hook, with_kwargs=True),
        qwen.model.language_model.register_forward_pre_hook(language_hook, with_kwargs=True),
    ]
    try:
        with torch.inference_mode():
            backbone_inputs, action_inputs = model.prepare_input(inputs)
            output = model.backbone(backbone_inputs)
    finally:
        for hook in hooks:
            hook.remove()
    captured["backbone_inputs"] = backbone_inputs
    captured["action_inputs"] = action_inputs
    captured["backbone_output"] = output
    return captured


def _visual_runs(mask: torch.Tensor) -> tuple[tuple[int, int], ...]:
    positions = torch.nonzero(mask.reshape(-1)).flatten().tolist()
    runs, start = [], positions[0]
    for current, following in zip(positions, [*positions[1:], None]):
        if following != current + 1:
            runs.append((start, current - start + 1))
            start = following
    return tuple(runs)


def trace(modules: dict, sample, image_keys) -> dict:
    """One reference pass through NVIDIA's model, with the cut's boundaries recorded.

    Refuses to continue if any geometry in `runtime.py` has drifted, or if the
    plan's unrolled denoise loop disagrees with `get_action_with_features`.
    """
    model, head = modules["model"], modules["head"]
    captured = _capture_backbone(model, sample["collated"]["inputs"])
    output = captured["backbone_output"]
    action_inputs = captured["action_inputs"]

    # --- geometry, re-checked against the live model -----------------------
    sequence = int(captured["inputs_embeds"].shape[1])
    runs = _visual_runs(captured["visual_pos_masks"])
    grid = captured["grid_thw"].tolist() if captured["grid_thw"] is not None else None
    problems = []
    if sequence != rt.SEQUENCE:
        problems.append(f"sequence {sequence} != {rt.SEQUENCE}")
    if runs != rt.VISUAL_RUNS:
        problems.append(f"visual runs {runs} != {rt.VISUAL_RUNS}")
    if grid != [[1, rt.PATCH_SIDE, rt.PATCH_SIDE]] * rt.IMAGES:
        problems.append(f"grid_thw {grid}")
    if tuple(captured["pixel_values"].shape) != (rt.IMAGES * rt.PATCH_TOKENS, rt.PATCH_CHANNELS):
        problems.append(f"pixel_values {tuple(captured['pixel_values'].shape)}")
    if len(captured["deepstack"]) != len(rt.DEEPSTACK_BLOCKS):
        problems.append(f"{len(captured['deepstack'])} deepstack features")
    if int(action_inputs.embodiment_id.reshape(-1)[0]) != rt.EMBODIMENT_ID:
        problems.append(f"embodiment id {int(action_inputs.embodiment_id.reshape(-1)[0])}")
    if tuple(action_inputs.state.shape) != (1, 1, rt.STATE_LANE):
        problems.append(f"state {tuple(action_inputs.state.shape)}")
    if problems:
        raise ValueError("GR00T N1.7 geometry drifted from runtime.py: " + "; ".join(problems))

    llm_features = output["backbone_features"].detach().float().clone()
    image_mask = output["image_mask"].cpu()
    attention_mask = output["backbone_attention_mask"].cpu()
    with torch.inference_mode():
        processed = head.process_backbone_output(
            type(output)(data={"backbone_features": llm_features.clone(),
                               "image_mask": image_mask,
                               "backbone_attention_mask": attention_mask}))
    backbone_features = processed["backbone_features"].float()

    state = action_inputs.state.float()
    embodiment = torch.tensor([rt.EMBODIMENT_ID], dtype=torch.long)
    image_additive, text_additive = additive_masks(image_mask, attention_mask)

    # --- the plan's unrolled loop ------------------------------------------
    torch.manual_seed(123)
    noise = torch.randn((1, rt.CHUNK, rt.ACTION_LANE), dtype=torch.float32)
    steps = []
    with torch.inference_mode():
        state_features = StateProject(head)(state)
        actions = noise.clone()
        for index, bucket in enumerate(rt.TIMESTEP_BUCKETS):
            timestep = torch.full((1,), bucket, dtype=torch.long)
            temb = head.model.timestep_encoder(timestep).float()
            tau = head.action_encoder.pos_encoding(
                torch.full((1, rt.CHUNK), bucket, dtype=torch.long)).float()
            action_features = ActionProject(head)(actions, tau)
            hidden = torch.cat((state_features, action_features), dim=1)
            pair_inputs = []
            for pair, start in enumerate(range(0, 2 * rt.BLOCK_PAIRS, 2)):
                pair_inputs.append(hidden.detach().cpu().numpy())
                mask = text_additive if pair % 2 == 0 else image_additive
                hidden = DiTBlockPair(head.model, start)(hidden, temb, backbone_features, mask)
            velocity = ActionTail(head)(hidden, temb)
            steps.append({
                "actions_in": actions.detach().cpu().numpy(),
                "temb": temb.detach().cpu().numpy(),
                "tau": tau.detach().cpu().numpy(),
                "pair_inputs": pair_inputs,
                "tail_input": hidden.detach().cpu().numpy(),
                "velocity": velocity.detach().cpu().numpy(),
            })
            actions = actions + np.float32(rt.DT) * velocity

    # --- ... against NVIDIA's own loop, from the same noise ----------------
    with torch.inference_mode():
        state_reference = head.state_encoder(state.view(1, 1, -1), embodiment)
        torch.manual_seed(123)
        reference = head.get_action_with_features(
            backbone_features=backbone_features,
            state_features=state_reference,
            embodiment_id=embodiment,
            backbone_output=processed,
            action_input=type(output)(data={}),
        )["action_pred"].float()
    loop_difference = _require_close("unrolled denoise loop", actions, reference)

    return {
        "captured": captured,
        "prompt_embedding": captured["inputs_embeds"].cpu().numpy(),
        "image_embeds": captured["image_embeds"].cpu().numpy(),
        "deepstack": np.stack([feature.cpu().numpy() for feature in captured["deepstack"]]),
        "llm_features": llm_features.cpu().numpy(),
        "backbone_features": backbone_features.cpu().numpy(),
        "state": state.cpu().numpy(),
        "raw_state": sample["raw_state"],
        "image_additive_mask": image_additive.numpy(),
        "text_additive_mask": text_additive.numpy(),
        "noise": noise.numpy(),
        "final_action": actions.detach().cpu().numpy(),
        "steps": steps,
        "checks": {"denoise_loop_max_abs": loop_difference},
    }


def export_all(build_dir: Path, modules: dict, samples, traces, image_keys) -> list[Path]:
    """Write onnx/ and calibration/. Returns the graphs, in plan order."""
    build_dir = Path(build_dir)
    onnx_dir, calibration = build_dir / "onnx", build_dir / "calibration"
    reference = traces[0]
    written = _export_vision(onnx_dir, calibration, modules, reference)
    written += _export_language(onnx_dir, calibration, modules, reference)
    written += _export_action(onnx_dir, calibration, modules, reference)
    return written


def _single_image_rope(vision) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid = torch.tensor([[1, rt.PATCH_SIDE, rt.PATCH_SIDE]])
    with torch.inference_mode():
        position = vision.fast_pos_embed_interpolate(grid).float()
        rotary = vision.rot_pos_emb(grid).float()
    emb = torch.cat((rotary, rotary), dim=-1)
    return position, emb.cos(), emb.sin()


def _export_vision(onnx_dir: Path, calibration: Path, modules: dict,
                   reference: dict) -> list[Path]:
    vision = modules["vision"].cpu().float()
    pixels = reference["captured"]["pixel_values"].cpu().float()
    per_image = [pixels[index * rt.PATCH_TOKENS:(index + 1) * rt.PATCH_TOKENS][None]
                 for index in range(rt.IMAGES)]
    position, cos, sin = _single_image_rope(vision)
    written: list[Path] = []

    def run_all(module, inputs):
        with torch.inference_mode():
            return [module(item) for item in inputs]

    def emit(name, module, inputs, input_name="hidden", output_name="output"):
        rank4 = name in RANK4_GRAPHS
        _save_calibration(calibration, name, rank4=rank4,
                          **{input_name: torch.stack(inputs).numpy()})
        path = onnx_dir / f"{name}.onnx"
        _export_onnx(module, (inputs[0],), path, (input_name,), (output_name,), rank4=rank4)
        written.append(path)
        return run_all(module, inputs)

    hidden = emit("vit_patch", VisionPatch(vision, position), per_image, "patches")
    deepstack = []
    last_pair = rt.VISION_BLOCKS - 2
    for start, name in zip(range(0, rt.VISION_BLOCKS, 2), rt.vision_block_names()):
        final_norm = vision.merger.norm if start == last_pair else None
        pair = VisionPair(vision.blocks[start:start + 2], cos, sin, final_norm)
        hidden = emit(name, pair, hidden)
        if start + 1 in rt.DEEPSTACK_BLOCKS:
            tap = rt.DEEPSTACK_BLOCKS.index(start + 1)
            merger = Merger(vision.deepstack_merger_list[tap])
            deepstack.append(torch.cat(emit(rt.deepstack_names()[tap], merger, hidden), dim=1))
        gc.collect()
    image_tokens = torch.cat(
        emit("vit_merger", Merger(vision.merger, norm_upstream=True), hidden), dim=1)

    # Per-image attention, the linear patch embedding and the reshape-fold must
    # together reproduce what Qwen3-VL's own vision tower produced.
    checks = reference["checks"]
    checks["image_tokens_max_abs"] = _require_close(
        "vision tower", image_tokens[0], reference["captured"]["image_embeds"])
    for tap, feature in enumerate(deepstack):
        checks[f"deepstack_{tap}_max_abs"] = _require_close(
            f"deepstack tap {tap}", feature[0], reference["captured"]["deepstack"][tap])
    return written


def _export_language(onnx_dir: Path, calibration: Path, modules: dict,
                     reference: dict) -> list[Path]:
    language, head = modules["language"].cpu().float(), modules["head"].cpu().float()
    captured = reference["captured"]
    hidden = captured["inputs_embeds"].cpu().float()
    with torch.inference_mode():
        cos, sin = language.rotary_emb(hidden, captured["position_ids"].cpu())
    deepstack_full = [
        torch.from_numpy(scatter_runs(feature.cpu().numpy()))
        for feature in captured["deepstack_visual_embeds"]
    ]
    written: list[Path] = []

    for start, name in zip(range(0, rt.LLM_LAYERS, 2), rt.llm_names()):
        module = LlmPair(language.layers[start:start + 2], start, cos, sin)
        taps = rt.LLM_ENTRY_DEEPSTACK.get(name, ())
        extra = tuple(deepstack_full[tap] for tap in taps)
        names = ("hidden", *(f"deepstack_{tap}" for tap in taps))
        rank4 = name in RANK4_GRAPHS
        _save_calibration(calibration, name, rank4=rank4, hidden=hidden[None],
                          **{f"deepstack_{tap}": deepstack_full[tap][None] for tap in taps})
        path = onnx_dir / f"{name}.onnx"
        _export_onnx(module, (hidden, *extra), path, names, ("output",), rank4=rank4)
        written.append(path)
        with torch.inference_mode():
            hidden = module(hidden, *extra)
        gc.collect()
    reference["checks"]["llm_max_abs"] = _require_close(
        "language model", hidden, reference["llm_features"])

    blocks = head.vl_self_attention.transformer_blocks
    for start, name in zip(range(0, rt.VL_SELF_ATTENTION_LAYERS, 2),
                           rt.vl_self_attention_names()):
        module = VlSelfAttentionPair(blocks[start:start + 2], head.vlln if start == 0 else None)
        _save_calibration(calibration, name, hidden=hidden[None])
        path = onnx_dir / f"{name}.onnx"
        _export_onnx(module, (hidden,), path, ("hidden",), ("output",))
        written.append(path)
        with torch.inference_mode():
            hidden = module(hidden)
    reference["checks"]["backbone_max_abs"] = _require_close(
        "VL self-attention", hidden, reference["backbone_features"])
    return written


def _export_action(onnx_dir: Path, calibration: Path, modules: dict,
                   reference: dict) -> list[Path]:
    head = modules["head"].cpu().float()
    steps = reference["steps"]
    written: list[Path] = []

    state = torch.from_numpy(reference["state"])
    backbone = torch.from_numpy(reference["backbone_features"])
    masks = {"text": torch.from_numpy(reference["text_additive_mask"]),
             "image": torch.from_numpy(reference["image_additive_mask"])}

    _save_calibration(calibration, "state_project", rank4=True, state=state[None])
    path = onnx_dir / "state_project.onnx"
    _export_onnx(StateProject(head), (state,), path, ("state",), ("state_features",),
                 rank4=True)
    written.append(path)

    _save_calibration(calibration, "action_project", rank4=True,
                      actions=np.stack([step["actions_in"] for step in steps]),
                      tau_embedding=np.stack([step["tau"] for step in steps]))
    path = onnx_dir / "action_project.onnx"
    _export_onnx(ActionProject(head),
                 (torch.from_numpy(steps[0]["actions_in"]), torch.from_numpy(steps[0]["tau"])),
                 path, ("actions", "tau_embedding"), ("action_features",), rank4=True)
    written.append(path)

    for pair, (start, name) in enumerate(zip(range(0, 2 * rt.BLOCK_PAIRS, 2), rt.block_names())):
        mask_name = "text" if pair % 2 == 0 else "image"
        _save_calibration(
            calibration, name, rank4=True,
            hidden=np.stack([step["pair_inputs"][pair] for step in steps]),
            temb=np.stack([step["temb"] for step in steps]),
            backbone_features=np.repeat(reference["backbone_features"][None], rt.DENOISE_STEPS, 0),
            additive_mask=np.repeat(
                reference[f"{mask_name}_additive_mask"][None], rt.DENOISE_STEPS, 0),
        )
        path = onnx_dir / f"{name}.onnx"
        _export_onnx(DiTBlockPair(head.model, start),
                     (torch.from_numpy(steps[0]["pair_inputs"][pair]),
                      torch.from_numpy(steps[0]["temb"]), backbone, masks[mask_name]),
                     path, ("hidden", "temb", "backbone_features", "additive_mask"),
                     ("hidden_out",), rank4=True)
        written.append(path)
        gc.collect()

    _save_calibration(calibration, "action_tail", rank4=True,
                      hidden=np.stack([step["tail_input"] for step in steps]),
                      temb=np.stack([step["temb"] for step in steps]))
    path = onnx_dir / "action_tail.onnx"
    _export_onnx(ActionTail(head),
                 (torch.from_numpy(steps[0]["tail_input"]), torch.from_numpy(steps[0]["temb"])),
                 path, ("hidden", "temb"), ("velocity",), rank4=True)
    written.append(path)
    return written


def write_fixtures(build_dir: Path, sample, reference: dict, image_keys,
                   postprocessor) -> Path:
    """Reference tensors plus every constant the plan reads by name."""
    build_dir = Path(build_dir)
    constants = build_dir / "constants"
    constants.mkdir(parents=True, exist_ok=True)
    steps = reference["steps"]

    # The prompt embedding ships with the image runs zeroed: the runtime packs
    # live vision tokens over exactly those spans.
    prompt = np.array(reference["prompt_embedding"], dtype=np.float32)
    for start, length in rt.VISUAL_RUNS:
        prompt[:, start:start + length] = 0.0

    def write(name: str, values) -> None:
        np.asarray(values, dtype="<f4").ravel().tofile(constants / name)

    write("prompt_embedding", prompt)
    write("image_additive_mask", reference["image_additive_mask"])
    write("text_additive_mask", reference["text_additive_mask"])
    for index, step in enumerate(steps):
        write(f"tau_embedding_{index}", step["tau"])
        write(f"timestep_embedding_{index}", step["temb"])

    pixels = reference["captured"]["pixel_values"].cpu().numpy()
    fixture = build_dir / "groot17_fixture.npz"
    np.savez(
        fixture,
        patches=pixels.reshape(rt.IMAGES, rt.PATCH_TOKENS, rt.PATCH_CHANNELS),
        raw_state=reference["raw_state"],
        state=reference["state"],
        noise=reference["noise"],
        image_embeds=reference["image_embeds"],
        deepstack=reference["deepstack"],
        llm_features=reference["llm_features"],
        backbone_features=reference["backbone_features"],
        final_action=reference["final_action"],
        image_additive_mask=reference["image_additive_mask"],
        text_additive_mask=reference["text_additive_mask"],
        tau_embeddings=np.stack([step["tau"] for step in steps]),
        timestep_embeddings=np.stack([step["temb"] for step in steps]),
    )
    (build_dir / "groot17_export_checks.json").write_text(
        json.dumps(reference["checks"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fixture


def _statistics(checkpoint: str | Path) -> dict:
    path = Path(checkpoint) / "statistics.json"
    if not path.is_file():
        raise FileNotFoundError(f"GR00T N1.7 statistics not found at {path}")
    statistics = json.loads(path.read_text(encoding="utf-8"))
    if rt.EMBODIMENT_TAG not in statistics:
        raise KeyError(f"{path} has no statistics for {rt.EMBODIMENT_TAG}")
    return statistics[rt.EMBODIMENT_TAG]


def percentile_moments(block: dict) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) that make `(x - mean) / std` GR00T's percentile scaling.

    GR00T N1.7 normalizes to [-1, 1] over the q01..q99 range:
    `2 * (x - q01) / (q99 - q01) - 1`, which is `(x - centre) / half_range`.
    A zero range would divide by zero on the board, so it becomes 1.
    """
    low = np.asarray(block["q01"], dtype=np.float32)
    high = np.asarray(block["q99"], dtype=np.float32)
    half_range = (high - low) / 2.0
    return (low + high) / 2.0, np.where(half_range > 0, half_range, 1.0).astype(np.float32)


def write_normalization(checkpoint: str | Path, image_keys, out_path: Path) -> Path:
    """Percentile statistics for state (17) and per-timestep actions (40 x 17).

    Relative-action statistics differ at every step of the horizon, so the
    action statistics are laid out exactly like the gathered action chunk --
    40 rows of 17 -- and `denormalize` reads them elementwise.
    """
    statistics = _statistics(checkpoint)
    state_parts = [percentile_moments(statistics["state"][key]) for key, _ in rt.STATE_KEYS]
    state_mean = np.concatenate([mean for mean, _ in state_parts])
    state_std = np.concatenate([std for _, std in state_parts])

    rows_mean, rows_std = [], []
    for step in range(rt.CHUNK):
        means, stds = [], []
        for key, width in rt.STATE_KEYS:
            relative = statistics.get("relative_action", {}).get(key)
            if relative is not None:
                mean, std = percentile_moments(
                    {name: np.asarray(values)[step] for name, values in relative.items()})
            else:
                mean, std = percentile_moments(statistics["action"][key])
            if mean.shape != (width,):
                raise ValueError(f"{key} statistics have shape {mean.shape}, expected ({width},)")
            means.append(mean)
            stds.append(std)
        rows_mean.append(np.concatenate(means))
        rows_std.append(np.concatenate(stds))
    action_mean = np.stack(rows_mean).astype(np.float32)
    action_std = np.stack(rows_std).astype(np.float32)

    out_path = Path(out_path)
    np.savez(out_path, state_mean=state_mean, state_std=state_std,
             action_mean=action_mean, action_std=action_std)
    constants = out_path.parent / "constants"
    constants.mkdir(parents=True, exist_ok=True)
    for name, values in (("state_mean", state_mean), ("state_std", state_std),
                         ("action_mean", action_mean), ("action_std", action_std)):
        np.asarray(values, dtype="<f4").ravel().tofile(constants / name)
    return out_path


def verify_chain(onnx_dir: Path, fixture_path: Path, report_path: Path, *,
                 atol: float = 1e-2, rtol: float = 1e-2,
                 stage_dir: Path | None = None) -> dict:
    """Replay all 46 graphs under onnxruntime, exactly as the plan does."""
    from collections import OrderedDict

    import onnxruntime as ort

    onnx_dir = Path(onnx_dir)
    fixture = np.load(fixture_path)
    constants = Path(fixture_path).parent / "constants"
    # A bounded cache, not one session per graph: 46 sessions hold a second full
    # copy of the model's weights while the PyTorch reference is still alive in
    # the driver. Eighteen covers one denoise step's graphs without reloading.
    sessions: OrderedDict[str, Any] = OrderedDict()
    cache_limit = 18

    def run(name: str, feeds: dict) -> np.ndarray:
        if name in sessions:
            sessions.move_to_end(name)
        else:
            while len(sessions) >= cache_limit:
                sessions.popitem(last=False)
            sessions[name] = ort.InferenceSession(
                str(onnx_dir / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        # Rank-4 graphs declare [1, 1, W, C]; the plan's buffers are flat, so
        # reshape to whatever each session declares and hand back rank 3.
        declared = {item.name: [int(dim) for dim in item.shape]
                    for item in sessions[name].get_inputs()}
        result = sessions[name].run(None, {
            key: np.asarray(value, dtype=np.float32).reshape(declared[key])
            for key, value in feeds.items()})[0]
        if result.ndim == 4 and result.shape[1] == 1:
            result = result.reshape(result.shape[0], *result.shape[2:])
        if stage_dir is not None:
            Path(stage_dir).mkdir(parents=True, exist_ok=True)
            np.asarray(result, dtype="<f4").ravel().tofile(Path(stage_dir) / f"{name}.f32")
        return result

    def cosine(left, right) -> float:
        left, right = np.ravel(left).astype(np.float64), np.ravel(right).astype(np.float64)
        return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))

    image_tokens, deepstack = [], [[] for _ in rt.DEEPSTACK_BLOCKS]
    for image in range(rt.IMAGES):
        hidden = fixture["patches"][image][None]
        feed = "patches"
        for chain_index, chain in enumerate(rt.VISION_CHAINS):
            for name in chain:
                hidden = run(name, {feed: hidden})
                feed = "hidden"
            if chain_index < len(rt.DEEPSTACK_BLOCKS):
                deepstack[chain_index].append(
                    run(rt.deepstack_names()[chain_index], {"hidden": hidden}))
        image_tokens.append(run("vit_merger", {"hidden": hidden}))
    image_tokens = np.concatenate(image_tokens, axis=1)

    prompt = np.fromfile(constants / "prompt_embedding", dtype="<f4").reshape(
        1, rt.SEQUENCE, rt.LANGUAGE_CHANNELS)
    language = prompt + scatter_runs(image_tokens[0])
    deepstack_full = [scatter_runs(np.concatenate(feature, axis=1)[0]) for feature in deepstack]
    for name in rt.LLM_ENTRY:
        taps = rt.LLM_ENTRY_DEEPSTACK[name]
        language = run(name, {"hidden": language,
                              **{f"deepstack_{tap}": deepstack_full[tap] for tap in taps}})
    for name in rt.LANGUAGE_CHAIN:
        language = run(name, {"hidden": language})
    backbone_features = language

    # The state goes through the plan's own path -- `normalize` then pad, no
    # clip -- and state_project must still reproduce what the processor's
    # clipped normalization produces.
    state_mean = np.fromfile(constants / "state_mean", dtype="<f4")
    state_std = np.fromfile(constants / "state_std", dtype="<f4")
    state_lane = np.zeros((1, 1, rt.STATE_LANE), dtype=np.float32)
    state_lane[0, 0, :rt.STATE_DIM] = (fixture["raw_state"] - state_mean) / state_std
    state_normalization_max_abs = float(
        np.abs(np.clip(state_lane, -1.0, 1.0) - fixture["state"]).max())
    state_features = run("state_project", {"state": state_lane})
    state_features_max_abs = float(np.abs(
        state_features - run("state_project", {"state": fixture["state"]})).max())
    actions = fixture["noise"].astype(np.float32).copy()
    for index in range(rt.DENOISE_STEPS):
        action_features = run("action_project", {"actions": actions,
                                                 "tau_embedding": fixture["tau_embeddings"][index]})
        temb = fixture["timestep_embeddings"][index]
        hidden = np.concatenate((state_features, action_features), axis=1)
        for pair, name in enumerate(rt.block_names()):
            mask = fixture["text_additive_mask" if pair % 2 == 0 else "image_additive_mask"]
            hidden = run(name, {"hidden": hidden, "temb": temb,
                                "backbone_features": backbone_features, "additive_mask": mask})
        velocity = run("action_tail", {"hidden": hidden, "temb": temb})
        actions = actions + np.float32(rt.DT) * velocity

    expected = fixture["final_action"].astype(np.float32)
    difference = np.abs(actions - expected)
    report = {
        "ok": bool(np.allclose(actions, expected, atol=atol, rtol=rtol)
                   and state_normalization_max_abs <= atol
                   and state_features_max_abs <= atol),
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "action_cosine_similarity": cosine(actions, expected),
        "image_token_cosine_similarity": cosine(image_tokens, fixture["image_embeds"]),
        "backbone_cosine_similarity": cosine(backbone_features, fixture["backbone_features"]),
        "state_normalization_max_abs": state_normalization_max_abs,
        "state_features_max_abs": state_features_max_abs,
        "onnx_shape": list(actions.shape),
        "reference_shape": list(expected.shape),
    }
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    return report


def validate_checkpoint(checkpoint: str | Path) -> list[str]:
    """Problems that would only surface deep inside the export otherwise."""
    checkpoint = Path(checkpoint)
    problems: list[str] = []
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        problems.append(f"{checkpoint} has no config.json; is it a GR00T checkpoint?")
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_type") != "Gr00tN1d7":
            problems.append(f"model_type is {config.get('model_type')!r}, expected 'Gr00tN1d7'")
    statistics = checkpoint / "statistics.json"
    if not statistics.is_file():
        problems.append("statistics.json is missing; the board could not denormalize")
    elif rt.EMBODIMENT_TAG not in json.loads(statistics.read_text(encoding="utf-8")):
        problems.append(f"statistics.json has no {rt.EMBODIMENT_TAG} entry")
    return problems
