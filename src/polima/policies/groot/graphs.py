"""GR00T N1.6's torch side: checkpoint -> 45 ONNX graphs, calibration, fixtures.

Ported from the two scripts that produced the build tree PoLiMa adopts:

    GR00T-N1.6/scripts/export_groot_modalix_eagle.py    the 26-stage Eagle cut
    GR00T-N1.6/scripts/export_groot_modalix_action.py   the 19-graph action cut

This is the only GR00T module that imports torch. It runs in the `groot-n1.6`
conda environment; `polima.policies.groot` and `.runtime` stay importable in the
compiler venv and on the board, which is why the spec names these entry points
as dotted strings rather than holding the callables.

## Why GR00T supplies its own sampler

`polima.export.samples.load` builds calibration observations through
`lerobot.policies.make_pre_post_processors`, which needs a LeRobot policy
object. A GR00T checkpoint is a `transformers` AutoModel with its own
AutoProcessor over a LeRobot *v2.1* dataset, so that path does not apply. The
export driver therefore prefers a graphs module's own `load_samples` when it
defines one; GR00T does, ACT and SmolVLA do not.

## Why the cut lands where it does

Neither half fits the compiler whole. The Eagle backbone is emitted as 26
stages -- patch embedding, 14 SigLIP layer pairs, post-norm, the connector, 8
Qwen layer pairs and a fused output norm -- and the action head as 19: two
projectors, sixteen DiT block pairs, and the decoder tail. Two rewrites in here
are not cosmetic:

* `VisionPostNorm` and `OutputNorm` compute LayerNorm/RMSNorm as matmuls
  against a constant `ones/N` vector rather than calling the reduction
  operators, which Model Compiler cannot place on Modalix.
* `ActionProject` takes a *precomputed* tau encoding instead of a scalar
  timestep, because the sinusoidal path emits scalar Cast/Expand nodes that the
  compiler also rejects. The four encodings are constants of the fixed schedule
  and ship as sidecars.
"""

from __future__ import annotations

import gc
import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from polima.policies.groot import runtime as rt

#: GR00T traces one representative observation rather than the drawn batch:
#: every graph downstream of the backbone sees the same fixed 116-token
#: sequence, so a second trace would only duplicate it.
TRACE_FIRST_ONLY = True

DEFAULT_MODEL = "nvidia/GR00T-N1.6-3B"
DEFAULT_REVISION = "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"

#: The GR1 prompt's image-token id, and the vocabulary it indexes. Both are
#: checked rather than assumed: a checkpoint whose tokenizer moved would
#: otherwise scatter vision features over the wrong sequence positions.
IMAGE_TOKEN_ID = 151669
EMBODIMENT_ID = 20


# ------------------------------------------------------------------ modules


class StateProject(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.encoder = head.state_encoder
        self.register_buffer("embodiment", torch.tensor([EMBODIMENT_ID], dtype=torch.long))

    def forward(self, state):
        return self.encoder(state, self.embodiment)


class ActionProject(nn.Module):
    """The action lane plus this step's tau encoding.

    The four timestep values are fixed by the Euler schedule, so their
    sinusoidal encodings are computed once by the host and arrive as a tensor.
    Computing them here would emit scalar Cast/Expand operators that Model
    Compiler cannot place on Modalix.
    """

    def __init__(self, head):
        super().__init__()
        self.encoder = head.action_encoder
        self.position_embedding = head.position_embedding
        self.register_buffer("embodiment", torch.tensor([EMBODIMENT_ID], dtype=torch.long))

    def forward(self, actions, tau_embedding):
        action_embedding = self.encoder.W1(actions, self.embodiment)
        features = torch.cat((action_embedding, tau_embedding), dim=-1)
        features = self.encoder.W2(features, self.embodiment)
        features = features * torch.sigmoid(features)
        features = self.encoder.W3(features, self.embodiment)
        positions = torch.arange(rt.CHUNK, dtype=torch.long, device=actions.device)
        return features + self.position_embedding(positions).unsqueeze(0)


class DiTBlockPair(nn.Module):
    """One cross-attention block followed by one self-attention block."""

    def __init__(self, dit, start: int):
        super().__init__()
        if start % 2 or start < 0 or start >= 2 * rt.BLOCK_PAIRS:
            raise ValueError(f"pair must begin at an even block, got {start}")
        self.cross_block = dit.transformer_blocks[start]
        self.self_block = dit.transformer_blocks[start + 1]

    def forward(self, hidden, backbone_features, additive_mask, temb):
        hidden = self.cross_block(
            hidden,
            encoder_hidden_states=backbone_features,
            encoder_attention_mask=additive_mask,
            temb=temb,
        )
        return self.self_block(hidden, temb=temb)


class ActionTail(nn.Module):
    def __init__(self, head):
        super().__init__()
        dit = head.model
        self.norm_out = dit.norm_out
        self.proj_out_1 = dit.proj_out_1
        self.proj_out_2 = dit.proj_out_2
        self.decoder = head.action_decoder
        self.register_buffer("embodiment", torch.tensor([EMBODIMENT_ID], dtype=torch.long))

    def forward(self, hidden, temb):
        shift, scale = self.proj_out_1(torch.nn.functional.silu(temb)).chunk(2, dim=1)
        hidden = self.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        hidden = self.proj_out_2(hidden)
        decoded = self.decoder(hidden, self.embodiment)
        return decoded[:, -rt.CHUNK:]


class VisionPatch(nn.Module):
    def __init__(self, embeddings, image_height: int, image_width: int):
        super().__init__()
        self.patch = int(embeddings.patch_size)
        self.projection = embeddings.patch_embedding
        out_channels = self.projection.weight.shape[0]
        with torch.no_grad():
            position = embeddings.position_embedding.weight.reshape(
                embeddings.position_embedding_size,
                embeddings.position_embedding_size,
                out_channels,
            )
            spatial = torch.tensor([[image_height // self.patch, image_width // self.patch]])
            position = embeddings.resize_positional_embeddings(position, spatial)
        self.register_buffer("position", position)

    def forward(self, patches):
        return self.projection(patches) + self.position


class VisionPair(nn.Module):
    def __init__(self, layers, image_patches: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        side = int(image_patches ** 0.5)
        self._meta = [{"img_idx": 0, "patch_hw": (side, side),
                       "win_xy": (0, 0), "win_hw": (side, side)}]
        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden, output_attentions=False, rope_freqs_cis=None,
                           win_meta_list=self._meta, windows_attn=False)[0]
        return hidden


class VisionPostNorm(nn.Module):
    """LayerNorm as matmuls.

    `mean` and `var` reductions do not place on Modalix; a matmul against a
    constant `ones/N` column computes the same statistic in an operator the
    compiler does support.
    """

    def __init__(self, norm):
        super().__init__()
        features = int(norm.weight.numel())
        self.register_buffer("weight", norm.weight.detach().reshape(1, 1, features))
        self.register_buffer("bias", norm.bias.detach().reshape(1, 1, features))
        self.register_buffer("ones", torch.ones((features, 1)) / float(features))
        self.epsilon = float(norm.eps)

    def forward(self, hidden):
        mean = torch.matmul(hidden, self.ones)
        centered = hidden - mean
        variance = torch.matmul(centered * centered, self.ones)
        normalized = centered / torch.sqrt(variance + self.epsilon)
        return normalized * self.weight + self.bias


class VisionConnector(nn.Module):
    def __init__(self, connector):
        super().__init__()
        self.connector = connector

    def forward(self, hidden):
        return self.connector(hidden)


class QwenPair(nn.Module):
    def __init__(self, layers, cos, sin, sequence: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        mask = torch.zeros((1, 1, sequence, sequence), dtype=torch.float32)
        mask.masked_fill_(
            torch.triu(torch.ones((sequence, sequence), dtype=torch.bool), diagonal=1),
            -10000.0,
        )
        self.register_buffer("causal_mask", mask)
        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=self.causal_mask,
                           position_embeddings=(self.cos, self.sin),
                           use_cache=False, output_attentions=False)[0]
        return hidden


class OutputNorm(nn.Module):
    """Qwen's final RMSNorm fused with the action head's input LayerNorm.

    Fusing them saves a whole ELF -- and a round trip -- for two elementwise
    passes over the same 116x2048 tensor. Both reductions are matmuls, for the
    reason VisionPostNorm gives.
    """

    def __init__(self, qwen_norm, action_vlln):
        super().__init__()
        features = int(qwen_norm.weight.numel())
        self.register_buffer("qwen_weight", qwen_norm.weight.detach().reshape(1, 1, features))
        self.register_buffer("action_weight", action_vlln.weight.detach().reshape(1, 1, features))
        self.register_buffer("action_bias", action_vlln.bias.detach().reshape(1, 1, features))
        self.register_buffer("ones", torch.ones((features, 1)) / float(features))
        self.qwen_epsilon = float(qwen_norm.variance_epsilon)
        self.action_epsilon = float(action_vlln.eps)

    def forward(self, hidden):
        rms_variance = torch.matmul(hidden * hidden, self.ones)
        hidden = hidden / torch.sqrt(rms_variance + self.qwen_epsilon)
        hidden = hidden * self.qwen_weight
        mean = torch.matmul(hidden, self.ones)
        centered = hidden - mean
        variance = torch.matmul(centered * centered, self.ones)
        normalized = centered / torch.sqrt(variance + self.action_epsilon)
        return normalized * self.action_weight + self.action_bias


# --------------------------------------------------------------- host helpers


def patchify(pixels: torch.Tensor, patch: int) -> torch.Tensor:
    """(N, 3, H, W) -> (N, H/p * W/p, p*p*3).

    The live client runs the same fold before the wire, which is what keeps an
    image reshape off the board. See `patchify_for_wire`.
    """
    batch, channels, height, width = pixels.shape
    patches = pixels.reshape(batch, channels, height // patch, patch, width // patch, patch)
    return patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, -1, patch * patch * channels)


def patchify_for_wire(image: np.ndarray) -> np.ndarray:
    """Client-side preprocessor: an HWC uint8 frame -> the wire's patch tensor.

    Named by `RobotSpec.image_preprocessor`, so the robot client resolves it
    without importing torch.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an HWC RGB frame, got {array.shape}")
    if array.shape[0] != rt.IMAGE_SIDE or array.shape[1] != rt.IMAGE_SIDE:
        raise ValueError(
            f"expected a {rt.IMAGE_SIDE}x{rt.IMAGE_SIDE} frame, got {array.shape[:2]}"
        )
    grid, patch = rt.PATCH_SIDE, rt.PATCH
    folded = array.reshape(grid, patch, grid, patch, 3).transpose(0, 2, 1, 3, 4)
    return folded.reshape(rt.VISION_WIDTH, rt.PATCH_CHANNELS)


def pixel_unshuffle_tokens(hidden: torch.Tensor, grid: int, factor: int) -> torch.Tensor:
    """(N, grid*grid, C) -> (N, (grid/f)^2, C*f^2), as the runtime opcode does."""
    batch, _, channels = hidden.shape
    folded = hidden.transpose(1, 2).reshape(batch, channels, grid, grid)
    folded = torch.nn.functional.pixel_unshuffle(folded, factor)
    return folded.flatten(2).transpose(1, 2).contiguous()


def additive_masks(image_mask: torch.Tensor, attention_mask: torch.Tensor):
    """The two attention masks the DiT alternates between, as additive biases."""
    image_valid = image_mask & attention_mask
    text_valid = (~image_mask) & attention_mask
    zero = torch.zeros((), dtype=torch.float32, device=image_mask.device)
    blocked = torch.full((), -10000.0, dtype=torch.float32, device=image_mask.device)
    return (torch.where(image_valid, zero, blocked),
            torch.where(text_valid, zero, blocked))


#: `dynamo` only exists from torch 2.5. GR00T exports under 2.7, where the flag
#: is needed -- afe's importer is validated against opset 17 and the legacy
#: tracer, and the dynamo exporter emits a different (valid, unsupported) graph.
#: Older torch has no dynamo path at all, so omitting it there is the same
#: request, and it keeps this module runnable in PoLiMa's own host venv.
_EXPORT_TAKES_DYNAMO = "dynamo" in inspect.signature(torch.onnx.export).parameters


def _export_onnx(module: nn.Module, inputs: tuple, path: Path, names, outputs) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = module.cpu().float().eval()
    cpu_inputs = tuple(item.detach().cpu().float() for item in inputs)
    extra = {"dynamo": False} if _EXPORT_TAKES_DYNAMO else {}
    with torch.inference_mode():
        expected = module(*cpu_inputs)
        torch.onnx.export(module, cpu_inputs, str(path), input_names=list(names),
                          output_names=list(outputs), opset_version=17,
                          do_constant_folding=True, **extra)
    return expected.detach().cpu().numpy().astype(np.float32)


def _save_calibration(directory: Path, name: str, **tensors) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"{name}.npz",
             **{key: np.asarray(value, dtype=np.float32) for key, value in tensors.items()})


# ------------------------------------------------------ the driver's contract


def load_policy(checkpoint: str | Path, lerobot_src: str | Path | None = None):
    """(model, image_keys) for a GR00T checkpoint.

    `lerobot_src` is accepted and ignored: GR00T loads through `transformers`,
    not through a LeRobot source tree. The signature matches ACT's and
    SmolVLA's so `polima.export.driver` needs no branch.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(checkpoint)).eval()
    return model, ["observation.images.overhead", "observation.images.wrist"]


def load_samples(policy, checkpoint: str | Path, observation_keys: Sequence[str],
                 dataset_root: str | Path | None = None, count: int = 8,
                 lerobot_src: str | Path | None = None):
    """GR00T's own calibration draw, in place of `polima.export.samples.load`.

    Returns the same `(samples, postprocessor, root)` triple the generic path
    does. The postprocessor is None: GR00T denormalizes on the board from the
    statistics `write_normalization` extracts, so nothing downstream needs one.
    """
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from transformers import AutoProcessor

    root = Path(dataset_root) if dataset_root else _dataset_from_checkpoint(checkpoint)
    processor = AutoProcessor.from_pretrained(str(checkpoint))
    processor.eval()
    tag = EmbodimentTag.GR1
    modalities = processor.get_modality_configs()[tag.value]
    dataset = LeRobotEpisodeLoader(dataset_path=root, modality_configs=modalities,
                                   video_backend="torchcodec")
    samples = []
    # Even spacing, for the reason polima.export.samples gives: episodes are
    # recorded in order, so the first N frames are one moment of one episode.
    indices = np.linspace(0, len(dataset) - 1, num=min(count, len(dataset)), dtype=int)
    for index in indices:
        trajectory = dataset[int(index)]
        step = extract_step_data(trajectory, 0, modality_configs=modalities,
                                 embodiment_tag=tag)
        message = {"type": MessageType.EPISODE_STEP.value, "content": step}
        samples.append(processor.collator([processor([message])])["inputs"])
    return samples, None, root


def _dataset_from_checkpoint(checkpoint: str | Path) -> Path:
    """The dataset a GR00T run trained on, from its own experiment config."""
    for name in ("experiment_cfg/metadata.json", "train_config.json", "config.json"):
        path = Path(checkpoint) / name
        if not path.is_file():
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        for key in ("dataset_path", "dataset_root"):
            if config.get(key):
                return Path(config[key]).resolve()
        dataset = config.get("dataset") or {}
        if dataset.get("root"):
            return Path(dataset["root"]).resolve()
    raise FileNotFoundError(
        f"cannot infer the dataset for {checkpoint}; pass --dataset-root explicitly"
    )


def build_modules(policy) -> dict[str, Any]:
    """Split the checkpoint into the pieces every later stage indexes by name."""
    backbone = policy.backbone.model
    return {
        "model": policy,
        "head": policy.action_head,
        "eagle": backbone,
        "vision": backbone.vision_model.vision_model,
        "language": backbone.language_model.model,
        "connector": backbone.mlp1,
        "vlln": policy.action_head.vlln,
    }


def trace(modules: dict, sample, image_keys) -> dict:
    """One reference pass, recording every tensor a later stage needs.

    The four denoise steps are traced together because each graph's calibration
    set is exactly those four activations -- tau is bucketed at 0/250/500/750
    and the pipeline never sees anything else.
    """
    model, head = modules["model"], modules["head"]
    with torch.inference_mode():
        backbone_inputs, action_inputs = model.prepare_input(sample)
        backbone = model.backbone(backbone_inputs)

    sequence = int(backbone_inputs["input_ids"].shape[1])
    if sequence != rt.SEQUENCE:
        raise ValueError(f"expected a {rt.SEQUENCE}-token GR1 sequence, got {sequence}")

    backbone_features = head.vlln(backbone.backbone_features).float()
    state = action_inputs.state.float()
    embodiment = torch.tensor([EMBODIMENT_ID], dtype=torch.long)
    state_features = head.state_encoder(state, embodiment)
    generator = torch.Generator().manual_seed(123)
    actions = torch.randn((1, rt.CHUNK, rt.ACTION_LANE), generator=generator,
                          dtype=torch.float32)
    image_additive, text_additive = additive_masks(
        backbone.image_mask, backbone.backbone_attention_mask
    )

    steps = []
    with torch.inference_mode():
        for index in range(rt.DENOISE_STEPS):
            timestep = torch.tensor([rt.TIMESTEP_BUCKETS[index]], dtype=torch.float32)
            action_features = head.action_encoder(actions, timestep.long(), embodiment)
            positions = torch.arange(rt.CHUNK)
            action_features = action_features + head.position_embedding(positions).unsqueeze(0)
            hidden = torch.cat((state_features, action_features), dim=1)
            temb = head.model.timestep_encoder(timestep.long())
            pair_inputs = []
            for start in range(0, 2 * rt.BLOCK_PAIRS, 2):
                mask = text_additive if start % 4 == 0 else image_additive
                pair_inputs.append(hidden.detach().cpu().numpy())
                hidden = DiTBlockPair(head.model, start)(
                    hidden, backbone_features, mask, temb)
            velocity = ActionTail(head)(hidden, temb)
            actions = actions + np.float32(rt.DT) * velocity
            steps.append({
                "timestep": timestep.detach().cpu().numpy(),
                "temb": temb.detach().cpu().numpy(),
                "tau": head.action_encoder.pos_encoding(
                    torch.full((1, rt.CHUNK), float(rt.TIMESTEP_BUCKETS[index]))
                ).detach().cpu().numpy(),
                "pair_inputs": pair_inputs,
                "tail_input": hidden.detach().cpu().numpy(),
                "velocity": velocity.detach().cpu().numpy(),
                "actions": actions.detach().cpu().numpy(),
            })

    # The prompt's token embeddings, recorded here rather than during the Eagle
    # export so that `write_fixtures` does not need the language model again.
    input_ids = backbone_inputs["input_ids"].cpu()
    with torch.inference_mode():
        prompt_embedding = modules["language"].embed_tokens(input_ids).float()

    return {
        "backbone_inputs": backbone_inputs,
        "prompt_embedding": prompt_embedding.detach().cpu().numpy(),
        "state": state.detach().cpu().numpy(),
        "backbone_features": backbone_features.detach().cpu().numpy(),
        "image_additive_mask": image_additive.detach().cpu().numpy(),
        "text_additive_mask": text_additive.detach().cpu().numpy(),
        # The seed the first step started from, recovered by undoing its update.
        "noise": steps[0]["actions"] - np.float32(rt.DT) * steps[0]["velocity"],
        "final_action": steps[-1]["actions"],
        "steps": steps,
    }


def export_all(build_dir: Path, modules: dict, samples, traces, image_keys) -> list[Path]:
    """Write onnx/ and calibration/. Returns the graphs, in plan order."""
    build_dir = Path(build_dir)
    onnx_dir, calibration = build_dir / "onnx", build_dir / "calibration"
    reference = traces[0]
    written = _export_eagle(onnx_dir, calibration, modules, reference)
    written += _export_action(onnx_dir, calibration, modules, reference)
    return written


def _export_eagle(onnx_dir: Path, calibration: Path, modules: dict,
                  reference: dict) -> list[Path]:
    vision = modules["vision"]
    language = modules["language"]
    backbone_inputs = reference["backbone_inputs"]

    pixels = backbone_inputs["pixel_values"][0].cpu().float()
    input_ids = backbone_inputs["input_ids"].cpu()
    image_positions = input_ids == IMAGE_TOKEN_ID
    if int(image_positions.sum()) != rt.CONNECTOR_WIDTH:
        raise ValueError(
            f"expected {rt.CONNECTOR_WIDTH} image tokens, got {int(image_positions.sum())}"
        )
    written: list[Path] = []

    patch_module = VisionPatch(vision.embeddings, int(pixels.shape[-2]), int(pixels.shape[-1]))
    patches = patchify(pixels, patch_module.patch)
    _save_calibration(calibration, "eagle_vision_patch", patches=patches[None])
    path = onnx_dir / "eagle_vision_patch.onnx"
    hidden = torch.from_numpy(_export_onnx(patch_module, (patches,), path,
                                           ("patches",), ("output",)))
    written.append(path)

    layers = list(vision.encoder.layers)
    for start in range(0, len(layers), 2):
        end = min(start + 2, len(layers))
        name = f"eagle_vision_{start:02d}_{end - 1:02d}"
        _save_calibration(calibration, name, hidden=hidden[None])
        path = onnx_dir / f"{name}.onnx"
        hidden = torch.from_numpy(_export_onnx(
            VisionPair(layers[start:end], rt.VISION_WIDTH), (hidden,), path,
            ("hidden",), ("output",)))
        written.append(path)

    _save_calibration(calibration, "eagle_vision_post_norm", hidden=hidden[None])
    path = onnx_dir / "eagle_vision_post_norm.onnx"
    hidden = torch.from_numpy(_export_onnx(
        VisionPostNorm(vision.post_layernorm), (hidden,), path, ("hidden",), ("output",)))
    written.append(path)

    folded = pixel_unshuffle_tokens(hidden, rt.PATCH_SIDE, rt.UNSHUFFLE)
    _save_calibration(calibration, "eagle_vision_connector", hidden=folded[None])
    path = onnx_dir / "eagle_vision_connector.onnx"
    image_tokens = torch.from_numpy(_export_onnx(
        VisionConnector(modules["connector"]), (folded,), path, ("hidden",), ("output",)))
    written.append(path)

    # The prompt's own embeddings, with the image window overwritten. Only the
    # non-image half ships as a constant; the runtime packs the rest.
    token_embeddings = language.embed_tokens(input_ids).float()
    token_embeddings[image_positions] = image_tokens.reshape(-1, image_tokens.shape[-1])
    positions = torch.arange(rt.SEQUENCE, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        cos, sin = language.rotary_emb(token_embeddings, positions)

    hidden = token_embeddings
    layers = list(language.layers)
    for start in range(0, len(layers), 2):
        end = min(start + 2, len(layers))
        name = f"eagle_qwen_{start:02d}_{end - 1:02d}"
        _save_calibration(calibration, name, hidden=hidden[None])
        path = onnx_dir / f"{name}.onnx"
        hidden = torch.from_numpy(_export_onnx(
            QwenPair(layers[start:end], cos, sin, rt.SEQUENCE), (hidden,), path,
            ("hidden",), ("output",)))
        written.append(path)

    _save_calibration(calibration, "eagle_output_norm", hidden=hidden[None])
    path = onnx_dir / "eagle_output_norm.onnx"
    _export_onnx(OutputNorm(language.norm, modules["vlln"]), (hidden,), path,
                 ("hidden",), ("output",))
    written.append(path)
    return written


def _export_action(onnx_dir: Path, calibration: Path, modules: dict,
                   reference: dict) -> list[Path]:
    head = modules["head"].cpu().float()
    steps = reference["steps"]
    written: list[Path] = []

    state = torch.from_numpy(reference["state"])
    noise = torch.from_numpy(reference["noise"])
    tau = torch.from_numpy(steps[0]["tau"])
    backbone_features = torch.from_numpy(reference["backbone_features"])
    text_mask = torch.from_numpy(reference["text_additive_mask"])
    image_mask = torch.from_numpy(reference["image_additive_mask"])

    _save_calibration(calibration, "state_project", state=state[None])
    path = onnx_dir / "state_project.onnx"
    _export_onnx(StateProject(head), (state,), path, ("state",), ("state_features",))
    written.append(path)

    _save_calibration(
        calibration, "action_project",
        actions=np.stack([reference["noise"], *(s["actions"] for s in steps[:-1])], axis=0),
        tau_embedding=np.stack([s["tau"] for s in steps], axis=0),
    )
    path = onnx_dir / "action_project.onnx"
    _export_onnx(ActionProject(head), (noise, tau), path,
                 ("actions", "tau_embedding"), ("action_features",))
    written.append(path)

    sample_state = StateProject(head)(state)
    sample_action = ActionProject(head)(noise, tau)
    hidden = torch.cat((sample_state, sample_action), dim=1)
    temb = torch.from_numpy(steps[0]["temb"])
    for pair, start in enumerate(range(0, 2 * rt.BLOCK_PAIRS, 2)):
        mask = text_mask if pair % 2 == 0 else image_mask
        name = f"dit_blocks_{start:02d}_{start + 1:02d}"
        _save_calibration(
            calibration, name,
            hidden=np.stack([s["pair_inputs"][pair] for s in steps], axis=0),
            backbone_features=np.repeat(
                reference["backbone_features"][None], rt.DENOISE_STEPS, axis=0),
            additive_mask=np.repeat(
                reference["text_additive_mask" if pair % 2 == 0 else "image_additive_mask"][None],
                rt.DENOISE_STEPS, axis=0),
            temb=np.stack([s["temb"] for s in steps], axis=0),
        )
        module = DiTBlockPair(head.model, start)
        path = onnx_dir / f"{name}.onnx"
        _export_onnx(module, (hidden, backbone_features, mask, temb), path,
                     ("hidden", "backbone_features", "additive_mask", "temb"),
                     ("hidden_out",))
        written.append(path)
        with torch.inference_mode():
            hidden = module(hidden, backbone_features, mask, temb)
        gc.collect()

    _save_calibration(
        calibration, "action_tail",
        hidden=np.stack([s["tail_input"] for s in steps], axis=0),
        temb=np.stack([s["temb"] for s in steps], axis=0),
    )
    path = onnx_dir / "action_tail.onnx"
    _export_onnx(ActionTail(head), (hidden, temb), path, ("hidden", "temb"), ("velocity",))
    written.append(path)
    return written


def write_fixtures(build_dir: Path, sample, reference: dict, image_keys,
                   postprocessor) -> Path:
    """Reference tensors plus every constant the plan reads by name.

    The sidecars are written under the names `runtime.SIDECARS` lists, so bundle
    packing copies them into constants/ without a policy-specific rename.
    """
    build_dir = Path(build_dir)
    constants = build_dir / "constants"
    constants.mkdir(parents=True, exist_ok=True)
    steps = reference["steps"]

    backbone_inputs = reference["backbone_inputs"]
    input_ids = backbone_inputs["input_ids"].cpu()
    image_positions = (input_ids == IMAGE_TOKEN_ID)[0].numpy()

    # The prompt embedding ships with the image window zeroed: the runtime packs
    # the live vision tokens over exactly that span, so anything left there
    # would be dead weight in the bundle and a silent trap if the span moved.
    prompt = np.array(reference["prompt_embedding"], dtype=np.float32)
    prompt[:, image_positions] = 0.0
    start = int(np.flatnonzero(image_positions)[0])
    if start != rt.IMAGE_TOKEN_START or not np.all(np.diff(np.flatnonzero(image_positions)) == 1):
        raise ValueError(
            f"image tokens must be contiguous at {rt.IMAGE_TOKEN_START}, found {start}"
        )

    def write(name: str, values) -> None:
        np.asarray(values, dtype="<f4").ravel().tofile(constants / name)

    write("prompt_embedding", prompt)
    write("image_additive_mask", reference["image_additive_mask"])
    write("text_additive_mask", reference["text_additive_mask"])
    for index, step in enumerate(steps):
        write(f"tau_embedding_{index}", step["tau"])
        write(f"timestep_embedding_{index}", step["temb"])

    fixture = build_dir / "groot_fixture.npz"
    np.savez(
        fixture,
        patches=patchify(backbone_inputs["pixel_values"][0].cpu().float(), rt.PATCH).numpy(),
        state=reference["state"],
        noise=reference["noise"],
        backbone_features=reference["backbone_features"],
        final_action=reference["final_action"],
        image_additive_mask=reference["image_additive_mask"],
        text_additive_mask=reference["text_additive_mask"],
        tau_embeddings=np.stack([s["tau"] for s in steps], axis=0),
        timestep_embeddings=np.stack([s["temb"] for s in steps], axis=0),
    )
    return fixture


def write_normalization(checkpoint: str | Path, image_keys, out_path: Path) -> Path:
    """state/action mean and std for the 6 real joints, from GR00T's metadata.

    GR00T keeps per-embodiment statistics in `experiment_cfg/metadata.json`
    rather than in the checkpoint's weights, and stores them over the padded
    lane; only the leading `state_dim` entries describe real joints.
    """
    path = Path(checkpoint) / "experiment_cfg" / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"GR00T normalization metadata not found at {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    embodiments = metadata.get("embodiments") or metadata
    # A fine-tuned checkpoint carries exactly one embodiment; naming it here
    # would hardcode NEW_EMBODIMENT and break the GR1 base build.
    if len(embodiments) != 1:
        raise ValueError(f"expected one embodiment in {path}, found {list(embodiments)}")
    statistics = next(iter(embodiments.values()))["statistics"]

    def moments(section: str) -> tuple[np.ndarray, np.ndarray]:
        block = statistics[section]
        mean = np.asarray(block["mean"], dtype=np.float32).ravel()[:rt.STATE_DIM]
        std = np.asarray(block["std"], dtype=np.float32).ravel()[:rt.STATE_DIM]
        # A zero deviation on a joint that never moved would divide by zero on
        # the board, where there is no traceback to read.
        return mean, np.where(std > 0, std, 1.0).astype(np.float32)

    state_mean, state_std = moments("state")
    action_mean, action_std = moments("action")
    out_path = Path(out_path)
    np.savez(out_path, state_mean=state_mean, state_std=state_std,
             action_mean=action_mean, action_std=action_std)
    constants = out_path.parent / "constants"
    constants.mkdir(parents=True, exist_ok=True)
    for name, values in (("state_mean", state_mean), ("state_std", state_std),
                         ("action_mean", action_mean), ("action_std", action_std)):
        values.astype("<f4").tofile(constants / name)
    return out_path


def verify_chain(onnx_dir: Path, fixture_path: Path, report_path: Path, *,
                 atol: float = 1e-2, rtol: float = 1e-2,
                 stage_dir: Path | None = None) -> dict:
    """Replay all 45 graphs under onnxruntime and compare against the trace.

    Run before anything is quantized, so a mismatch on the board afterwards is
    the compiler's or the runtime's, not the cut's.
    """
    import onnxruntime as ort

    onnx_dir = Path(onnx_dir)
    fixture = np.load(fixture_path)
    sessions: dict[str, Any] = {}

    def run(name: str, feeds: dict) -> np.ndarray:
        if name not in sessions:
            sessions[name] = ort.InferenceSession(
                str(onnx_dir / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        result = sessions[name].run(None, feeds)[0]
        if stage_dir is not None:
            Path(stage_dir).mkdir(parents=True, exist_ok=True)
            np.asarray(result, dtype="<f4").ravel().tofile(Path(stage_dir) / f"{name}.f32")
        return result

    hidden = run("eagle_vision_patch", {"patches": fixture["patches"].astype(np.float32)})
    for name in rt.vision_stage_names():
        hidden = run(name, {"hidden": hidden})
    hidden = run("eagle_vision_post_norm", {"hidden": hidden})
    folded = pixel_unshuffle_tokens(torch.from_numpy(hidden), rt.PATCH_SIDE, rt.UNSHUFFLE)
    image_tokens = run("eagle_vision_connector", {"hidden": folded.numpy()})

    # The prompt half of the sequence is a constant; the fixture's recorded
    # backbone input carries it, with the image window replaced here.
    language = np.zeros((1, rt.SEQUENCE, rt.LANGUAGE_CHANNELS), dtype=np.float32)
    prompt = np.fromfile(Path(fixture_path).parent / "constants" / "prompt_embedding",
                         dtype="<f4").reshape(1, rt.SEQUENCE, rt.LANGUAGE_CHANNELS)
    language += prompt
    window = slice(rt.IMAGE_TOKEN_START, rt.IMAGE_TOKEN_START + rt.CONNECTOR_WIDTH)
    language[:, window] = image_tokens
    for name in rt.qwen_stage_names():
        language = run(name, {"hidden": language})
    backbone_features = run("eagle_output_norm", {"hidden": language})

    eagle_reference = fixture["backbone_features"].astype(np.float32)
    eagle_cosine = float(
        np.dot(backbone_features.ravel(), eagle_reference.ravel())
        / (np.linalg.norm(backbone_features.ravel()) * np.linalg.norm(eagle_reference.ravel()))
    )

    state_features = run("state_project", {"state": fixture["state"].astype(np.float32)})
    actions = fixture["noise"].astype(np.float32).copy()
    per_step = []
    for index in range(rt.DENOISE_STEPS):
        action_features = run("action_project", {
            "actions": actions,
            "tau_embedding": fixture["tau_embeddings"][index].astype(np.float32),
        })
        temb = fixture["timestep_embeddings"][index].astype(np.float32)
        hidden = np.concatenate((state_features, action_features), axis=1)
        for pair, name in enumerate(rt.block_names()):
            mask = fixture["text_additive_mask" if pair % 2 == 0 else "image_additive_mask"]
            hidden = run(name, {
                "hidden": hidden,
                "backbone_features": backbone_features,
                "additive_mask": mask.astype(np.float32),
                "temb": temb,
            })
        velocity = run("action_tail", {"hidden": hidden, "temb": temb})
        actions = actions + np.float32(rt.DT) * velocity
        per_step.append({"step": index, "velocity_l2": float(np.linalg.norm(velocity))})

    expected = fixture["final_action"].astype(np.float32)
    difference = np.abs(actions - expected)
    report = {
        "ok": bool(np.allclose(actions, expected, atol=atol, rtol=rtol)),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "eagle_cosine_similarity": eagle_cosine,
        "onnx_shape": list(actions.shape),
        "reference_shape": list(expected.shape),
        "steps": per_step,
    }
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    return report


def validate_checkpoint(checkpoint: str | Path) -> list[str]:
    """Problems that would only surface deep inside the export otherwise."""
    checkpoint = Path(checkpoint)
    problems: list[str] = []
    if not (checkpoint / "config.json").is_file():
        problems.append(f"{checkpoint} has no config.json; is it a GR00T checkpoint?")
    if not (checkpoint / "experiment_cfg" / "metadata.json").is_file():
        problems.append(
            "experiment_cfg/metadata.json is missing; normalization statistics "
            "cannot be recovered and the board would denormalize with the wrong scale"
        )
    return problems
