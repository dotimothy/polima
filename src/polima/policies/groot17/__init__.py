"""GR00T N1.7 -- Cosmos-Reason2 (Qwen3-VL) VLM + flow-matching DiT.

Ported from the `n1.7-release` tag of Isaac-GR00T and the public
`nvidia/GR00T-N1.7-3B` base checkpoint. The geometry in `runtime.py` is
measured from a traced inference, and `graphs.trace` re-checks it at every
export.

## How it differs from GR00T N1.6

N1.6 cut into 45 graphs around an Eagle backbone; N1.7 cuts into 46 around a
Qwen3-VL one:

    vision    17   patch embed, 12 ViT block pairs, 3 deepstack mergers, merger
    language  10   8 LLM layer pairs, 2 VL self-attention pairs
    action    19   state projector, action projector, 16 DiT pairs, tail

The action head is architecturally the same kind of flow-matching DiT, but at
N1.7's dimensions: a 40-step horizon, 132-wide state and action lanes, and a
282-token backbone sequence to cross-attend over.

## The base checkpoint's backbone is built from config

`Gr00tN1d7.__init__` constructs its VLM with
`Qwen3VLForConditionalGeneration.from_pretrained("nvidia/Cosmos-Reason2-2B")`.
The GR00T checkpoint already contains every backbone tensor, so `graphs` builds
that module from Cosmos's *config* instead and lets the GR00T weights fill it --
verified with zero missing and zero unexpected keys. Only Cosmos's config,
tokenizer and processor files are fetched; that repository is gated, so the
Hugging Face account doing the export must have accepted its license.

## Embodiment

The base weights ship no SO-101 embodiment (that needs a fine-tune under
`NEW_EMBODIMENT`). This spec compiles the base model under
`oxe_droid_relative_eef_relative_joint`: two cameras and one arm, the closest
pretrain embodiment to the SO-101 setup, and the one whose geometry a
fine-tuned checkpoint would most likely share.
"""

from __future__ import annotations

from polima.policies.base import (
    CalibrationSource,
    CompilePlan,
    DatasetContract,
    GraphSpec,
    PolicySpec,
    RobotSpec,
    TensorSpec,
    TrainSpec,
    WireSpec,
)
from polima.policies.groot17.runtime import (
    ACTION_DIM,
    ACTION_LANE,
    CHUNK,
    DEFAULT_PORT,
    DENOISE_STEPS,
    HIDDEN_CHANNELS,
    HIDDEN_WIDTH,
    IMAGES,
    LANGUAGE_CHANNELS,
    LLM_ENTRY_DEEPSTACK,
    MASK_ELEMENTS,
    MERGED_TOKENS,
    PATCH_CHANNELS,
    PATCH_TOKENS,
    SEQUENCE,
    STATE_DIM,
    STATE_KEYS,
    STATE_LANE,
    TEMB_ELEMENTS,
    VISION_CHANNELS,
    WIRE_MAGIC,
    block_names,
    deepstack_names,
    llm_names,
    vision_block_names,
    vl_self_attention_names,
)
from polima.policies.registry import register_policy

#: DROID's 17 state and action dimensions, in modality order.
STATE_NAMES = tuple(
    f"{key}.{index}" if width > 1 else key
    for key, width in STATE_KEYS
    for index in range(width)
)

#: DROID's native camera resolution. The client letterboxes to square and
#: resizes to 256 before patchifying, so this only describes the recording.
CAPTURE_HEIGHT, CAPTURE_WIDTH = 180, 320


def _token_tensor(name: str, width: int, channels: int) -> TensorSpec:
    return TensorSpec(name, (1, width, channels))


def _rank4_tensor(name: str, width: int, channels: int) -> TensorSpec:
    """[1, 1, W, C]: ModelSDK 2.1's MPK packager requires rank 4 for unchained graphs."""
    return TensorSpec(name, (1, 1, width, channels))


def _tessellated(name: str, width: int, channels: int) -> TensorSpec:
    """An MLA output the host detessellates from HWC16, rank 4 for the packager."""
    return TensorSpec(name, (1, 1, width, channels), dram_layout="hwc16",
                      logical_width=width, logical_channels=channels)


def _chained_stage(name: str, width: int, in_channels: int, out_channels: int,
                   *, samples: int) -> GraphSpec:
    """One device-resident stage.

    Chained stages expose a flat HWC boundary at both ends so one ELF's output
    buffer binds directly to the next ELF's input; `promote_rank3_hwc` makes
    that legal for [N, W, C] token graphs under ModelSDK 2.1.
    """
    return GraphSpec(
        name=name,
        builder=f"polima.policies.groot17.graphs:{name}",
        inputs=(_token_tensor("hidden", width, in_channels),),
        outputs=(_token_tensor("output", width, out_channels),),
        layout="NHWC",
        precision="bf16",
        calibration=CalibrationSource("npz", samples=samples),
        mla_tessellation=False,
        external_dram_layout="HWC",
        promote_rank3_hwc=True,
        exit_on_stable_elf=True,
    )


def _merger(name: str) -> GraphSpec:
    """256 ViT tokens -> 64 language-width tokens. Breaks the chain's width."""
    return GraphSpec(
        name=name,
        builder=f"polima.policies.groot17.graphs:{name}",
        inputs=(_rank4_tensor("hidden", PATCH_TOKENS, VISION_CHANNELS),),
        outputs=(_tessellated("output", MERGED_TOKENS, LANGUAGE_CHANNELS),),
        layout="NHWC",
        precision="bf16",
        calibration=CalibrationSource("npz", samples=IMAGES),
        exit_on_stable_elf=True,
    )


def _vision_graphs() -> tuple[GraphSpec, ...]:
    graphs = [
        # Flattened 3x2x16x16 patches in. The patch embedding is a Conv3d whose
        # kernel equals its stride, which is exactly a linear layer -- exported
        # as one, since Model Compiler has no 3D convolution.
        _chained_stage("vit_patch", PATCH_TOKENS, PATCH_CHANNELS, VISION_CHANNELS,
                       samples=IMAGES),
    ]
    graphs += [
        _chained_stage(name, PATCH_TOKENS, VISION_CHANNELS, VISION_CHANNELS,
                       samples=IMAGES)
        for name in vision_block_names()
    ]
    graphs += [_merger(name) for name in deepstack_names()]
    graphs.append(_merger("vit_merger"))
    return tuple(graphs)


def _language_graphs() -> tuple[GraphSpec, ...]:
    graphs = []
    # The first two pairs add deepstack features, so they take extra inputs
    # and cannot join the one-input chain.
    for name in llm_names()[:2]:
        taps = LLM_ENTRY_DEEPSTACK[name]
        graphs.append(GraphSpec(
            name=name,
            builder=f"polima.policies.groot17.graphs:{name}",
            inputs=(
                _rank4_tensor("hidden", SEQUENCE, LANGUAGE_CHANNELS),
                *(_rank4_tensor(f"deepstack_{tap}", SEQUENCE, LANGUAGE_CHANNELS)
                  for tap in taps),
            ),
            outputs=(_tessellated("output", SEQUENCE, LANGUAGE_CHANNELS),),
            layout="NHWC",
            precision="bf16",
            calibration=CalibrationSource("npz", samples=1),
            exit_on_stable_elf=True,
        ))
    graphs += [
        _chained_stage(name, SEQUENCE, LANGUAGE_CHANNELS, LANGUAGE_CHANNELS, samples=1)
        for name in (*llm_names()[2:], *vl_self_attention_names())
    ]
    return tuple(graphs)


def _action_graphs() -> tuple[GraphSpec, ...]:
    #: Calibration covers the four denoise steps -- the whole range these graphs
    #: see, since tau is bucketed at 0/250/500/750 and nothing else.
    calibration = CalibrationSource("npz", samples=DENOISE_STEPS)
    graphs = [
        GraphSpec(
            name="state_project",
            builder="polima.policies.groot17.graphs:state_project",
            inputs=(_rank4_tensor("state", 1, STATE_LANE),),
            outputs=(_tessellated("state_features", 1, HIDDEN_CHANNELS),),
            layout="NHWC",
            precision="bf16",
            calibration=CalibrationSource("npz", samples=1),
        ),
        GraphSpec(
            name="action_project",
            builder="polima.policies.groot17.graphs:action_project",
            inputs=(
                _rank4_tensor("actions", CHUNK, ACTION_LANE),
                _rank4_tensor("tau_embedding", CHUNK, HIDDEN_CHANNELS),
            ),
            outputs=(_tessellated("action_features", CHUNK, HIDDEN_CHANNELS),),
            layout="NHWC",
            precision="bf16",
            calibration=calibration,
        ),
    ]
    # Input order here is the order the plan concatenates and the order the
    # ONNX is exported in. N1.6 exported these four in a different order than
    # it declared them; `graphs._export_action` now exports from this tuple.
    graphs += [
        GraphSpec(
            name=name,
            builder=f"polima.policies.groot17.graphs:{name}",
            inputs=(
                _rank4_tensor("hidden", HIDDEN_WIDTH, HIDDEN_CHANNELS),
                _rank4_tensor("temb", 1, TEMB_ELEMENTS),
                _rank4_tensor("backbone_features", SEQUENCE, LANGUAGE_CHANNELS),
                _rank4_tensor("additive_mask", 1, MASK_ELEMENTS),
            ),
            outputs=(_tessellated("hidden_out", HIDDEN_WIDTH, HIDDEN_CHANNELS),),
            layout="NHWC",
            precision="bf16",
            calibration=calibration,
            exit_on_stable_elf=True,
        )
        for name in block_names()
    ]
    graphs.append(GraphSpec(
        name="action_tail",
        builder="polima.policies.groot17.graphs:action_tail",
        inputs=(
            _rank4_tensor("hidden", HIDDEN_WIDTH, HIDDEN_CHANNELS),
            _rank4_tensor("temb", 1, TEMB_ELEMENTS),
        ),
        outputs=(_tessellated("velocity", CHUNK, ACTION_LANE),),
        layout="NHWC",
        precision="bf16",
        calibration=calibration,
    ))
    return tuple(graphs)


GROOT17_SPEC = PolicySpec(
    name="groot17",
    display_name="GR00T N1.7 (Cosmos-Reason2 VLM + flow-matching DiT)",
    # --------------------------------------------------------------- dataset --
    dataset=DatasetContract(
        state_names=STATE_NAMES,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        camera_keys=("observation.images.exterior", "observation.images.wrist"),
        camera_shape=(CAPTURE_HEIGHT, CAPTURE_WIDTH, 3),
        fps=15,
        codebase_version="v3.0",
        single_task=False,
        task_canonicalizer="polima.data.contract:canonical_task",
    ),
    # ------------------------------------------------- checkpoint provenance --
    train=TrainSpec(
        backend="groot-launch-finetune",
        conda_env="groot-n1.7",
        repo_dir_hint="GR00T-N1.6/Isaac-GR00T",
        entrypoint=("gr00t/experiment/launch_finetune.py",),
        build_args="polima.policies.groot17.train:build_args",
        defaults={
            "base_model": "nvidia/GR00T-N1.7-3B",
            "steps": 20_000,
            "batch_size": 8,
            "embodiment_tag": "NEW_EMBODIMENT",
            "action_horizon": CHUNK,
            "tune_projector": True,
            "tune_diffusion_model": True,
            "tune_llm": False,
            "tune_visual": False,
        },
        augmentation_tfs=(
            '{"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}'
        ),
        checkpoint_glob="checkpoint-*",
    ),
    # --------------------------------------------------------------- compile --
    compile=CompilePlan(
        export_entry="polima.policies.groot17.graphs:export_all",
        verify_entry="polima.policies.groot17.graphs:verify_chain",
        fixture_entry="polima.policies.groot17.graphs:write_fixtures",
        normalization_entry="polima.policies.groot17.graphs:write_normalization",
        fixture_file="groot17_fixture.npz",
        verify_atol=1e-2,
        verify_rtol=1e-2,
        graphs=(*_vision_graphs(), *_language_graphs(), *_action_graphs()),
    ),
    # ------------------------------------------------------------------ wire --
    wire=WireSpec(
        magic=WIRE_MAGIC,                 # "GR17" little-endian
        version=1,
        default_port=DEFAULT_PORT,
        request_header="<IIII",
        response_header="<IIIIfI",
        request_tensors=(
            # Four images -- two views at frames -15 and 0 -- each letterboxed,
            # resized to 256, normalized and patchified by the client.
            *(TensorSpec(f"patches_{image}", (PATCH_TOKENS, PATCH_CHANNELS))
              for image in range(IMAGES)),
            TensorSpec("state", (STATE_DIM,)),
            # The flow-matching seed, client-supplied so a run is reproducible.
            TensorSpec("noise", (CHUNK, ACTION_LANE)),
        ),
        response_shape=(CHUNK, ACTION_DIM),
        normalization_side="server",
        stats_file="normalization_stats.npz",
    ),
    # ----------------------------------------------------------------- robot --
    robot=RobotSpec(
        camera_roles=(("exterior", "Exterior"), ("wrist", "Wrist")),
        camera_hints={"exterior": "C920", "wrist": "Sonix"},
        joint_names=STATE_NAMES,
        actions_per_chunk=CHUNK,
        default_fps=15,
        max_relative_target=12,
        aggregate_fn="weighted_average",
        image_preprocessor="polima.policies.groot17.graphs:patchify_for_wire",
    ),
    runtime_plan_builder="polima.policies.groot17.runtime:build_plan",
    checkpoint_validator="polima.policies.groot17.graphs:validate_checkpoint",
)

register_policy(GROOT17_SPEC)

__all__ = ["GROOT17_SPEC", "STATE_NAMES"]
