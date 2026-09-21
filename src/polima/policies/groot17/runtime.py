"""GR00T N1.7's on-device pipeline, as plan.json.

Every constant here was measured, not assumed: they come from one traced
inference of the `nvidia/GR00T-N1.7-3B` base checkpoint under the
`oxe_droid_relative_eef_relative_joint` embodiment, with the model loaded from
the `n1.7-release` tag of Isaac-GR00T (see `graphs.trace`, which re-checks every
one of them at export time and refuses to continue if any has drifted).

## The pipeline

    for each of the 4 images (2 views x 2 frames):
        patches(256x1536)
          -> vit_patch, vit_blocks_00_01 .. 04_05   chain 0 -> deepstack tap 0
          -> vit_blocks_06_07 .. 10_11              chain 1 -> deepstack tap 1
          -> vit_blocks_12_13 .. 16_17              chain 2 -> deepstack tap 2
          -> vit_blocks_18_19 .. 22_23              chain 3
          -> vit_merger                             -> 64x2048 image tokens
    pack 282x2048: prompt embedding + four 64-token image runs
    pack three 282x2048 deepstack tensors at the same four runs
      -> llm_00_01(hidden, deepstack_0, deepstack_1)
      -> llm_02_03(hidden, deepstack_2)
      -> llm_04_05 .. llm_14_15, vl_self_attention_00_01, _02_03   one chain
                                                    -> backbone features

    state(17) -> normalize -> pad to 132 -> state_project -> 1x1536
    x = noise (40x132)
    repeat 4 times, tau = 0, 250, 500, 750:
        action_project( [x | tau_embedding] )       -> 40x1536
        hidden = [state_features | action_features] -> 41x1536
        for pair in 0..15:
            dit_blocks_2p_2p+1( hidden, temb, backbone, mask )
        action_tail( hidden, temb )                 -> velocity 40x132
        x += 0.25 * velocity
    actions = denormalize(gather x[t*132 .. +17])   -> 40x17

## What changed from N1.6, and why the cut moved

N1.7 replaces the Eagle backbone with Cosmos-Reason2-2B, a Qwen3-VL model.
Three of its properties decide the decomposition:

* **Deepstack.** Qwen3-VL taps its vision encoder after blocks 5, 11 and 17
  and adds those features into the language model's first three layers. The
  vision chain is therefore split at the taps, and the first two language
  graphs take the deepstack features as extra inputs -- which is also why they
  cannot join the device-resident chain, which admits one input only.
* **Per-image attention.** Qwen3-VL's vision attention never crosses an image
  boundary, so each of the four images runs through the same ELFs separately.
  That is exact, not an approximation, and it keeps every vision graph at 256
  tokens instead of 1024.
* **M-RoPE and a fixed prompt.** The 3D rotary positions depend only on the
  prompt and the image grid, both fixed for a deployed checkpoint, so they are
  computed at export and baked into each language graph.

Two things carried over from N1.6 that look arbitrary and are not:

* **The alternating attention mask.** Cross-attention blocks alternate between
  text and image tokens (`attend_text_every_n_blocks: 2`); even pairs read the
  text mask, odd pairs the image mask. Feeding one mask to every block does not
  error -- it silently attends to the wrong half of the sequence.
* **The Euler update adds.** GR00T integrates `x += (1/N) * velocity`; the
  `euler` opcode computes `out - scalar * src`, so the plan carries `-DT`.

## What this bundle's actions are

The DROID embodiment predicts *relative* end-effector and joint actions. The
plan denormalizes them (per timestep -- the statistics differ at every step of
the horizon) but does not convert them to absolute targets: composing a 9D
end-effector pose against the current state is not linear, so it belongs to the
robot client, not to a buffer interpreter.
"""

from __future__ import annotations

from typing import Any

from polima.policies.base import PolicySpec, RuntimePlan, Step

# ------------------------------------------------------- vision geometry

#: The embodiment the base checkpoint was traced under. Two cameras, one arm --
#: the closest pretrain embodiment to a single-arm SO-101 setup.
EMBODIMENT_TAG = "oxe_droid_relative_eef_relative_joint"
EMBODIMENT_ID = 24

VIEWS = 2                                        # exterior + wrist
FRAMES = 2                                       # delta_indices [-15, 0]
IMAGES = VIEWS * FRAMES                          # 4 images into the backbone

IMAGE_SIDE = 256                                 # after letterbox, resize, crop
PATCH = 16
TEMPORAL_PATCH = 2                               # a still image is doubled in time
PATCH_SIDE = IMAGE_SIDE // PATCH                 # 16
PATCH_TOKENS = PATCH_SIDE * PATCH_SIDE           # 256 per image
PATCH_CHANNELS = 3 * TEMPORAL_PATCH * PATCH * PATCH      # 1536
VISION_CHANNELS = 1024                           # Qwen3-VL ViT width
VISION_BLOCKS = 24
VISION_PAIRS = VISION_BLOCKS // 2                # 12

MERGE = 2
MERGED_TOKENS = (PATCH_SIDE // MERGE) ** 2       # 64 tokens per image
MERGED_CHANNELS = VISION_CHANNELS * MERGE ** 2   # 4096

#: Vision blocks whose output feeds the language model's first three layers.
DEEPSTACK_BLOCKS = (5, 11, 17)

#: Qwen3-VL's image normalization (preprocessor_config.json). Re-checked at
#: export against the processor that produced the reference.
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)

# ----------------------------------------------------- language geometry

LANGUAGE_CHANNELS = 2048                         # Qwen3 hidden width
SEQUENCE = 282                                   # fixed prompt + 4 x 64 image tokens

#: (start, length) of each image's token run in the prompt. Two tokens --
#: <|vision_end|><|vision_start|> -- separate consecutive runs.
VISUAL_RUNS = ((4, 64), (70, 64), (136, 64), (202, 64))

LLM_LAYERS = 16                                  # select_layer: pre-truncated
LLM_PAIRS = LLM_LAYERS // 2                      # 8
VL_SELF_ATTENTION_LAYERS = 4
VL_SELF_ATTENTION_PAIRS = VL_SELF_ATTENTION_LAYERS // 2  # 2

PATCH_ELEMENTS = PATCH_TOKENS * PATCH_CHANNELS           # 393216
VISION_ELEMENTS = PATCH_TOKENS * VISION_CHANNELS         # 262144
IMAGE_TOKEN_ELEMENTS = MERGED_TOKENS * LANGUAGE_CHANNELS  # 131072
BACKBONE_ELEMENTS = SEQUENCE * LANGUAGE_CHANNELS         # 577536
MASK_ELEMENTS = SEQUENCE                                 # 282

# ------------------------------------------------------- action geometry

#: DROID state/action layout: eef_9d (9), gripper_position (1), joint_position (7).
STATE_KEYS = (("eef_9d", 9), ("gripper_position", 1), ("joint_position", 7))
STATE_DIM = sum(width for _, width in STATE_KEYS)       # 17
ACTION_DIM = STATE_DIM                                   # 17
STATE_LANE = 132                                         # max_state_dim
ACTION_LANE = 132                                        # max_action_dim
CHUNK = 40                                               # action_horizon

HIDDEN_WIDTH = 1 + CHUNK                                 # 41: state + actions
HIDDEN_CHANNELS = 1536                                   # DiT input width
TEMB_ELEMENTS = HIDDEN_CHANNELS                          # 1536
BLOCK_PAIRS = 16                                         # 32 DiT blocks, paired

ACTION_ELEMENTS = CHUNK * ACTION_LANE                    # 5280
TAU_ELEMENTS = CHUNK * HIDDEN_CHANNELS                   # 61440
ACTION_INPUT_ELEMENTS = ACTION_ELEMENTS + TAU_ELEMENTS   # 66720
HIDDEN_ELEMENTS = HIDDEN_WIDTH * HIDDEN_CHANNELS         # 62976

#: The fixed four-step Euler schedule. The host precomputes both sinusoidal
#: encodings, so nothing on the board has to cast or expand a scalar.
DENOISE_STEPS = 4
TIMESTEP_BUCKETS = (0, 250, 500, 750)
DT = 1.0 / DENOISE_STEPS                                 # 0.25
EULER_SCALAR = -DT

RESPONSE_ELEMENTS = CHUNK * ACTION_DIM                   # 680

WIRE_MAGIC = int.from_bytes(b"GR17", "little")           # 0x37315247
DEFAULT_PORT = 8094


def two_digits(value: int) -> str:
    return f"{value:02d}"


def vision_block_names() -> tuple[str, ...]:
    return tuple(
        f"vit_blocks_{two_digits(start)}_{two_digits(start + 1)}"
        for start in range(0, VISION_BLOCKS, 2)
    )


def deepstack_names() -> tuple[str, ...]:
    return tuple(f"vit_deepstack_{index}" for index in range(len(DEEPSTACK_BLOCKS)))


def llm_names() -> tuple[str, ...]:
    return tuple(
        f"llm_{two_digits(start)}_{two_digits(start + 1)}"
        for start in range(0, LLM_LAYERS, 2)
    )


def vl_self_attention_names() -> tuple[str, ...]:
    return tuple(
        f"vl_self_attention_{two_digits(start)}_{two_digits(start + 1)}"
        for start in range(0, VL_SELF_ATTENTION_LAYERS, 2)
    )


def block_names() -> tuple[str, ...]:
    return tuple(
        f"dit_blocks_{two_digits(start)}_{two_digits(start + 1)}"
        for start in range(0, 2 * BLOCK_PAIRS, 2)
    )


def _vision_chains() -> tuple[tuple[str, ...], ...]:
    """Four chains, split exactly where a deepstack tap needs a download.

    A pair ending at block b is `vit_blocks_(b-1)_b`, so the tap after block 5
    closes the chain at `vit_blocks_04_05`.
    """
    pairs = list(vision_block_names())
    chains, current = [], ["vit_patch"]
    for name in pairs:
        current.append(name)
        last_block = int(name.rsplit("_", 1)[1])
        if last_block in DEEPSTACK_BLOCKS:
            chains.append(tuple(current))
            current = []
    chains.append(tuple(current))
    return tuple(chains)


#: Vision chains; chain k (k < 3) ends at deepstack tap k.
VISION_CHAINS = _vision_chains()

#: The two language graphs that take deepstack inputs, and so stand alone.
LLM_ENTRY = llm_names()[:2]
#: Everything after the deepstack layers shares one fixed-size buffer.
LANGUAGE_CHAIN = (*llm_names()[2:], *vl_self_attention_names())

#: Which deepstack tensors each entry graph adds, in input order after `hidden`.
LLM_ENTRY_DEEPSTACK = {"llm_00_01": (0, 1), "llm_02_03": (2,)}


# -------------------------------------------------------------- buffers


def buffers() -> dict[str, int]:
    """Every buffer the plan touches, allocated once at load."""
    sizes = {
        "state": STATE_DIM,
        "noise": ACTION_ELEMENTS,
        # vision, ping-ponged so no chain reads and writes one buffer
        "vision_a": VISION_ELEMENTS,
        "vision_b": VISION_ELEMENTS,
        # language
        "language_embeddings": BACKBONE_ELEMENTS,
        "llm_a": BACKBONE_ELEMENTS,
        "llm_b": BACKBONE_ELEMENTS,
        "backbone_features": BACKBONE_ELEMENTS,
        "text_mask": MASK_ELEMENTS,
        "image_mask": MASK_ELEMENTS,
        # state path
        "state_normalized": STATE_DIM,
        "state_lane": STATE_LANE,
        "state_features": HIDDEN_CHANNELS,
        # denoise loop
        "actions": ACTION_ELEMENTS,
        "action_input": ACTION_INPUT_ELEMENTS,
        "action_features": TAU_ELEMENTS,
        "hidden_a": HIDDEN_ELEMENTS,
        "hidden_b": HIDDEN_ELEMENTS,
        "velocity": ACTION_ELEMENTS,
        # output
        "actions_lane": RESPONSE_ELEMENTS,
        "action_chunk": RESPONSE_ELEMENTS,
    }
    for image in range(IMAGES):
        sizes[f"patches_{image}"] = PATCH_ELEMENTS
        sizes[f"image_tokens_{image}"] = IMAGE_TOKEN_ELEMENTS
        for tap in range(len(DEEPSTACK_BLOCKS)):
            sizes[f"deepstack_{tap}_{image}"] = IMAGE_TOKEN_ELEMENTS
    for tap in range(len(DEEPSTACK_BLOCKS)):
        sizes[f"deepstack_full_{tap}"] = BACKBONE_ELEMENTS
    for step in range(DENOISE_STEPS):
        sizes[f"temb_{step:02d}"] = TEMB_ELEMENTS
    return sizes


# ---------------------------------------------------------------- steps


def _step(op: str, out: str, **args) -> Step:
    return Step(op, out, args)


def _from_sidecar(name: str, sidecar: str, count: int) -> Step:
    return _step("pack", name,
                 parts=[{"src": sidecar, "dst_offset": 0, "count": count, "sidecar": True}])


def _runs(source: str) -> list[dict[str, Any]]:
    """One pack part per image, landing on that image's token run."""
    return [
        {"src": f"{source}_{image}", "dst_offset": start * LANGUAGE_CHANNELS,
         "count": length * LANGUAGE_CHANNELS}
        for image, (start, length) in enumerate(VISUAL_RUNS)
    ]


def build_steps() -> list[Step]:
    steps: list[Step] = []

    # --- constants into buffers -------------------------------------------
    steps.append(_from_sidecar("text_mask", "text_additive_mask", MASK_ELEMENTS))
    steps.append(_from_sidecar("image_mask", "image_additive_mask", MASK_ELEMENTS))
    for index in range(DENOISE_STEPS):
        steps.append(_from_sidecar(
            f"temb_{index:02d}", f"timestep_embedding_{index}", TEMB_ELEMENTS))

    # --- vision: each image separately, tapping deepstack between chains ----
    for image in range(IMAGES):
        source = f"patches_{image}"
        for chain_index, chain in enumerate(VISION_CHAINS):
            target = "vision_a" if chain_index % 2 == 0 else "vision_b"
            steps.append(_step("run_elf_chain", target, graphs=list(chain),
                               **{"in": [source]}))
            if chain_index < len(DEEPSTACK_BLOCKS):
                steps.append(_step("run_elf", f"deepstack_{chain_index}_{image}",
                                   graph=deepstack_names()[chain_index],
                                   **{"in": [target]}))
            source = target
        steps.append(_step("run_elf", f"image_tokens_{image}", graph="vit_merger",
                           **{"in": [source]}))

    # --- the 282-token language sequence ----------------------------------
    steps.append(_step("pack", "language_embeddings", parts=[
        {"src": "prompt_embedding", "dst_offset": 0,
         "count": BACKBONE_ELEMENTS, "sidecar": True},
        *_runs("image_tokens"),
    ]))
    for tap in range(len(DEEPSTACK_BLOCKS)):
        steps.append(_step("pack", f"deepstack_full_{tap}", parts=_runs(f"deepstack_{tap}")))

    steps.append(_step("run_elf", "llm_a", graph="llm_00_01", **{"in": [
        "language_embeddings",
        *(f"deepstack_full_{tap}" for tap in LLM_ENTRY_DEEPSTACK["llm_00_01"]),
    ]}))
    steps.append(_step("run_elf", "llm_b", graph="llm_02_03", **{"in": [
        "llm_a",
        *(f"deepstack_full_{tap}" for tap in LLM_ENTRY_DEEPSTACK["llm_02_03"]),
    ]}))
    steps.append(_step("run_elf_chain", "backbone_features",
                       graphs=list(LANGUAGE_CHAIN), **{"in": ["llm_b"]}))

    # --- state: normalize 17, widen to the 132 lane, project --------------
    # Normalizing the 132-wide lane directly would apply the statistics to the
    # 115 padding slots too, turning structural zeros into -mean/std.
    steps.append(_step("normalize", "state_normalized", src="state",
                       mean="state_mean", std="state_std"))
    steps.append(_step("pack", "state_lane",
                       parts=[{"src": "state_normalized", "dst_offset": 0,
                               "count": STATE_DIM}]))
    steps.append(_step("run_elf", "state_features", graph="state_project",
                       **{"in": ["state_lane"]}))

    # --- flow matching, unrolled ------------------------------------------
    steps.append(_step("slice", "actions", src="noise", count=ACTION_ELEMENTS))
    for index in range(DENOISE_STEPS):
        temb = f"temb_{index:02d}"
        steps.append(_step("pack", "action_input", parts=[
            {"src": "actions", "dst_offset": 0, "count": ACTION_ELEMENTS},
            {"src": f"tau_embedding_{index}", "dst_offset": ACTION_ELEMENTS,
             "count": TAU_ELEMENTS, "sidecar": True},
        ]))
        steps.append(_step("run_elf", "action_features", graph="action_project",
                           **{"in": ["action_input"]}))
        steps.append(_step("pack", "hidden_a", parts=[
            {"src": "state_features", "dst_offset": 0, "count": HIDDEN_CHANNELS},
            {"src": "action_features", "dst_offset": HIDDEN_CHANNELS,
             "count": TAU_ELEMENTS},
        ]))
        # Sixteen pairs is even, so the last block lands back in hidden_a.
        for pair, name in enumerate(block_names()):
            source = "hidden_a" if pair % 2 == 0 else "hidden_b"
            target = "hidden_b" if pair % 2 == 0 else "hidden_a"
            mask = "text_mask" if pair % 2 == 0 else "image_mask"
            steps.append(_step("run_elf", target, graph=name,
                               **{"in": [source, temb, "backbone_features", mask]}))
        steps.append(_step("run_elf", "velocity", graph="action_tail",
                           **{"in": ["hidden_a", temb]}))
        steps.append(_step("euler", "actions", src="velocity", scalar=EULER_SCALAR))

    # --- 132-wide lane -> 17 real dimensions, denormalized per timestep ----
    steps.append(_step("gather_strided", "actions_lane", src="actions",
                       stride=ACTION_LANE, take=ACTION_DIM, count=CHUNK))
    # GR00T clips normalized actions to [-1, 1] before unnormalizing
    # (`unnormalize_values_minmax`); without it an out-of-range prediction
    # would be scaled past the q01..q99 envelope the statistics describe.
    steps.append(_step("denormalize", "action_chunk", src="actions_lane",
                       mean="action_mean", std="action_std",
                       clip_min=-1.0, clip_max=1.0))
    return steps


#: Constants shipped in the bundle's constants/ directory and read by name.
#: action_mean/std are CHUNK * ACTION_DIM long: relative-action statistics
#: differ at every horizon step, and `denormalize` indexes stats modulo length.
SIDECARS = (
    "prompt_embedding",
    "image_additive_mask",
    "text_additive_mask",
    *(f"tau_embedding_{index}" for index in range(DENOISE_STEPS)),
    *(f"timestep_embedding_{index}" for index in range(DENOISE_STEPS)),
    "state_mean",
    "state_std",
    "action_mean",
    "action_std",
)


def build_plan(spec: PolicySpec, context: Any = None) -> RuntimePlan:
    return RuntimePlan(
        buffers=buffers(),
        steps=tuple(build_steps()),
        result="action_chunk",
        sidecars=SIDECARS,
    )
