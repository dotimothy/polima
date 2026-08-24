"""GR00T N1.6's on-device pipeline, as plan.json.

Every constant here is transcribed from the working DevKit binaries in the
GR00T-N1.6 tree, which are what currently run this policy on Modalix:

    GR00T-N1.6/devkit/groot_eagle_llima/groot_eagle_llima.cpp    Eagle geometry
    GR00T-N1.6/devkit/groot_action_llima/groot_action_llima.cpp  action geometry
    GR00T-N1.6/devkit/groot_full_llima/groot_full_llima.cpp      the joined pipeline
    GR00T-N1.6/scripts/export_groot_modalix_eagle.py             the Eagle cut
    GR00T-N1.6/scripts/export_groot_modalix_action.py            the action cut

## The pipeline

    patches(324x588)                                     } one device-resident
      -> eagle_vision_patch                              } chain: 16 ELFs, one
      -> eagle_vision_00_01 .. 26_26   (14 pairs)        } upload, one download
      -> eagle_vision_post_norm                          }
    pixel_unshuffle 18x18x1152 -> 81x4608
      -> eagle_vision_connector                          -> 81x2048 image tokens
    pack 116x2048: prompt embedding, image tokens at token 32
      -> eagle_qwen_00_01 .. 14_15 (8 pairs)             } second chain: 9 ELFs
      -> eagle_output_norm                               } -> backbone features

    state(6) -> normalize -> pad to 128 -> state_project -> 1x1536
    x = noise (50x128)
    repeat 4 times, tau = 0, 250, 500, 750:
        action_project( [x | tau_embedding] )  -> 50x1536
        hidden = [state_features | action_features]      -> 51x1536
        for pair in 0..15:
            dit_blocks_2p_2p+1( hidden, temb, backbone, mask )
        action_tail( hidden, temb )                      -> velocity 50x128
        x += 0.25 * velocity
    actions = denormalize(gather x[t*128 .. +6])         -> 50x6

Four things that look arbitrary and are not:

* **The alternating attention mask.** Even pairs attend to text, odd pairs to
  images. `export_groot_modalix_action.py` selects `text_additive if start % 4
  == 0 else image_additive` over block starts, which is pair parity. Feeding one
  mask to every block does not error; it silently attends to the wrong half of
  the prompt.
* **The 128-wide state and action lanes carrying 6 real joints.** GR00T pads
  every embodiment to its maximum state/action width. Positions 6..127 are zero
  going in and ignored coming out.
* **The 81 image tokens land at sequence position 32.** The GR1 prompt places a
  contiguous run of `<image>` tokens there, which is what lets the language
  embedding be a `pack` of one constant plus one live buffer instead of a
  gather against a 151,680-row vocabulary table.
* **The Euler update adds where SmolVLA's subtracts.** GR00T integrates
  `x += (1/N) * velocity`; the `euler` opcode computes `out - scalar * src`, so
  the plan carries a negative DT. See `EULER_SCALAR`.

## Why the loop is unrolled

Four denoise steps x (1 + 16 + 1) ELF calls are emitted as explicit steps rather
than a loop opcode, exactly as SmolVLA's ten are: the interpreter needs no
control flow, each step's tau is a literal, and `--verbose` reports which of the
seventy-odd MLA calls is slow.
"""

from __future__ import annotations

from typing import Any

from polima.policies.base import PolicySpec, RuntimePlan, Step

# ------------------------------------------------------- Eagle geometry

#: GR1's fixed image contract: one 252x252 frame, 14-pixel patches, 18x18 grid.
IMAGE_SIDE = 252
PATCH = 14
PATCH_SIDE = IMAGE_SIDE // PATCH                         # 18
VISION_WIDTH = PATCH_SIDE * PATCH_SIDE                   # 324 patch tokens
PATCH_CHANNELS = PATCH * PATCH * 3                       # 588
VISION_CHANNELS = 1152                                   # SigLIP hidden width
VISION_PAIRS = 14                                        # 27 layers, paired

#: pixel_unshuffle(2) folds the 18x18 grid into 9x9 and quadruples the channels.
UNSHUFFLE = 2
CONNECTOR_WIDTH = (PATCH_SIDE // UNSHUFFLE) ** 2         # 81 image tokens
CONNECTOR_CHANNELS = VISION_CHANNELS * UNSHUFFLE ** 2    # 4608

SEQUENCE = 116                                           # prompt + image tokens
LANGUAGE_CHANNELS = 2048                                 # Qwen hidden width
QWEN_PAIRS = 8                                           # 16 layers, paired

#: Where the contiguous run of 81 <image> tokens begins in the GR1 prompt.
#: Verified against eagle_fixture.npz: input_ids == 151669 at 32..112.
IMAGE_TOKEN_START = 32

PATCH_ELEMENTS = VISION_WIDTH * PATCH_CHANNELS           # 190512
VISION_ELEMENTS = VISION_WIDTH * VISION_CHANNELS         # 373248
CONNECTOR_ELEMENTS = CONNECTOR_WIDTH * CONNECTOR_CHANNELS        # 373248
IMAGE_TOKEN_ELEMENTS = CONNECTOR_WIDTH * LANGUAGE_CHANNELS       # 165888
BACKBONE_ELEMENTS = SEQUENCE * LANGUAGE_CHANNELS                 # 237568
MASK_ELEMENTS = SEQUENCE                                         # 116

# ------------------------------------------------------ action geometry

STATE_DIM = 6                                            # real SO-101 joints
ACTION_DIM = 6
STATE_LANE = 128                                         # GR00T's padded width
ACTION_LANE = 128
CHUNK = 50                                               # action horizon

HIDDEN_WIDTH = 1 + CHUNK                                 # 51: state + actions
HIDDEN_CHANNELS = 1536                                   # DiT width
TEMB_ELEMENTS = HIDDEN_CHANNELS                          # 1536
BLOCK_PAIRS = 16                                         # 32 DiT blocks, paired

STATE_ELEMENTS = 1 * STATE_LANE                          # 128
ACTION_ELEMENTS = CHUNK * ACTION_LANE                    # 6400
TAU_ELEMENTS = CHUNK * HIDDEN_CHANNELS                   # 76800
ACTION_INPUT_ELEMENTS = ACTION_ELEMENTS + TAU_ELEMENTS   # 83200
HIDDEN_ELEMENTS = HIDDEN_WIDTH * HIDDEN_CHANNELS         # 78336

#: The fixed four-step Euler schedule. GR00T buckets tau at 0/250/500/750 and
#: the host precomputes both sinusoidal encodings, so nothing on the board has
#: to cast or expand a scalar -- Model Compiler cannot place those on Modalix.
DENOISE_STEPS = 4
TIMESTEP_BUCKETS = (0, 250, 500, 750)
DT = 1.0 / DENOISE_STEPS                                 # 0.25

#: `euler` computes `out - scalar * src`. GR00T's update is `x += DT * v`, so
#: the scalar is negated here rather than a second opcode being added.
EULER_SCALAR = -DT

RESPONSE_ELEMENTS = CHUNK * ACTION_DIM                   # 300

WIRE_MAGIC = 0x54555247                                  # "GRUT"
DEFAULT_PORT = 8093


def two_digits(value: int) -> str:
    return f"{value:02d}"


def vision_stage_names() -> tuple[str, ...]:
    """The 27 SigLIP layers, paired -- the last one is odd and stands alone.

    `export_groot_modalix_eagle.py` names a pair by its first and last layer,
    so the tail stage is `eagle_vision_26_26`, not `eagle_vision_26_27`.
    """
    names = []
    for start in range(0, 2 * VISION_PAIRS, 2):
        end = min(start + 2, 27)
        names.append(f"eagle_vision_{two_digits(start)}_{two_digits(end - 1)}")
    return tuple(names)


def qwen_stage_names() -> tuple[str, ...]:
    return tuple(
        f"eagle_qwen_{two_digits(start)}_{two_digits(start + 1)}"
        for start in range(0, 2 * QWEN_PAIRS, 2)
    )


def block_names() -> tuple[str, ...]:
    return tuple(
        f"dit_blocks_{two_digits(start)}_{two_digits(start + 1)}"
        for start in range(0, 2 * BLOCK_PAIRS, 2)
    )


#: The two device-resident chains. One upload and one download each, with every
#: intermediate staying in MLA DRAM -- this is what took Eagle from 362 ms to
#: 169 ms in the GR00T-N1.6 build tree's own validation runs.
VISION_CHAIN = ("eagle_vision_patch", *vision_stage_names(), "eagle_vision_post_norm")
QWEN_CHAIN = (*qwen_stage_names(), "eagle_output_norm")


# -------------------------------------------------------------- buffers


def buffers() -> dict[str, int]:
    """Every buffer the plan touches, allocated once at load."""
    sizes = {
        # wire inputs
        "patches": PATCH_ELEMENTS,
        "state": STATE_DIM,
        "noise": ACTION_ELEMENTS,
        # Eagle
        "vision_hidden": VISION_ELEMENTS,
        "connector_input": CONNECTOR_ELEMENTS,
        "image_tokens": IMAGE_TOKEN_ELEMENTS,
        "language_embeddings": BACKBONE_ELEMENTS,
        "backbone_features": BACKBONE_ELEMENTS,
        # masks and time embeddings, materialised once from constants/
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
    for step in range(DENOISE_STEPS):
        sizes[f"temb_{step:02d}"] = TEMB_ELEMENTS
    return sizes


# ---------------------------------------------------------------- steps


def _step(op: str, out: str, **args) -> Step:
    return Step(op, out, args)


def _from_sidecar(name: str, sidecar: str, count: int) -> Step:
    """Materialise one constant into a buffer.

    `run_elf` reads buffers, not sidecars, and the masks and time embeddings are
    each read 64 times per inference. Copying them once at the top of the plan
    keeps the hot path free of special cases.
    """
    return _step("pack", name,
                 parts=[{"src": sidecar, "dst_offset": 0, "count": count, "sidecar": True}])


def build_steps() -> list[Step]:
    steps: list[Step] = []

    # --- constants into buffers -------------------------------------------
    steps.append(_from_sidecar("text_mask", "text_additive_mask", MASK_ELEMENTS))
    steps.append(_from_sidecar("image_mask", "image_additive_mask", MASK_ELEMENTS))
    for index in range(DENOISE_STEPS):
        steps.append(_from_sidecar(
            f"temb_{index:02d}", f"timestep_embedding_{index}", TEMB_ELEMENTS))

    # --- Eagle vision: one chain, then the host's channel fold -------------
    # The client patchifies: the 3x252x252 frame becomes 324 rows of 588 before
    # the wire, so the board never reshapes an image.
    steps.append(_step("run_elf_chain", "vision_hidden",
                       graphs=list(VISION_CHAIN), **{"in": ["patches"]}))
    steps.append(_step("pixel_unshuffle", "connector_input", src="vision_hidden",
                       grid=PATCH_SIDE, channels=VISION_CHANNELS, factor=UNSHUFFLE))
    steps.append(_step("run_elf", "image_tokens", graph="eagle_vision_connector",
                       **{"in": ["connector_input"]}))

    # --- the 116-token language sequence ----------------------------------
    # Every non-image token is fixed for a deployed checkpoint, so the prompt's
    # embeddings ship as one constant with the image slots zeroed; the vision
    # tokens are written over that contiguous window.
    steps.append(_step("pack", "language_embeddings", parts=[
        {"src": "prompt_embedding", "dst_offset": 0,
         "count": BACKBONE_ELEMENTS, "sidecar": True},
        {"src": "image_tokens", "dst_offset": IMAGE_TOKEN_START * LANGUAGE_CHANNELS,
         "count": IMAGE_TOKEN_ELEMENTS},
    ]))
    steps.append(_step("run_elf_chain", "backbone_features",
                       graphs=list(QWEN_CHAIN), **{"in": ["language_embeddings"]}))

    # --- state: normalize 6, widen to the 128 lane, project to 1536 --------
    # Normalizing into a 6-wide buffer and packing separately is deliberate.
    # Normalizing the 128-wide lane directly would apply the statistics to the
    # 122 padding slots too, turning structural zeros into -mean/std.
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
        # Ping-pong so no ELF reads and writes the same buffer. Sixteen pairs is
        # even, so the last block lands back in hidden_a for the tail.
        for pair, name in enumerate(block_names()):
            source = "hidden_a" if pair % 2 == 0 else "hidden_b"
            target = "hidden_b" if pair % 2 == 0 else "hidden_a"
            mask = "text_mask" if pair % 2 == 0 else "image_mask"
            steps.append(_step("run_elf", target, graph=name,
                               **{"in": [source, temb, "backbone_features", mask]}))
        steps.append(_step("run_elf", "velocity", graph="action_tail",
                           **{"in": ["hidden_a", temb]}))
        steps.append(_step("euler", "actions", src="velocity", scalar=EULER_SCALAR))

    # --- 128-wide lane -> 6 real joints, then denormalize ------------------
    steps.append(_step("gather_strided", "actions_lane", src="actions",
                       stride=ACTION_LANE, take=ACTION_DIM, count=CHUNK))
    steps.append(_step("denormalize", "action_chunk", src="actions_lane",
                       mean="action_mean", std="action_std"))
    return steps


#: Constants shipped in the bundle's constants/ directory and read by name.
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
