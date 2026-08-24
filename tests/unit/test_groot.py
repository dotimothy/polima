"""GR00T's spec and execution plan, pinned against the DevKit sources.

GR00T is the first policy PoLiMa cuts into more graphs than a person can check
by eye -- forty-five -- and the first whose pipeline is two device-resident
chains with host arithmetic between them. Both facts make the interesting
failures silent: a transposed channel fold, a mask fed to the wrong half of the
blocks, or an Euler update with the wrong sign all produce finite, plausible,
wrong actions.

So the geometry is verified against the C++ that currently runs this policy on
the board (`GR00T-N1.6/devkit/*.cpp`) rather than restated here, and the host
arithmetic is verified by replaying the plan with the MLA stubbed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from polima.policies.groot import GROOT_SPEC
from polima.policies.groot import runtime as rt
from polima.policies.registry import get_policy
from polima.wire.server_stub import StubPlan

GROOT_TREE = Path(__file__).resolve().parents[3] / "GR00T-N1.6"
DEVKIT = GROOT_TREE / "devkit"
BUILD = GROOT_TREE / "outputs" / "modalix_groot_n1d6_base_gr1"

needs_devkit = pytest.mark.skipif(
    not (DEVKIT / "groot_action_llima" / "groot_action_llima.cpp").is_file(),
    reason="GR00T-N1.6 devkit sources not present",
)
needs_fixture = pytest.mark.skipif(
    not (BUILD / "eagle_fixture" / "eagle_fixture.npz").is_file(),
    reason="GR00T-N1.6 build tree not present",
)


def devkit_constants(name: str) -> dict[str, int]:
    """`constexpr size_t kFoo = 123;` -> {"kFoo": 123}, arithmetic excluded."""
    source = (DEVKIT / name / f"{name}.cpp").read_text(encoding="utf-8")
    pattern = re.compile(r"constexpr (?:size_t|int|int64_t) (k\w+) = (\d+);")
    return {match.group(1): int(match.group(2)) for match in pattern.finditer(source)}


# ------------------------------------------------------------------ the spec


def test_spec_registers_with_forty_five_graphs():
    spec = get_policy("groot")
    assert spec is GROOT_SPEC
    assert len(spec.compile.graphs) == 45
    # 26 Eagle stages + 19 action graphs, as the two export reports record.
    assert sum(name.startswith("eagle_") for name in spec.compile.names) == 26


def test_the_two_chains_cover_every_eagle_stage_but_the_connector():
    chained = set(rt.VISION_CHAIN) | set(rt.QWEN_CHAIN)
    eagle = {name for name in GROOT_SPEC.compile.names if name.startswith("eagle_")}
    # The connector is the one Eagle graph left out, because the host has to
    # fold 18x18x1152 into 9x9x4608 immediately before it.
    assert eagle - chained == {"eagle_vision_connector"}
    assert len(rt.VISION_CHAIN) == 16 and len(rt.QWEN_CHAIN) == 9


def test_chained_graphs_expose_a_flat_hwc_boundary():
    # run_elf_chain binds one ELF's output buffer straight to the next ELF's
    # input. That is only sound if neither end is tessellated.
    for name in (*rt.VISION_CHAIN, *rt.QWEN_CHAIN):
        graph = GROOT_SPEC.graph(name)
        assert graph.external_dram_layout == "HWC", name
        assert not graph.mla_tessellation, name
        assert graph.promote_rank3_hwc, name
        assert graph.layout == "NHWC", name


def test_unchained_outputs_are_hwc16_for_the_host_to_unpack():
    # Everything the host reads back corresponds to a detessellate_hwc16 call
    # in groot_action_llima.cpp.
    for name in ("eagle_vision_connector", "state_project", "action_project",
                 "action_tail", *rt.block_names()):
        output = GROOT_SPEC.graph(name).outputs[0]
        assert output.dram_layout == "hwc16", name
        assert output.logical_width and output.logical_channels, name


def test_wire_magic_is_grut():
    assert GROOT_SPEC.wire.magic_ascii == "GRUT"
    assert GROOT_SPEC.wire.default_port == 8093


def test_dataset_needs_the_v2_1_downgrade_in_its_own_environment():
    converter = GROOT_SPEC.dataset.converter
    assert converter is not None
    assert converter.target_codebase_version == "v2.1"
    # The converter and the trainer need incompatible lerobot versions, which is
    # why this is a second env and not a function call.
    assert converter.conda_env != GROOT_SPEC.train.conda_env


# ---------------------------------------------------------------- geometry


@needs_devkit
def test_action_geometry_matches_the_devkit_binary():
    constants = devkit_constants("groot_action_llima")
    assert constants["kStateElements"] == rt.STATE_LANE
    assert constants["kWidth"] == rt.CHUNK
    assert constants["kActionChannels"] == rt.ACTION_LANE
    assert constants["kHiddenWidth"] == rt.HIDDEN_WIDTH
    assert constants["kHiddenChannels"] == rt.HIDDEN_CHANNELS
    assert constants["kTembElements"] == rt.TEMB_ELEMENTS
    assert constants["kBackboneWidth"] == rt.SEQUENCE
    assert constants["kBackboneChannels"] == rt.LANGUAGE_CHANNELS
    assert constants["kMaskElements"] == rt.MASK_ELEMENTS
    assert constants["kBlockPairs"] == rt.BLOCK_PAIRS


@needs_devkit
def test_eagle_geometry_matches_the_devkit_binary():
    constants = devkit_constants("groot_eagle_llima")
    assert constants["kImageSide"] == rt.IMAGE_SIDE
    assert constants["kPatch"] == rt.PATCH
    assert constants["kPatchSide"] == rt.PATCH_SIDE
    assert constants["kVisionWidth"] == rt.VISION_WIDTH
    assert constants["kPatchChannels"] == rt.PATCH_CHANNELS
    assert constants["kVisionChannels"] == rt.VISION_CHANNELS
    assert constants["kConnectorWidth"] == rt.CONNECTOR_WIDTH
    assert constants["kConnectorChannels"] == rt.CONNECTOR_CHANNELS
    assert constants["kSequence"] == rt.SEQUENCE
    assert constants["kLanguageChannels"] == rt.LANGUAGE_CHANNELS
    assert constants["kVisionPairs"] == rt.VISION_PAIRS
    assert constants["kQwenPairs"] == rt.QWEN_PAIRS


@needs_fixture
def test_image_tokens_are_contiguous_where_the_pack_assumes():
    # The language embedding is a `pack` of one constant plus one live buffer,
    # which is only correct while the image tokens form one unbroken run.
    fixture = np.load(BUILD / "eagle_fixture" / "eagle_fixture.npz")
    positions = np.flatnonzero(fixture["input_ids"][0] == 151669)
    assert positions.size == rt.CONNECTOR_WIDTH
    assert positions[0] == rt.IMAGE_TOKEN_START
    assert np.all(np.diff(positions) == 1)


# -------------------------------------------------------------------- plan


def plan_dict() -> dict:
    plan = GROOT_SPEC.build_runtime_plan()
    return {
        "buffers": dict(plan.buffers),
        "steps": [{"op": s.op, "out": s.out, "args": dict(s.args)} for s in plan.steps],
        "result": plan.result,
    }


def graph_calls(op_filter=("run_elf", "run_elf_chain")) -> list[str]:
    calls = []
    for step in plan_dict()["steps"]:
        if step["op"] == "run_elf":
            calls.append(step["args"]["graph"])
        elif step["op"] == "run_elf_chain":
            calls.extend(step["args"]["graphs"])
    return calls


def test_plan_runs_each_graph_the_right_number_of_times():
    calls = graph_calls()
    # Eagle runs once; every action graph runs once per denoise step.
    assert calls.count("eagle_vision_patch") == 1
    assert calls.count("eagle_output_norm") == 1
    assert calls.count("state_project") == 1
    assert calls.count("action_project") == rt.DENOISE_STEPS
    assert calls.count("action_tail") == rt.DENOISE_STEPS
    for name in rt.block_names():
        assert calls.count(name) == rt.DENOISE_STEPS, name
    # 26 Eagle + 1 state + 4 x (1 project + 16 pairs + 1 tail)
    assert len(calls) == 26 + 1 + rt.DENOISE_STEPS * 18


def test_denoise_loop_is_unrolled_not_looped():
    assert not GROOT_SPEC.build_runtime_plan().loops


def test_blocks_alternate_text_and_image_masks_by_pair_parity():
    # export_groot_modalix_action.py picks `text if start % 4 == 0 else image`
    # over block starts, which is pair parity. One mask everywhere still runs.
    seen = {}
    for step in plan_dict()["steps"]:
        if step["op"] != "run_elf":
            continue
        graph = step["args"]["graph"]
        if not graph.startswith("dit_blocks_"):
            continue
        mask = step["args"]["in"][3]
        seen.setdefault(graph, set()).add(mask)
    assert len(seen) == rt.BLOCK_PAIRS
    for pair, name in enumerate(rt.block_names()):
        expected = "text_mask" if pair % 2 == 0 else "image_mask"
        assert seen[name] == {expected}, name


def test_euler_adds_because_groot_integrates_forward():
    # The opcode computes `out - scalar * src`; GR00T's update is `x += DT * v`.
    steps = [s for s in plan_dict()["steps"] if s["op"] == "euler"]
    assert len(steps) == rt.DENOISE_STEPS
    assert all(step["args"]["scalar"] == pytest.approx(-rt.DT) for step in steps)


def test_state_is_normalized_before_padding_not_after():
    # Normalizing the 128-wide lane would apply the statistics to the 122
    # padding slots, turning structural zeros into -mean/std.
    ops = [(s["op"], s["out"]) for s in plan_dict()["steps"]]
    assert ops.index(("normalize", "state_normalized")) < ops.index(("pack", "state_lane"))
    normalize = next(s for s in plan_dict()["steps"] if s["op"] == "normalize")
    assert GROOT_SPEC.build_runtime_plan().buffers[normalize["out"]] == rt.STATE_DIM


def test_no_elf_reads_and_writes_the_same_buffer():
    for step in plan_dict()["steps"]:
        if step["op"] in ("run_elf", "run_elf_chain"):
            assert step["out"] not in step["args"]["in"], step


def test_vision_tokens_land_on_the_image_window():
    pack = next(s for s in plan_dict()["steps"]
                if s["op"] == "pack" and s["out"] == "language_embeddings")
    prompt, image = pack["args"]["parts"]
    assert prompt["sidecar"] and prompt["count"] == rt.BACKBONE_ELEMENTS
    assert not image.get("sidecar")
    assert image["dst_offset"] == rt.IMAGE_TOKEN_START * rt.LANGUAGE_CHANNELS
    assert image["count"] == rt.CONNECTOR_WIDTH * rt.LANGUAGE_CHANNELS


# ------------------------------------------------------- host arithmetic


def test_pixel_unshuffle_matches_the_channel_order_torch_produces():
    # out[(i*s+j), c*f^2 + dy*f + dx] = src[(i*f+dy)*grid + (j*f+dx), c].
    # Transposing this still yields finite features, so it is checked directly.
    grid, channels, factor = 6, 3, 2
    side = grid // factor
    source = np.arange(grid * grid * channels, dtype=np.float32)
    expected = np.zeros((side * side, channels * factor * factor), dtype=np.float32)
    for i in range(side):
        for j in range(side):
            for c in range(channels):
                for dy in range(factor):
                    for dx in range(factor):
                        expected[i * side + j, c * factor * factor + dy * factor + dx] = (
                            source[((i * factor + dy) * grid + (j * factor + dx)) * channels + c]
                        )

    plan = StubPlan(
        buffers={"src": source.size, "dst": expected.size},
        steps=[{"op": "pixel_unshuffle", "out": "dst",
                "args": {"src": "src", "grid": grid, "channels": channels,
                         "factor": factor}}],
        result="dst",
    )
    got = plan.run({"src": source})
    assert np.array_equal(got, expected.ravel())


def test_plan_replays_end_to_end_with_the_mla_stubbed(tmp_path):
    """Buffer plumbing, start to finish.

    Every ELF is replaced by a deterministic function of its input, so this says
    nothing about the model -- only that all 45 graphs receive the element counts
    the spec declares and that the result reaches the wire's shape.
    """
    plan_json = plan_dict()
    sizes = plan_json["buffers"]
    seen: list[tuple[str, int]] = []

    def graph_fn(name: str, values: np.ndarray) -> np.ndarray:
        seen.append((name, values.size))
        graph = GROOT_SPEC.graph(name)
        expected = sum(tensor.elements for tensor in graph.inputs)
        assert values.size == expected, f"{name}: got {values.size}, spec says {expected}"
        out = sum(tensor.elements for tensor in graph.outputs)
        # A cheap, input-dependent stand-in: enough that a mis-plumbed buffer
        # changes the answer, cheap enough to run 45 graphs in a unit test.
        return np.full(out, float(values.sum() % 7.0), dtype=np.float32)

    constants = {
        "prompt_embedding": np.zeros(rt.BACKBONE_ELEMENTS, dtype=np.float32),
        "image_additive_mask": np.zeros(rt.MASK_ELEMENTS, dtype=np.float32),
        "text_additive_mask": np.zeros(rt.MASK_ELEMENTS, dtype=np.float32),
        "state_mean": np.zeros(rt.STATE_DIM, dtype=np.float32),
        "state_std": np.ones(rt.STATE_DIM, dtype=np.float32),
        "action_mean": np.zeros(rt.ACTION_DIM, dtype=np.float32),
        "action_std": np.ones(rt.ACTION_DIM, dtype=np.float32),
    }
    for index in range(rt.DENOISE_STEPS):
        constants[f"tau_embedding_{index}"] = np.zeros(rt.TAU_ELEMENTS, dtype=np.float32)
        constants[f"timestep_embedding_{index}"] = np.zeros(rt.TEMB_ELEMENTS, dtype=np.float32)

    for name, values in constants.items():
        values.astype("<f4").tofile(tmp_path / name)

    plan = StubPlan(buffers=sizes, steps=plan_json["steps"],
                    result=plan_json["result"], constants_dir=tmp_path)
    result = plan.run(
        {
            "patches": np.linspace(0, 1, sizes["patches"], dtype=np.float32),
            "state": np.arange(rt.STATE_DIM, dtype=np.float32),
            "noise": np.zeros(sizes["noise"], dtype=np.float32),
        },
        graph_fn=graph_fn,
    )
    assert result.size == GROOT_SPEC.wire.response_elements == rt.CHUNK * rt.ACTION_DIM
    assert len(seen) == 26 + 1 + rt.DENOISE_STEPS * 18


# ------------------------------------------------------------- the torch side

torch = pytest.importorskip("torch", reason="graphs.py is the only torch-side module")
nn = torch.nn
graphs = pytest.importorskip("polima.policies.groot.graphs")


def test_the_client_and_the_export_patchify_identically():
    """A disagreement here would calibrate on one tensor and infer on another.

    `patchify` runs at export against NCHW pixels; `patchify_for_wire` runs in
    the live client against an HWC frame. They must produce the same 324x588.
    """
    pixels = torch.randn(1, 3, rt.IMAGE_SIDE, rt.IMAGE_SIDE)
    from_export = graphs.patchify(pixels, rt.PATCH)[0].numpy()
    from_wire = graphs.patchify_for_wire(pixels[0].permute(1, 2, 0).numpy())
    assert from_export.shape == (rt.VISION_WIDTH, rt.PATCH_CHANNELS)
    assert np.array_equal(from_export, from_wire)


def test_the_opcode_reproduces_torch_pixel_unshuffle_at_eagle_size():
    """The board's fold and the export's fold, on the real 18x18x1152 map."""
    hidden = torch.randn(1, rt.VISION_WIDTH, rt.VISION_CHANNELS)
    expected = graphs.pixel_unshuffle_tokens(hidden, rt.PATCH_SIDE, rt.UNSHUFFLE)
    assert tuple(expected.shape) == (1, rt.CONNECTOR_WIDTH, rt.CONNECTOR_CHANNELS)
    plan = StubPlan(
        buffers={"src": hidden.numel(), "dst": expected.numel()},
        steps=[{"op": "pixel_unshuffle", "out": "dst",
                "args": {"src": "src", "grid": rt.PATCH_SIDE,
                         "channels": rt.VISION_CHANNELS, "factor": rt.UNSHUFFLE}}],
        result="dst",
    )
    assert np.array_equal(plan.run({"src": hidden.numpy().ravel()}),
                          expected.numpy().ravel())


def test_vision_post_norm_matmuls_equal_the_layernorm_they_replace():
    layer_norm = nn.LayerNorm(rt.VISION_CHANNELS)
    nn.init.normal_(layer_norm.weight)
    nn.init.normal_(layer_norm.bias)
    hidden = torch.randn(1, rt.VISION_WIDTH, rt.VISION_CHANNELS)
    assert torch.allclose(graphs.VisionPostNorm(layer_norm)(hidden),
                          layer_norm(hidden), atol=2e-5)


def test_output_norm_equals_rms_then_layernorm():
    """The fused stage is Qwen's final RMSNorm followed by the head's vlln."""

    class Rms(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(width))
            self.variance_epsilon = 1e-6

        def forward(self, hidden):
            scale = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            return hidden * scale * self.weight

    rms = Rms(rt.LANGUAGE_CHANNELS)
    vlln = nn.LayerNorm(rt.LANGUAGE_CHANNELS)
    nn.init.normal_(vlln.weight)
    nn.init.normal_(vlln.bias)
    hidden = torch.randn(1, rt.SEQUENCE, rt.LANGUAGE_CHANNELS)
    assert torch.allclose(graphs.OutputNorm(rms, vlln)(hidden), vlln(rms(hidden)), atol=5e-5)


def test_the_norm_rewrites_emit_no_reduction_operators(tmp_path):
    """The whole point of the matmul form: Modalix cannot place the reductions."""
    onnx = pytest.importorskip("onnx")
    runtime = pytest.importorskip("onnxruntime")

    layer_norm = nn.LayerNorm(rt.VISION_CHANNELS)
    hidden = torch.randn(1, rt.VISION_WIDTH, rt.VISION_CHANNELS)
    path = tmp_path / "eagle_vision_post_norm.onnx"
    expected = graphs._export_onnx(graphs.VisionPostNorm(layer_norm), (hidden,), path,
                                   ("hidden",), ("output",))
    onnx.checker.check_model(str(path))
    operators = {node.op_type for node in onnx.load(str(path)).graph.node}
    assert not operators & {"ReduceMean", "ReduceSumSquare", "LayerNormalization"}
    session = runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert np.allclose(session.run(None, {"hidden": hidden.numpy()})[0], expected, atol=1e-5)


def test_additive_masks_are_complementary_over_the_image_window():
    image = torch.zeros(1, rt.SEQUENCE, dtype=torch.bool)
    window = slice(rt.IMAGE_TOKEN_START, rt.IMAGE_TOKEN_START + rt.CONNECTOR_WIDTH)
    image[0, window] = True
    attention = torch.ones(1, rt.SEQUENCE, dtype=torch.bool)
    image_additive, text_additive = graphs.additive_masks(image, attention)
    assert bool((image_additive[0, window] == 0).all())
    assert bool((image_additive[0, :rt.IMAGE_TOKEN_START] == -10000).all())
    # Every position is open in exactly one of the two.
    assert bool(((image_additive == 0) ^ (text_additive == 0)).all())


def test_normalization_trims_to_real_joints_and_survives_a_still_joint(tmp_path):
    """GR00T stores statistics over the padded 128 lane, per embodiment.

    A joint that never moved has std 0. Dividing by it on the board produces
    inf with no traceback to read, so it is clamped at export.
    """
    checkpoint = tmp_path / "ckpt"
    (checkpoint / "experiment_cfg").mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    statistics = {"new_embodiment": {"statistics": {
        "state": {"mean": [1.0] * 6 + [0.0] * 122,
                  "std": [2.0] * 5 + [0.0] + [1.0] * 122},
        "action": {"mean": [3.0] * 128, "std": [4.0] * 128},
    }}}
    (checkpoint / "experiment_cfg" / "metadata.json").write_text(
        json.dumps({"embodiments": statistics}), encoding="utf-8")

    loaded = np.load(graphs.write_normalization(checkpoint, [], tmp_path / "stats.npz"))
    assert all(loaded[key].shape == (rt.STATE_DIM,) for key in loaded)
    assert loaded["state_std"].tolist() == [2.0] * 5 + [1.0]
    # Written under the names runtime.SIDECARS lists, so packing needs no rename.
    written = {path.name for path in (tmp_path / "constants").iterdir()}
    assert written == {"state_mean", "state_std", "action_mean", "action_std"}
    assert np.fromfile(tmp_path / "constants" / "action_std", dtype="<f4").tolist() == [4.0] * 6


def test_validator_flags_a_checkpoint_with_no_statistics(tmp_path):
    checkpoint = tmp_path / "ckpt"
    (checkpoint / "experiment_cfg").mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    assert len(graphs.validate_checkpoint(checkpoint)) == 1
    (checkpoint / "experiment_cfg" / "metadata.json").write_text("{}", encoding="utf-8")
    assert graphs.validate_checkpoint(checkpoint) == []
