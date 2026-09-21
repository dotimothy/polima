"""GR00T N1.7's spec and execution plan.

N1.7 is cut into 46 graphs, with host arithmetic between four vision chains
per image, two deepstack-fed language graphs and a device-resident language
chain. Most ways to get that wrong still produce finite, plausible actions -- a
deepstack tap moved by one block, image tokens packed at the wrong run, a mask
fed to the wrong pair -- so the structure is pinned here, and the host
arithmetic is replayed with the MLA stubbed.

The model-dependent claims (per-image attention, the tap positions, the token
runs) are re-checked against the live checkpoint by `graphs.trace` and
`graphs.export_all` on every export; these tests pin the plan against the spec.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from polima.config.base import DEFAULT_PORTS
from polima.policies.groot17 import GROOT17_SPEC
from polima.policies.groot17 import runtime as rt
from polima.policies.registry import get_policy
from polima.wire.server_stub import StubPlan

# ------------------------------------------------------------------ the spec


def test_spec_registers_with_forty_six_graphs():
    spec = get_policy("groot17")
    assert spec is GROOT17_SPEC
    names = spec.compile.names
    assert len(names) == 46
    assert sum(name.startswith("vit_") for name in names) == 17
    assert sum(name.startswith("llm_") for name in names) == 8
    assert sum(name.startswith("vl_self_attention_") for name in names) == 2
    assert sum(name.startswith("dit_blocks_") for name in names) == 16


def test_vision_chains_split_exactly_at_the_deepstack_taps():
    """A tap after block b must close a chain at the pair ending in b."""
    chains = rt.VISION_CHAINS
    assert len(chains) == len(rt.DEEPSTACK_BLOCKS) + 1
    for chain, block in zip(chains, rt.DEEPSTACK_BLOCKS):
        assert chain[-1].endswith(f"_{block:02d}")
    flattened = [name for chain in chains for name in chain]
    assert flattened == ["vit_patch", *rt.vision_block_names()]


def test_language_chain_is_everything_after_the_deepstack_layers():
    assert rt.LLM_ENTRY == ("llm_00_01", "llm_02_03")
    assert rt.LANGUAGE_CHAIN == (*rt.llm_names()[2:], *rt.vl_self_attention_names())
    # Deepstack adds at layers 0, 1, 2 -- the entry graphs take exactly those.
    taps = [tap for name in rt.LLM_ENTRY for tap in rt.LLM_ENTRY_DEEPSTACK[name]]
    assert taps == list(range(len(rt.DEEPSTACK_BLOCKS)))


def test_chained_graphs_expose_a_flat_hwc_boundary():
    for name in (*[n for chain in rt.VISION_CHAINS for n in chain], *rt.LANGUAGE_CHAIN):
        graph = GROOT17_SPEC.graph(name)
        assert graph.external_dram_layout == "HWC", name
        assert not graph.mla_tessellation, name
        assert graph.promote_rank3_hwc, name
        assert graph.layout == "NHWC", name


def test_unchained_outputs_are_hwc16_for_the_host_to_unpack():
    chained = {n for chain in rt.VISION_CHAINS for n in chain} | set(rt.LANGUAGE_CHAIN)
    for graph in GROOT17_SPEC.compile.graphs:
        if graph.name not in chained:
            assert graph.outputs[0].dram_layout == "hwc16", graph.name


def test_dit_inputs_are_declared_in_the_order_the_plan_feeds_them():
    """N1.6 exported these four in a different order than it declared them."""
    for name in rt.block_names():
        assert [t.name for t in GROOT17_SPEC.graph(name).inputs] == [
            "hidden", "temb", "backbone_features", "additive_mask"]
    for step in plan_dict()["steps"]:
        if step["op"] == "run_elf" and step["args"]["graph"].startswith("dit_blocks_"):
            source, temb, backbone, mask = step["args"]["in"]
            assert source.startswith("hidden_") and temb.startswith("temb_")
            assert backbone == "backbone_features" and mask.endswith("_mask")


def test_wire_magic_is_gr17_and_the_port_is_consistent():
    assert rt.WIRE_MAGIC.to_bytes(4, "little") == b"GR17"
    assert GROOT17_SPEC.wire.default_port == rt.DEFAULT_PORT == DEFAULT_PORTS["groot17"]
    assert len(set(DEFAULT_PORTS.values())) == len(DEFAULT_PORTS)


def test_visual_runs_tile_the_sequence_with_two_token_separators():
    runs = rt.VISUAL_RUNS
    assert len(runs) == rt.IMAGES
    assert all(length == rt.MERGED_TOKENS for _, length in runs)
    for (start, length), (following, _) in pairwise(runs):
        assert following - (start + length) == 2
    assert runs[-1][0] + runs[-1][1] < rt.SEQUENCE
    assert rt.IMAGES * rt.MERGED_TOKENS == 256


def test_geometry_is_internally_consistent():
    assert rt.PATCH_TOKENS == rt.PATCH_SIDE ** 2 == 256
    assert rt.PATCH_CHANNELS == 3 * rt.TEMPORAL_PATCH * rt.PATCH ** 2 == 1536
    assert rt.MERGED_TOKENS * rt.MERGE ** 2 == rt.PATCH_TOKENS
    assert rt.STATE_DIM == 17 and rt.CHUNK == 40 and rt.ACTION_LANE == 132
    assert rt.HIDDEN_WIDTH == rt.CHUNK + 1


def test_unchained_graphs_expose_rank4_io_for_the_mpk_packager():
    """ModelSDK 2.1's MPK packager unpacks tessellated tensors as 4D ("expected 4, got 3")."""
    chained = {n for chain in rt.VISION_CHAINS for n in chain} | set(rt.LANGUAGE_CHAIN)
    for graph in GROOT17_SPEC.compile.graphs:
        tensors = (*graph.inputs, *graph.outputs)
        if graph.name in chained:
            assert all(len(t.shape) == 3 for t in tensors), graph.name
            assert graph.promote_rank3_hwc, graph.name
        else:
            assert all(len(t.shape) == 4 and t.shape[:2] == (1, 1) for t in tensors), graph.name
            assert graph.mla_tessellation and not graph.promote_rank3_hwc, graph.name


# ---------------------------------------------------------------- the plan


def plan_dict() -> dict:
    plan = GROOT17_SPEC.build_runtime_plan()
    return {
        "buffers": dict(plan.buffers),
        "steps": [{"op": s.op, "out": s.out, "args": dict(s.args)} for s in plan.steps],
        "result": plan.result,
    }


def graph_calls() -> list[str]:
    calls = []
    for step in plan_dict()["steps"]:
        if step["op"] == "run_elf":
            calls.append(step["args"]["graph"])
        elif step["op"] == "run_elf_chain":
            calls.extend(step["args"]["graphs"])
    return calls


def test_plan_runs_each_graph_the_right_number_of_times():
    calls = graph_calls()
    for name in ("vit_patch", *rt.vision_block_names(), *rt.deepstack_names(), "vit_merger"):
        assert calls.count(name) == rt.IMAGES, name
    for name in (*rt.llm_names(), *rt.vl_self_attention_names(), "state_project"):
        assert calls.count(name) == 1, name
    for name in ("action_project", *rt.block_names(), "action_tail"):
        assert calls.count(name) == rt.DENOISE_STEPS, name
    assert len(calls) == rt.IMAGES * 17 + 10 + 1 + rt.DENOISE_STEPS * 18


def test_blocks_alternate_text_and_image_masks_by_pair_parity():
    """Even pairs attend to text, odd pairs to images -- attend_text_every_n_blocks=2."""
    for step in plan_dict()["steps"]:
        if step["op"] == "run_elf" and step["args"]["graph"].startswith("dit_blocks_"):
            start = int(step["args"]["graph"].split("_")[2])
            expected = "text_mask" if start % 4 == 0 else "image_mask"
            assert step["args"]["in"][3] == expected, step["args"]["graph"]


def test_euler_adds_because_groot_integrates_forward():
    eulers = [s for s in plan_dict()["steps"] if s["op"] == "euler"]
    assert len(eulers) == rt.DENOISE_STEPS
    assert all(s["args"]["scalar"] == -rt.DT for s in eulers)


def test_state_is_normalized_before_padding_not_after():
    steps = plan_dict()["steps"]
    normalize = next(i for i, s in enumerate(steps) if s["op"] == "normalize")
    pad = next(i for i, s in enumerate(steps) if s["out"] == "state_lane")
    assert normalize < pad
    assert steps[normalize]["args"]["src"] == "state"


def test_no_elf_reads_and_writes_the_same_buffer():
    for step in plan_dict()["steps"]:
        if step["op"] in ("run_elf", "run_elf_chain"):
            assert step["out"] not in step["args"]["in"], step


def test_image_tokens_and_deepstack_land_on_the_same_runs():
    steps = {s["out"]: s for s in plan_dict()["steps"] if s["op"] == "pack"}
    expected = [(start * rt.LANGUAGE_CHANNELS, length * rt.LANGUAGE_CHANNELS)
                for start, length in rt.VISUAL_RUNS]
    language = [(p["dst_offset"], p["count"])
                for p in steps["language_embeddings"]["args"]["parts"] if not p.get("sidecar")]
    assert language == expected
    for tap in range(len(rt.DEEPSTACK_BLOCKS)):
        parts = steps[f"deepstack_full_{tap}"]["args"]["parts"]
        assert [(p["dst_offset"], p["count"]) for p in parts] == expected
        assert [p["src"] for p in parts] == [f"deepstack_{tap}_{i}" for i in range(rt.IMAGES)]


def test_plan_replays_end_to_end_with_the_mla_stubbed(tmp_path):
    """Buffer plumbing, start to finish, with per-timestep action statistics.

    Every ELF is a deterministic function of its input, so this says nothing
    about the model -- only that all 46 graphs receive the element counts the
    spec declares and the result reaches the wire's shape.
    """
    plan_json = plan_dict()
    sizes = plan_json["buffers"]
    seen: list[str] = []

    def graph_fn(name: str, values: np.ndarray) -> np.ndarray:
        seen.append(name)
        graph = GROOT17_SPEC.graph(name)
        expected = sum(tensor.elements for tensor in graph.inputs)
        assert values.size == expected, f"{name}: got {values.size}, spec says {expected}"
        out = sum(tensor.elements for tensor in graph.outputs)
        return np.full(out, float(values.sum() % 7.0), dtype=np.float32)

    constants = {
        "prompt_embedding": np.zeros(rt.BACKBONE_ELEMENTS, dtype=np.float32),
        "image_additive_mask": np.zeros(rt.MASK_ELEMENTS, dtype=np.float32),
        "text_additive_mask": np.zeros(rt.MASK_ELEMENTS, dtype=np.float32),
        "state_mean": np.zeros(rt.STATE_DIM, dtype=np.float32),
        "state_std": np.ones(rt.STATE_DIM, dtype=np.float32),
        # One row of 17 per horizon step, as write_normalization lays them out.
        "action_mean": np.arange(rt.RESPONSE_ELEMENTS, dtype=np.float32),
        "action_std": np.ones(rt.RESPONSE_ELEMENTS, dtype=np.float32),
    }
    for index in range(rt.DENOISE_STEPS):
        constants[f"tau_embedding_{index}"] = np.zeros(rt.TAU_ELEMENTS, dtype=np.float32)
        constants[f"timestep_embedding_{index}"] = np.zeros(rt.TEMB_ELEMENTS, dtype=np.float32)
    for name, values in constants.items():
        values.astype("<f4").tofile(tmp_path / name)

    inputs = {f"patches_{i}": np.linspace(0, 1, sizes[f"patches_{i}"], dtype=np.float32)
              for i in range(rt.IMAGES)}
    inputs["state"] = np.arange(rt.STATE_DIM, dtype=np.float32)
    inputs["noise"] = np.zeros(sizes["noise"], dtype=np.float32)

    plan = StubPlan(buffers=sizes, steps=plan_json["steps"],
                    result=plan_json["result"], constants_dir=tmp_path)
    result = plan.run(inputs, graph_fn=graph_fn)
    assert result.size == GROOT17_SPEC.wire.response_elements == rt.CHUNK * rt.ACTION_DIM
    assert len(seen) == len(graph_calls())


def test_action_denormalize_clips_before_scaling_like_groot(tmp_path):
    """`unnormalize_values_minmax` clips to [-1, 1] first; the plan must too."""
    step = next(s for s in plan_dict()["steps"] if s["op"] == "denormalize")
    assert (step["args"]["clip_min"], step["args"]["clip_max"]) == (-1.0, 1.0)

    np.full(3, 10.0, dtype="<f4").tofile(tmp_path / "mean")
    np.full(3, 2.0, dtype="<f4").tofile(tmp_path / "std")
    plan = StubPlan(
        buffers={"x": 3, "y": 3},
        steps=[{"op": "denormalize", "out": "y",
                "args": {"src": "x", "mean": "mean", "std": "std",
                         "clip_min": -1.0, "clip_max": 1.0}}],
        result="y", constants_dir=tmp_path)
    result = plan.run({"x": np.array([-3.0, 0.5, 3.0], dtype=np.float32)})
    np.testing.assert_allclose(result, [8.0, 11.0, 12.0])


# ------------------------------------------------------ torch-side helpers


def test_patchify_for_wire_matches_the_qwen_image_processor():
    """The client's torch-free patchify must reproduce Qwen's flatten order exactly."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from polima.policies.groot17.graphs import patchify_for_wire

    processor = transformers.Qwen2VLImageProcessor(
        do_resize=False, image_mean=list(rt.IMAGE_MEAN), image_std=list(rt.IMAGE_STD),
        patch_size=rt.PATCH, temporal_patch_size=rt.TEMPORAL_PATCH, merge_size=rt.MERGE,
    )
    image = np.random.default_rng(0).integers(0, 256, (rt.IMAGE_SIDE, rt.IMAGE_SIDE, 3),
                                              dtype=np.uint8)
    reference = processor(images=[image], return_tensors="np")
    assert reference["image_grid_thw"].tolist() == [[1, rt.PATCH_SIDE, rt.PATCH_SIDE]]
    np.testing.assert_allclose(patchify_for_wire(image), reference["pixel_values"], atol=1e-5)


def test_patch_embedding_conv3d_is_the_linear_layer_it_exports_as():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from polima.policies.groot17.graphs import VisionPatch

    conv = torch.nn.Conv3d(3, 8, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=True)
    vision = SimpleNamespace(patch_embed=SimpleNamespace(proj=conv))
    position = torch.zeros(4, 8)
    patches = torch.randn(1, 4, rt.PATCH_CHANNELS)
    with torch.no_grad():
        reference = conv(patches.view(-1, 3, 2, 16, 16)).view(1, 4, 8)
        exported = VisionPatch(vision, position)(patches)
    torch.testing.assert_close(exported, reference, atol=1e-5, rtol=1e-5)


def test_state_project_clips_like_the_processor_before_encoding():
    """The plan's `normalize` is affine; GR00T clips to [-1, 1] afterwards.

    Without the clip inside state_project, any state outside the q01..q99 range
    reaches the encoder unclipped -- finite, plausible, and wrong.
    """
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from polima.policies.groot17.graphs import StateProject

    head = SimpleNamespace(state_encoder=lambda state, embodiment: state)
    lane = torch.zeros(1, 1, rt.STATE_LANE)
    lane[0, 0, :4] = torch.tensor([-2.26, 0.5, 1.35, -1.0])
    projected = StateProject(head)(lane)
    torch.testing.assert_close(projected[0, 0, :4], torch.tensor([-1.0, 0.5, 1.0, -1.0]))
    assert not projected[0, 0, 4:].any(), "padding must stay zero"


def test_percentile_moments_reproduce_grooTs_min_max_scaling():
    pytest.importorskip("torch")
    from polima.policies.groot17.graphs import percentile_moments

    block = {"q01": [-2.0, 0.0, 5.0], "q99": [2.0, 4.0, 5.0]}
    mean, std = percentile_moments(block)
    x = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    low, high = np.asarray(block["q01"]), np.asarray(block["q99"])
    groot = np.where(high > low, 2 * (x - low) / np.where(high > low, high - low, 1) - 1, 0.0)
    np.testing.assert_allclose((x - mean) / std, groot, atol=1e-6)


def test_scatter_runs_places_each_image_on_its_run():
    pytest.importorskip("torch")
    from polima.policies.groot17.graphs import scatter_runs

    tokens = np.arange(rt.IMAGES * rt.MERGED_TOKENS * 3, dtype=np.float32).reshape(-1, 3)
    full = scatter_runs(tokens)
    assert full.shape == (1, rt.SEQUENCE, 3)
    per_image = tokens.reshape(rt.IMAGES, rt.MERGED_TOKENS, 3)
    for image, (start, length) in enumerate(rt.VISUAL_RUNS):
        np.testing.assert_array_equal(full[0, start:start + length], per_image[image])
    mask = np.ones(rt.SEQUENCE, dtype=bool)
    for start, length in rt.VISUAL_RUNS:
        mask[start:start + length] = False
    assert not full[0, mask].any()


def test_rank4_wrapper_is_a_numerical_no_op():
    torch = pytest.importorskip("torch")
    from polima.policies.groot17.graphs import Rank4, _rank4_shape

    class Blend(torch.nn.Module):
        def forward(self, hidden, temb, mask):
            return hidden * 2 + temb[:, None, :hidden.shape[-1]] + mask[:, :1, None]

    hidden, temb, mask = torch.randn(1, 5, 8), torch.randn(1, 16), torch.randn(1, 7)
    reference = Blend()(hidden, temb, mask)
    wrapped = Rank4(Blend(), [hidden.shape, temb.shape, mask.shape])
    output = wrapped(*(item.reshape(_rank4_shape(item.shape)) for item in (hidden, temb, mask)))
    assert output.shape == (1, 1, 5, 8)
    torch.testing.assert_close(output.reshape(reference.shape), reference)


def test_final_merger_norm_moved_into_the_last_vit_pair_is_exact():
    """vit_merger's pre-norm runs at the end of vit_blocks_22_23 instead.

    ModelSDK 2.1 cannot place a LayerNorm that feeds the 256x1024 -> 64x4096
    fold, so the norm moved one graph upstream. The composition must be the
    same function, and only a pre-shuffle norm may move.
    """
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from polima.policies.groot17.graphs import Merger, VisionPair

    torch.manual_seed(0)
    norm = torch.nn.LayerNorm(rt.VISION_CHANNELS, eps=1e-6)
    torch.nn.init.normal_(norm.weight)
    torch.nn.init.normal_(norm.bias)
    merger = SimpleNamespace(
        use_postshuffle_norm=False, norm=norm, act_fn=torch.nn.GELU(),
        linear_fc1=torch.nn.Linear(rt.MERGED_CHANNELS, 32),
        linear_fc2=torch.nn.Linear(32, 16),
    )
    hidden = torch.randn(1, rt.PATCH_TOKENS, rt.VISION_CHANNELS)
    angles = torch.zeros(rt.PATCH_TOKENS, 4)
    with torch.no_grad():
        reference = Merger(merger)(hidden)
        pair = VisionPair([], angles.cos(), angles.sin(), final_norm=norm)
        moved = Merger(merger, norm_upstream=True)(pair(hidden))
    assert torch.equal(moved, reference)

    deepstack = SimpleNamespace(**{**vars(merger), "use_postshuffle_norm": True})
    with pytest.raises(ValueError, match="pre-shuffle"):
        Merger(deepstack, norm_upstream=True)
