"""Importing a legacy compiler build tree.

The decoy tests are the point. The real ACT build carries 16 ELFs across 11
directories in `retained/`, six of them abandoned experiments:

    vision_backbone/                    <- shipped
    vision_backbone_rank3/              <- byte-identical to the above
    vision_backbone_rejected_rank4/     <- different, and wrong
    decoder_action_tail/                <- shipped
    decoder_action_tail_v2/             <- byte-identical to the above
    decoder_action_tail_rejected_apu/   <- different, and wrong

Picking the wrong one produces a bundle that deploys cleanly and returns
garbage, which is the worst possible failure mode.
"""

from __future__ import annotations

import json

import pytest

from polima.bundle import retained
from polima.bundle.import_legacy import _declared_graphs, detect, resolve_elfs
from polima.policies.act import ACT_SPEC

GRAPHS = list(ACT_SPEC.compile.names)


def make_elf(path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def make_retained(root, layout: dict[str, bytes]):
    """layout: variant_dir -> elf payload. The graph name is the variant's prefix."""
    for variant, payload in layout.items():
        graph = variant
        for candidate in sorted(GRAPHS, key=len, reverse=True):
            if variant.startswith(candidate):
                graph = candidate
                break
        make_elf(root / "retained" / variant / f"{graph}_stage1_mla.elf", payload)
    return root


def test_exact_directory_wins_over_decoys(tmp_path):
    make_retained(tmp_path, {
        "vision_backbone": b"SHIPPED",
        "vision_backbone_rank3": b"SHIPPED",              # identical twin
        "vision_backbone_rejected_rank4": b"WRONG",
    })
    chosen = retained.select(tmp_path / "retained", "vision_backbone")
    assert chosen.variant == "vision_backbone"
    assert chosen.path.read_bytes() == b"SHIPPED"


def test_rejected_variants_are_flagged(tmp_path):
    make_retained(tmp_path, {
        "decoder_action_tail_rejected_apu": b"WRONG",
        "decoder_action_tail_v2": b"SHIPPED",
    })
    variants = retained.find_variants(tmp_path / "retained", "decoder_action_tail")
    by_name = {c.variant: c for c in variants}
    assert by_name["decoder_action_tail_rejected_apu"].rejected
    assert not by_name["decoder_action_tail_v2"].rejected


def test_single_viable_variant_is_accepted(tmp_path):
    make_retained(tmp_path, {
        "decoder_action_tail_rejected_apu": b"WRONG",
        "decoder_action_tail_v2": b"SHIPPED",
    })
    chosen = retained.select(tmp_path / "retained", "decoder_action_tail")
    assert chosen.variant == "decoder_action_tail_v2"


def test_ambiguity_raises_rather_than_guessing(tmp_path):
    """Two plausible variants and no exact name: refuse, don't pick one."""
    make_retained(tmp_path, {
        "decoder_action_tail_v2": b"ONE",
        "decoder_action_tail_v3": b"TWO",
    })
    with pytest.raises(ValueError, match="ambiguous"):
        retained.select(tmp_path / "retained", "decoder_action_tail")


def test_prefix_collision_does_not_match(tmp_path):
    """`encoder_layer_01` must not be offered as a variant of `encoder_layer_0`."""
    make_elf(
        tmp_path / "retained" / "encoder_layer_01" / "encoder_layer_01_stage1_mla.elf", b"X"
    )
    assert retained.find_variants(tmp_path / "retained", "encoder_layer_0") == []


def test_missing_graph_raises(tmp_path):
    (tmp_path / "retained").mkdir()
    with pytest.raises(FileNotFoundError):
        retained.select(tmp_path / "retained", "vision_backbone")


def test_declared_graphs_drops_tensor_prepared_intermediates():
    """input_contract.json lists 12 entries for a 6-graph policy."""
    contract = {"graphs": sorted(
        [f"{g}.onnx" for g in GRAPHS] + [f"{g}_tensor_prepared.onnx" for g in GRAPHS]
    )}
    assert _declared_graphs(contract) == sorted(GRAPHS)


def test_deployed_tree_is_preferred_over_retained(tmp_path):
    """models_uncompressed/ is what the legacy deploy actually shipped."""
    make_retained(tmp_path, {"vision_backbone": b"FROM_RETAINED"})
    (tmp_path / "input_contract.json").write_text(json.dumps({
        "format": "act-modalix-v1",
        "graphs": [f"{g}.onnx" for g in GRAPHS],
        "checkpoint": "/x/checkpoints/100000/pretrained_model",
        "dataset_root": "/ml_datasets/rcwb_f_t",
    }))
    for graph in GRAPHS:
        make_elf(
            tmp_path / "models_uncompressed" / graph / "share" / f"{graph}_stage1_mla.elf",
            b"FROM_DEPLOYED",
        )

    build = detect(tmp_path)
    elfs = resolve_elfs(build, ACT_SPEC)
    assert elfs["vision_backbone"].path.read_bytes() == b"FROM_DEPLOYED"
    assert elfs["vision_backbone"].variant == "deployed"


def test_detect_reads_contract_metadata(tmp_path):
    (tmp_path / "input_contract.json").write_text(json.dumps({
        "format": "act-modalix-v1",
        "checkpoint": "/a/b/checkpoints/100000/pretrained_model",
        "dataset_root": "/ml_datasets/rcwb_f_t",
        "graphs": [f"{g}.onnx" for g in GRAPHS],
    }))
    build = detect(tmp_path)
    assert build.format == "act-modalix-v1"
    assert build.dataset_name == "rcwb_f_t"
    assert build.steps == 100000
    assert set(build.graphs) == set(GRAPHS)


def test_detect_rejects_unrecognised_tree(tmp_path):
    with pytest.raises(ValueError, match="no recognisable manifest"):
        detect(tmp_path)


def test_detect_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect(tmp_path / "absent")
