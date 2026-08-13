"""`polima clean` -- what it removes, and more importantly what it refuses to.

The whole risk here is deleting an ELF that exists nowhere else. That already
nearly happened once in this tree: four SmolVLA denoise expert ELFs, 540 MB,
lived only inside .tar.gz archives, so a blind delete of the compiled/ tree
would have destroyed them.
"""

from __future__ import annotations

import io
import tarfile

from polima.cli.clean import plan_for


def _tree(root, *, with_archive_elf=None, loose_elf=True):
    (root / "retained" / "g").mkdir(parents=True)
    if loose_elf:
        (root / "retained" / "g" / "g_stage1_mla.elf").write_bytes(b"\x7fELF" * 64)
    (root / "retained" / "g" / "g_stage1_mla.mlc").write_bytes(b"x" * 4096)
    (root / "retained" / "g" / "model_graph_json").mkdir()
    (root / "retained" / "g" / "model_graph_json" / "dump").write_bytes(b"y" * 2048)

    (root / "compiled" / "bf16").mkdir(parents=True)
    (root / "compiled" / "bf16" / "junk.bin").write_bytes(b"z" * 8192)
    if with_archive_elf:
        archive = root / "compiled" / "bf16" / "g_mpk.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            data = b"\x7fELF" * 32
            info = tarfile.TarInfo(with_archive_elf)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))

    (root / "onnx").mkdir()
    (root / "onnx" / "g.onnx").write_bytes(b"onnx" * 512)
    (root / "onnx" / "g_tensor_prepared.onnx").write_bytes(b"prep" * 512)
    (root / "calibration").mkdir()
    (root / "calibration" / "g.npz").write_bytes(b"c" * 4096)
    (root / "models_uncompressed" / "g" / "share").mkdir(parents=True)
    return root


def test_scratch_keeps_the_elf_and_the_inputs(tmp_path):
    plan = plan_for(_tree(tmp_path), "scratch")
    removed = {p.name for p in plan.paths}
    assert "g_stage1_mla.elf" not in removed
    assert plan.kept_elfs == 1
    assert "g_stage1_mla.mlc" in removed and "model_graph_json" in removed
    assert "compiled" in removed
    assert "g_tensor_prepared.onnx" in removed
    # onnx/ and calibration/ stay, so the content key is unchanged and the tree
    # still resumes rather than recompiling.
    assert "onnx" not in removed and "calibration" not in removed


def test_inputs_level_also_drops_the_export_inputs(tmp_path):
    removed = {p.name for p in plan_for(_tree(tmp_path), "inputs").paths}
    assert "onnx" in removed and "calibration" in removed


def test_all_removes_the_tree(tmp_path):
    plan = plan_for(_tree(tmp_path), "all")
    assert [p.name for p in plan.paths] == [tmp_path.name]


def test_an_archive_only_elf_is_never_deleted(tmp_path):
    """The SmolVLA near-miss: the only copy of the ELF is inside the mpk."""
    root = _tree(tmp_path, with_archive_elf="denoise_stage1_mla.elf")
    plan = plan_for(root, "scratch")
    assert "compiled" not in {p.name for p in plan.paths}
    assert any("denoise_stage1_mla.elf" in reason for reason in plan.skipped)


def test_an_archive_whose_elf_exists_outside_is_removable(tmp_path):
    """Same archive, but the ELF was already extracted -- safe to drop."""
    root = _tree(tmp_path, with_archive_elf="g_stage1_mla.elf")
    plan = plan_for(root, "scratch")
    assert "compiled" in {p.name for p in plan.paths}
    assert plan.skipped == []


def test_a_loose_orphan_elf_also_protects_its_directory(tmp_path):
    root = _tree(tmp_path, loose_elf=False)
    (root / "compiled" / "bf16" / "only_here.elf").write_bytes(b"\x7fELF")
    plan = plan_for(root, "scratch")
    assert "compiled" not in {p.name for p in plan.paths}
    assert any("only_here.elf" in reason for reason in plan.skipped)


def test_dry_run_is_the_default(tmp_path):
    from polima.cli import clean

    root = _tree(tmp_path)
    assert clean.run([str(root)]) == 0
    assert (root / "compiled").exists()      # nothing removed without --yes
