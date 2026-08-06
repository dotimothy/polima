"""Deploy logic that can be tested without a board.

The board-dependent paths are covered by the Phase-1a hardware proof; these
cover the pure logic, especially the shell-quoting and command-shape bugs that
cost real debugging time on hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from polima.config.base import BoardConfig
from polima.deploy.build import source_hash
from polima.deploy.smoke import (
    DEFAULT_COSINE_MIN,
    DEFAULT_MEAN_ABS_MAX,
    SmokeReport,
    compare,
    compare_against_reference,
)


# ------------------------------------------------------------------- config


def test_board_paths():
    board = BoardConfig(root="/media/nvme/polima")
    assert board.bundles_dir == "/media/nvme/polima/bundles"
    assert board.current_link == "/media/nvme/polima/current"
    assert board.path("bundles", "abc") == "/media/nvme/polima/bundles/abc"
    assert board.path("var/log") == "/media/nvme/polima/var/log"
    assert board.bin_dir == "/media/nvme/polima/bin"


def test_board_address_and_user():
    board = BoardConfig(host="sima@192.168.91.211")
    assert board.user == "sima"
    assert board.address == "192.168.91.211"
    assert BoardConfig(host="192.168.91.211").address == "192.168.91.211"


def test_trailing_slashes_do_not_double_up():
    board = BoardConfig(root="/media/nvme/polima/")
    assert board.path("bundles") == "/media/nvme/polima/bundles"


def test_default_build_jobs_is_not_the_legacy_two():
    """The board has 16 cores; both legacy deploy scripts hardcode -j2."""
    assert BoardConfig().build_jobs > 2


# -------------------------------------------------------------- build hashing


def test_source_hash_is_deterministic():
    assert source_hash() == source_hash()


def test_source_hash_changes_with_content(tmp_path):
    (tmp_path / "a.cpp").write_text("int main(){}")
    (tmp_path / "CMakeLists.txt").write_text("project(x)")
    first = source_hash(tmp_path)
    (tmp_path / "a.cpp").write_text("int main(){return 1;}")
    assert source_hash(tmp_path) != first


def test_source_hash_ignores_unrelated_files(tmp_path):
    (tmp_path / "a.cpp").write_text("int main(){}")
    before = source_hash(tmp_path)
    (tmp_path / "notes.md").write_text("hello")
    (tmp_path / "build.log").write_text("noise")
    assert source_hash(tmp_path) == before


# --------------------------------------------------------------------- smoke


def test_identical_arrays_pass():
    values = np.arange(600, dtype=np.float32)
    result = compare("x", values, values)
    assert result.ok
    assert result.cosine == pytest.approx(1.0)
    assert result.mean_abs == 0.0


def test_thresholds_match_the_legacy_shell_assertion():
    """compile_deploy_act_som.sh asserts cosine >= 0.999 and mean_abs <= 0.01."""
    assert DEFAULT_COSINE_MIN == 0.999
    assert DEFAULT_MEAN_ABS_MAX == 0.01


def test_small_perturbation_still_passes():
    rng = np.random.default_rng(0)
    expected = rng.standard_normal(600).astype(np.float32)
    actual = expected + rng.standard_normal(600).astype(np.float32) * 0.001
    assert compare("x", actual, expected).ok


def test_large_perturbation_fails():
    rng = np.random.default_rng(0)
    expected = rng.standard_normal(600).astype(np.float32)
    actual = expected + rng.standard_normal(600).astype(np.float32) * 0.5
    assert not compare("x", actual, expected).ok


def test_size_mismatch_fails_without_raising():
    result = compare("x", np.zeros(600, np.float32), np.zeros(300, np.float32))
    assert not result.ok
    assert "size mismatch" in result.note


def test_reference_comparison_is_far_stricter():
    """Two servers running the same ELFs on the same MLA must agree almost
    exactly; only the host-side glue could differ."""
    rng = np.random.default_rng(0)
    expected = rng.standard_normal(600).astype(np.float32)
    drifted = expected + 0.001

    assert compare("loose", drifted, expected).ok          # fine vs pytorch
    assert not compare_against_reference(drifted, expected).ok  # not fine vs a peer
    assert compare_against_reference(expected, expected).ok


def test_report_aggregates():
    report = SmokeReport()
    values = np.ones(10, dtype=np.float32)
    report.add(compare("a", values, values))
    assert report.ok
    report.add(compare("b", values, values * 5))
    assert not report.ok
    assert len(report.to_dict()["results"]) == 2


def test_summary_is_readable():
    values = np.ones(10, dtype=np.float32)
    assert "PASS" in compare("a", values, values).summary()
    assert "FAIL" in compare("b", values, values * 9).summary()
