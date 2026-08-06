"""Dataset contract validation, driven by DatasetContract instead of hardcoded.

Generalizes ACT/scripts/validate_act_datasets.py -- the only dataset validator in
the legacy tree. Its SO-101 assertions (six named joints, exactly two camera
keys, HWC [480,640,3], 30 fps, single task) are inlined constants there, so
SmolVLA and GR00T get nothing but `[[ -f meta/info.json ]]`. Making the rules
data means every policy is validated by the same code.

Two behavioural differences from the legacy script, both deliberate:

  * it collects ALL violations instead of raising on the first, so one run tells
    you everything wrong with a dataset; and
  * pyarrow is optional. When it is missing, task checking degrades to a
    recorded "skipped" rather than an ImportError, because this module is
    imported on the board where pyarrow is not installed.

`polima data validate` reproduces the legacy verdict exactly; see
tests/unit/test_contract.py and the Phase-0 parity proof.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polima.policies.base import DatasetContract
from polima.util.jsonio import read_json


def canonical_task(value: str) -> str:
    """Casefold, collapse whitespace, strip trailing sentence punctuation.

    Verbatim from ACT/scripts/validate_act_datasets.py:20-21 -- byte-identical
    behaviour matters, since it decides whether two datasets may be combined.
    """
    return re.sub(r"[.!?]+$", "", " ".join(value.casefold().split())).strip()


@dataclass
class DatasetReport:
    root: str
    ok: bool = True
    violations: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    task_labels: list[str] = field(default_factory=list)
    episodes: int | None = None
    frames: int | None = None
    fps: int | None = None
    codebase_version: str | None = None
    camera_keys: list[str] = field(default_factory=list)
    task_check: str = "ok"          # ok | skipped-no-pyarrow | skipped-missing

    def fail(self, message: str) -> None:
        self.ok = False
        self.violations.append(message)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "ok": self.ok,
            "violations": self.violations,
            "tasks": self.tasks,
            "task_labels": self.task_labels,
            "episodes": self.episodes,
            "frames": self.frames,
            "fps": self.fps,
            "codebase_version": self.codebase_version,
            "camera_keys": self.camera_keys,
            "task_check": self.task_check,
        }


class DatasetError(ValueError):
    """A dataset violates its policy's contract."""


def validate(
    root: str | Path,
    contract: DatasetContract,
    *,
    allow_mixed_tasks: bool = False,
) -> DatasetReport:
    """Check one dataset root against `contract`. Never raises for contract
    violations -- inspect `report.ok`. Raises only when the path is unusable."""
    root = Path(root).resolve()
    report = DatasetReport(root=str(root))

    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    if not info_path.is_file() or not (root / "data").is_dir():
        report.fail(f"{root}: incomplete LeRobot dataset (missing meta/info.json or data/)")
        return report

    try:
        info = read_json(info_path)
    except Exception as exc:  # noqa: BLE001 - report, don't explode
        report.fail(f"{root}: cannot read meta/info.json: {exc}")
        return report

    features: dict[str, Any] = info.get("features") or {}
    report.episodes = info.get("total_episodes")
    report.frames = info.get("total_frames")
    report.fps = info.get("fps")
    report.codebase_version = info.get("codebase_version")

    _check_vectors(report, root, features, contract)
    _check_cameras(report, root, features, contract)

    if contract.fps and info.get("fps") != contract.fps:
        report.fail(f"{root}: requires {contract.fps} fps, found {info.get('fps')!r}")

    if contract.codebase_version and report.codebase_version:
        if report.codebase_version != contract.codebase_version:
            report.fail(
                f"{root}: codebase_version {report.codebase_version!r}, "
                f"contract requires {contract.codebase_version!r}"
                + (" (a converter is configured)" if contract.converter else "")
            )

    for relative in contract.extra_meta_files:
        if not (root / relative).is_file():
            report.fail(f"{root}: missing required {relative}")

    _check_tasks(report, root, tasks_path, contract, allow_mixed_tasks)
    return report


def validate_all(
    roots: list[str | Path],
    contract: DatasetContract,
    *,
    allow_mixed_tasks: bool = False,
) -> tuple[list[DatasetReport], list[str]]:
    """Validate several roots and cross-check that they describe one task.

    Mirrors main() in validate_act_datasets.py, which rejects combining datasets
    with different task labels unless --allow-mixed-tasks is given (ACT has no
    language conditioning, so mixing tasks silently degrades the policy).
    """
    reports = [validate(root, contract, allow_mixed_tasks=allow_mixed_tasks) for root in roots]
    tasks = sorted({task for report in reports for task in report.tasks})
    if contract.single_task and len(tasks) > 1 and not allow_mixed_tasks:
        for report in reports:
            report.fail(f"datasets describe different tasks: {tasks}")
    return reports, tasks


def require_valid(
    roots: list[str | Path],
    contract: DatasetContract,
    *,
    allow_mixed_tasks: bool = False,
) -> list[DatasetReport]:
    """validate_all, raising DatasetError on any violation."""
    reports, _ = validate_all(roots, contract, allow_mixed_tasks=allow_mixed_tasks)
    broken = [r for r in reports if not r.ok]
    if broken:
        lines = [f"  {v}" for report in broken for v in report.violations]
        raise DatasetError("dataset contract violated:\n" + "\n".join(lines))
    return reports


# ------------------------------------------------------------------ internals


def _check_vectors(
    report: DatasetReport, root: Path, features: dict, contract: DatasetContract
) -> None:
    for key, dim in (("observation.state", contract.state_dim), ("action", contract.action_dim)):
        feature = features.get(key)
        if feature is None:
            report.fail(f"{root}: missing feature {key!r}")
            continue
        if list(feature.get("shape") or []) != [dim]:
            report.fail(f"{root}: {key} shape must be [{dim}], got {feature.get('shape')!r}")
        names = feature.get("names")
        if contract.state_names and list(names or []) != list(contract.state_names):
            report.fail(
                f"{root}: {key} names must be {list(contract.state_names)}, got {names!r}"
            )


def _check_cameras(
    report: DatasetReport, root: Path, features: dict, contract: DatasetContract
) -> None:
    found = {key for key in features if key.startswith("observation.images.")}
    report.camera_keys = sorted(found)
    expected = set(contract.camera_keys)
    if found != expected:
        report.fail(f"{root}: expected image keys {sorted(expected)}, got {sorted(found)}")
    shape = list(contract.camera_shape)
    for key in sorted(expected & found):
        if list(features[key].get("shape") or []) != shape:
            report.fail(
                f"{root}: {key} must have HWC shape {shape}, got {features[key].get('shape')!r}"
            )


def _check_tasks(
    report: DatasetReport,
    root: Path,
    tasks_path: Path,
    contract: DatasetContract,
    allow_mixed_tasks: bool,
) -> None:
    if not tasks_path.is_file():
        report.task_check = "skipped-missing"
        report.fail(f"{root}: missing meta/tasks.parquet")
        return
    try:
        import pyarrow.parquet as parquet  # noqa: PLC0415 - absent on the board
    except ImportError:
        report.task_check = "skipped-no-pyarrow"
        return

    table = parquet.read_table(tasks_path, columns=["task"])
    labels = [str(value) for value in table.column("task").to_pylist()]
    canonicalize = canonical_task
    if contract.task_canonicalizer:
        from polima.policies.base import resolve

        canonicalize = resolve(contract.task_canonicalizer)

    report.task_labels = labels
    report.tasks = sorted({canonicalize(value) for value in labels})
    if contract.single_task and len(report.tasks) != 1 and not allow_mixed_tasks:
        report.fail(f"{root}: policy is single-task; found {labels}")
