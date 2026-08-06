"""Combine several LeRobot datasets into one.

The single home for what is currently a Python heredoc copy-pasted *verbatim*
between ACT/train_act_local.sh:128-138 and SmolVLA/train_smolvla_local.sh, the
only difference being the environment-variable prefix used to smuggle arguments
across the shell/Python boundary (ACT_SOURCE_ROOTS/ACT_SOURCE_IDS/... vs
SMOLVLA_SOURCE_ROOTS/...). GR00T has no equivalent, so it cannot combine
datasets at all today.

Calling lerobot's aggregate_datasets directly removes the env-var marshalling
and the two-way drift.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from polima.data.discover import DatasetEntry, is_dataset
from polima.util.logging import get

log = get("data.aggregate")


@dataclass(frozen=True)
class Aggregated:
    root: Path
    repo_id: str
    sources: tuple[Path, ...]
    reused: bool = False

    @property
    def name(self) -> str:
        return self.root.name


def combined_name(entries: list[DatasetEntry], *, stamp: str | None = None) -> str:
    """`<a>_<b>_combined_<timestamp>` -- the legacy naming convention."""
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    return "_".join([*(entry.name for entry in entries), "combined", stamp])


def aggregate(
    entries: list[DatasetEntry],
    *,
    output_root: str | Path,
    output_repo_id: str | None = None,
    reuse_existing: bool = True,
    dry_run: bool = False,
) -> Aggregated:
    """Merge `entries` into `output_root` via lerobot.datasets.aggregate.

    A single entry is returned as-is: aggregating one dataset would copy it for
    nothing, which the legacy scripts also avoid by branching on count.
    """
    if not entries:
        raise ValueError("nothing to aggregate")

    output_root = Path(output_root).resolve()
    repo_id = output_repo_id or f"local/{output_root.name}"
    sources = tuple(entry.root for entry in entries)

    if len(entries) == 1:
        only = entries[0]
        return Aggregated(root=only.root, repo_id=only.repo_id, sources=sources, reused=True)

    if reuse_existing and is_dataset(output_root):
        log.info("reusing existing combined dataset at %s", output_root)
        return Aggregated(root=output_root, repo_id=repo_id, sources=sources, reused=True)

    log.info(
        "aggregating %d datasets -> %s (%s)",
        len(entries), output_root, ", ".join(e.name for e in entries),
    )
    if dry_run:
        print(
            "[dry-run] aggregate_datasets("
            f"repo_ids={[e.repo_id for e in entries]}, "
            f"roots={[str(e.root) for e in entries]}, "
            f"aggr_repo_id={repo_id!r}, aggr_root={str(output_root)!r})"
        )
        return Aggregated(root=output_root, repo_id=repo_id, sources=sources)

    # Imported lazily: lerobot exists only in the training envs, and this module
    # is importable from the compiler venv and the board.
    from lerobot.datasets.aggregate import aggregate_datasets

    output_root.parent.mkdir(parents=True, exist_ok=True)
    aggregate_datasets(
        repo_ids=[entry.repo_id for entry in entries],
        roots=[entry.root for entry in entries],
        aggr_repo_id=repo_id,
        aggr_root=output_root,
    )
    if not is_dataset(output_root):
        raise RuntimeError(f"aggregation produced no dataset at {output_root}")
    return Aggregated(root=output_root, repo_id=repo_id, sources=sources)
