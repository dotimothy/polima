"""Dataset selection, interactive or not.

The legacy scripts always prompt ("Select one or more datasets (for example
1,2,4):"), which makes them unusable from cron or CI. Here the prompt is a
fallback: if the caller named datasets, or stdin is not a TTY, selection is
non-interactive.
"""

from __future__ import annotations

import sys

from polima.data.discover import DatasetEntry, discover, resolve_roots
from polima.data.episodes import EpisodeSpecError, parse_episode_spec
from polima.util.table import render


class SelectionError(ValueError):
    pass


def select(
    *,
    names: list[str] | None = None,
    roots: list[str] | None = None,
    parent: str,
    interactive: bool | None = None,
) -> list[DatasetEntry]:
    """Resolve the dataset selection.

    Precedence: explicit --dataset-root, then --dataset names, then an
    interactive pick from `parent`.
    """
    explicit = list(roots or []) + list(names or [])
    if explicit:
        return [
            DatasetEntry(root=root, repo_id=f"local/{root.name}")
            for root in resolve_roots(explicit, parent=parent)
        ]

    entries = discover(parent)
    if not entries:
        raise SelectionError(f"no LeRobot datasets found under {parent}")

    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise SelectionError(
            f"no dataset given and stdin is not a TTY; pass --dataset NAME "
            f"(available under {parent}: {', '.join(e.name for e in entries)})"
        )
    return prompt(entries)


def prompt(entries: list[DatasetEntry]) -> list[DatasetEntry]:
    """Numbered multi-select, accepting the same "1,2,4" and "1-3" grammar as
    episode specs."""
    rows = [
        [f"{index + 1})", entry.name, entry.episodes or "?", entry.frames or "?",
         f"{entry.fps or '?'}fps"]
        for index, entry in enumerate(entries)
    ]
    print(render(rows, headers=["#", "dataset", "episodes", "frames", "rate"]))

    while True:
        raw = input("Select one or more datasets (for example 1,2,4): ").strip()
        if not raw:
            continue
        try:
            picked = parse_episode_spec(raw)
        except EpisodeSpecError as exc:
            print(f"  {exc}")
            continue
        out_of_range = [n for n in picked if not 1 <= n <= len(entries)]
        if out_of_range:
            print(f"  out of range: {out_of_range} (1-{len(entries)})")
            continue
        return [entries[n - 1] for n in picked]
