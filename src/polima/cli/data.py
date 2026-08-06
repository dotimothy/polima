"""`polima data` -- dataset inspection, validation and combination.

`polima data validate` is the direct replacement for
ACT/scripts/validate_act_datasets.py, and the Phase-0 parity proof is that the
two agree on every dataset in /ml_datasets.
"""

from __future__ import annotations

import argparse
import json
import sys

from polima.config.loader import load
from polima.data.contract import validate_all
from polima.data.discover import discover, resolve_roots
from polima.policies.registry import get_policy
from polima.util import table


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima data")
    sub = parser.add_subparsers(dest="action", required=True)

    validate_parser = sub.add_parser("validate", help="check datasets against a policy contract")
    validate_parser.add_argument("roots", nargs="+", help="dataset names or paths")
    validate_parser.add_argument("--policy", default="act", help="contract to check against")
    validate_parser.add_argument("--allow-mixed-tasks", action="store_true")
    validate_parser.add_argument("--json", action="store_true")
    validate_parser.add_argument("--dataset-parent", default=None)

    list_parser = sub.add_parser("list", help="list datasets under the dataset parent")
    list_parser.add_argument("--dataset-parent", default=None)
    list_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    config = load(config_file=getattr(parent, "config", None))
    dataset_parent = args.dataset_parent or config.paths.dataset_parent

    if args.action == "list":
        return _list(dataset_parent, args.json)
    return _validate(args, dataset_parent)


def _list(dataset_parent, as_json: bool) -> int:
    entries = discover(dataset_parent)
    if as_json:
        print(json.dumps([
            {"name": e.name, "root": str(e.root), "repo_id": e.repo_id,
             "episodes": e.episodes, "frames": e.frames, "fps": e.fps,
             "codebase_version": e.codebase_version}
            for e in entries
        ], indent=2))
        return 0
    if not entries:
        print(f"no datasets under {dataset_parent}")
        return 1
    print(table.render(
        [[e.name, e.episodes, e.frames, e.fps, e.codebase_version] for e in entries],
        headers=["dataset", "episodes", "frames", "fps", "version"],
    ))
    return 0


def _validate(args, dataset_parent) -> int:
    spec = get_policy(args.policy)
    roots = resolve_roots(args.roots, parent=dataset_parent)
    reports, tasks = validate_all(
        list(roots), spec.dataset, allow_mixed_tasks=args.allow_mixed_tasks
    )
    ok = all(report.ok for report in reports)

    if args.json:
        # Shape chosen to line up with validate_act_datasets.py --json for the
        # Phase-0 parity proof.
        print(json.dumps({
            "ok": ok,
            "policy": spec.name,
            "tasks": tasks,
            "mixed_tasks": len(tasks) > 1,
            "datasets": [report.to_dict() for report in reports],
        }, indent=2))
        return 0 if ok else 1

    for report in reports:
        name = report.root.rsplit("/", 1)[-1]
        if report.ok:
            print(table.status(
                table.OK, name,
                f"{report.episodes} eps, {report.frames} frames, {report.fps}fps"
                + (f", task: {report.tasks[0]}" if report.tasks else ""),
            ))
        else:
            print(table.status(table.FAIL, name))
            for violation in report.violations:
                print(f"      {violation}")
    if ok:
        print(f"\n{spec.display_name} dataset contract OK: {', '.join(tasks) or 'no task labels'}")
    else:
        print("\ncontract violated", file=sys.stderr)
    return 0 if ok else 1
