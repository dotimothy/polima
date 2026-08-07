"""`polima list` -- what policies exist and what each one compiles to."""

from __future__ import annotations

import argparse
import json

from polima.policies.registry import BUILTIN, load_all
from polima.util import table


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima list")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--graphs", action="store_true", help="show each policy's subgraphs")
    args = parser.parse_args(argv)

    specs = load_all(strict=False)

    if args.json:
        print(json.dumps({
            name: (specs[name].summary() if name in specs else {"status": "not ported yet"})
            for name in BUILTIN
        }, indent=2))
        return 0

    rows = []
    for name in BUILTIN:
        spec = specs.get(name)
        if spec is None:
            rows.append([name, "-", "not ported yet", "", ""])
            continue
        rows.append([
            name,
            f"{len(spec.compile.graphs)}",
            spec.train.backend,
            f"{spec.wire.magic_ascii}:{spec.wire.default_port}",
            f"{spec.robot.actions_per_chunk}x{spec.dataset.action_dim}",
        ])
    print(table.render(rows, headers=["policy", "graphs", "checkpoints from", "wire", "chunk"]))

    if args.graphs:
        for name, spec in specs.items():
            print(table.section(f"{name}: {spec.display_name}"))
            print(table.render(
                [[g.name,
                  "x".join(str(d) for d in g.inputs[0].shape),
                  "x".join(str(d) for d in g.outputs[0].shape),
                  g.layout, "/".join(g.precisions), g.compiler]
                 for g in spec.compile.graphs],
                headers=["graph", "input", "output", "layout", "precision", "compiler"],
            ))
    return 0
