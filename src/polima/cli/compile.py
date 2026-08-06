"""`polima-compile` -- produce a deployable bundle.

Phase 1a implements only `--import-legacy`, which adopts an existing compiler
build tree without recompiling. The full export/quantize/compile pipeline lands
in Phase 1b; until then the legacy `ACT/scripts/compile_act_modalix_elves.sh`
remains the way to produce new ELFs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polima.config.loader import load
from polima.policies.registry import get_policy
from polima.util import table


def needs_capability(argv: list[str]) -> str | None:
    """`--import-legacy` only copies already-built ELFs, so it runs anywhere.

    A real compile needs the SiMa model-compiler venv (afe + onnx + onnxsim);
    gating the whole command on that would wrongly block the import path, which
    is exactly what Phase 1a depends on.
    """
    if any(a == "--import-legacy" or a.startswith("--import-legacy=") for a in argv):
        return None
    return "compile"


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima-compile")
    parser.add_argument("--policy", default="act")
    parser.add_argument(
        "--import-legacy", metavar="BUILD_DIR",
        help="adopt an existing compiler build tree instead of recompiling",
    )
    parser.add_argument("--output-root", default=None, help="where bundles are written")
    parser.add_argument("--dataset", default=None, help="override the dataset name in the id")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = load(config_file=getattr(parent, "config", None))
    output_root = Path(
        args.output_root or (config.paths.outputs or _default_outputs()) / "bundles"
    )

    if not args.import_legacy:
        print(
            "polima-compile: the export/quantize/compile pipeline is Phase 1b.\n"
            "  Available now:  polima-compile --import-legacy <build_dir>\n"
            "  To build new ELFs meanwhile, use ACT/scripts/compile_act_modalix_elves.sh",
            file=sys.stderr,
        )
        return 2

    from polima.bundle.import_legacy import import_legacy

    spec = get_policy(args.policy)
    bundle = import_legacy(
        args.import_legacy, spec,
        output_root=output_root, dataset=args.dataset, steps=args.steps,
    )

    if args.json:
        from polima.util.jsonio import dumps

        print(dumps(bundle.to_dict()), end="")
        return 0

    print(table.section(f"bundle {bundle.bundle_id}"))
    print(table.render(
        [[a.name, a.precision, f"{a.elf_bytes / 1048576:.1f} MiB", a.sha256[:12],
          a.input_elements, a.output_elements] for a in bundle.graphs],
        headers=["graph", "precision", "elf", "sha256", "in", "out"],
    ))
    print(f"\n  root      {bundle.root}")
    print(f"  source    {bundle.source} <- {bundle.legacy_source_dir}")
    print(f"  total     {bundle.total_elf_bytes / 1048576:.1f} MiB of ELF")
    print(f"\nnext:  polima-deploy --bundle {bundle.bundle_id}")
    return 0


def _default_outputs() -> Path:
    from polima.util.paths import outputs_root

    return outputs_root()
