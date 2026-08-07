"""`polima-compile` -- produce a deployable bundle.

Two ways in:

  --import-legacy <build_dir>   adopt an already-built compiler tree unchanged.
                                Needs no compiler, so it runs anywhere.
  --build-dir <dir>             quantize and compile the ONNX graphs in that
                                tree into ELFs, then pack a bundle.

The second path replaces the per-policy shell loops in
`ACT/scripts/compile_deploy_act_som.sh` and
`SmolVLA/scripts/compile_deploy_smolvla_som.sh`. It is verified by reproduction:
run against the tree that produced the deployed ACT bundle, it regenerates the
ELFs byte for byte.

`--stop-after compile` leaves the build tree without packing, which is what you
want while iterating on a single graph.
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

    Anything else needs the SiMa model-compiler venv. Note that "needs" here
    means *reachable as a subprocess*, not importable in this interpreter: the
    compiler venv has afe but no torch, the training env has torch but no afe,
    so the two can never be the same interpreter. `role.detect` checks for the
    venv accordingly.
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
    parser.add_argument("--build-dir", metavar="DIR",
                        help="build tree holding onnx/ and calibration/ to compile")
    parser.add_argument("--graph", action="append", metavar="NAME",
                        help="compile only this graph (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="recompile even when the content key is unchanged")
    parser.add_argument("--stop-after", choices=("compile", "pack"), default="pack")
    parser.add_argument("--output-root", default=None, help="where bundles are written")
    parser.add_argument("--dataset", default=None, help="override the dataset name in the id")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = load(config_file=getattr(parent, "config", None))
    output_root = Path(
        args.output_root or (config.paths.outputs or _default_outputs()) / "bundles"
    )
    dry_run = bool(getattr(parent, "dry_run", False))

    if args.import_legacy and args.build_dir:
        print("polima-compile: pass either --import-legacy or --build-dir, not both",
              file=sys.stderr)
        return 2

    if args.build_dir:
        return _compile(args, config, output_root, dry_run)

    if not args.import_legacy:
        print(
            "polima-compile: nothing to do. Choose one:\n"
            "  --build-dir <dir>            compile onnx/ + calibration/ into ELFs\n"
            "  --import-legacy <build_dir>  adopt an existing compiler build tree",
            file=sys.stderr,
        )
        return 2

    return _import_legacy(args, output_root)


# ------------------------------------------------------------------- compile


def _compile(args, config, output_root: Path, dry_run: bool) -> int:
    from polima.compile.driver import CompileError, Driver, sdk_version
    from polima.compile.toolchain import compiler_env, require_compiler_python

    spec = get_policy(args.policy)
    build_dir = Path(args.build_dir).resolve()

    try:
        compiler_python = require_compiler_python(config)
    except FileNotFoundError as error:
        print(f"polima-compile: {error}", file=sys.stderr)
        return 2

    # The compiler venv has no polima installed, so it needs our src/ on
    # PYTHONPATH to run `-m polima.compile.tensor`. That module imports numpy
    # and (lazily) afe only, both of which the venv has.
    env = compiler_env(compiler_python.parent, source_root=_src_root())
    version = sdk_version(compiler_python, env) if not dry_run else ""

    driver = Driver(spec=spec, build_dir=build_dir, compiler_python=compiler_python,
                    env=env, dry_run=dry_run, force=args.force, sdk_version=version)

    print(table.section(f"compile {spec.name} in {build_dir.name}"))
    print(f"  compiler  {compiler_python}" + (f"  (ModelSDK {version})" if version else ""))
    try:
        results = driver.run(only=args.graph)
    except CompileError as error:
        print(f"polima-compile: {error}", file=sys.stderr)
        return 2

    built = [r for r in results if r.status == "compiled"]
    reused = [r for r in results if r.status == "reused"]
    failed = [r for r in results if r.status == "failed"]
    print(f"\n  {len(built)} built, {len(reused)} reused, {len(failed)} failed")

    if failed:
        for result in failed:
            print(f"\n  {result.name}: {result.note}", file=sys.stderr)
            for attempt in result.attempts:
                print(f"    {attempt['precision']}: {attempt.get('reason', '')}"
                      f"  see {attempt['log']}", file=sys.stderr)
        return 1

    if args.stop_after == "compile" or dry_run:
        print(f"\n  build tree {build_dir}")
        print(f"next:  polima-compile --build-dir {build_dir}   # to pack a bundle")
        return 0

    # A freshly compiled tree has the same shape as a legacy one -- that is
    # deliberate (see Driver's docstring), so the Phase-1a packer takes it as is.
    return _pack(args, build_dir, output_root)


def _pack(args, build_dir: Path, output_root: Path) -> int:
    from polima.bundle.import_legacy import import_legacy

    spec = get_policy(args.policy)
    bundle = import_legacy(build_dir, spec, output_root=output_root,
                           dataset=args.dataset, steps=args.steps,
                           source="polima-compile")
    return _report(bundle, args.json)


def _import_legacy(args, output_root: Path) -> int:
    from polima.bundle.import_legacy import import_legacy

    spec = get_policy(args.policy)
    bundle = import_legacy(args.import_legacy, spec, output_root=output_root,
                           dataset=args.dataset, steps=args.steps)
    return _report(bundle, args.json)


def _report(bundle, as_json: bool) -> int:
    if as_json:
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


def _src_root() -> Path:
    """The `src/` directory this package was imported from."""
    return Path(__file__).resolve().parents[2]


def _default_outputs() -> Path:
    from polima.util.paths import outputs_root

    return outputs_root()
