"""`polima clean` -- reclaim space from build trees, without breaking them.

A six-graph ACT build tree is ~1.9 GB, and most of it is scratch the compiler
left behind: `.mlc` intermediates and `model_graph_json` beside each ELF, mpk
archives whose contents were already extracted, and `_tensor_prepared.onnx`
copies that `polima compile` regenerates. The ELFs themselves are 108 MB.

Three levels, each naming what it costs you:

    scratch   compiler intermediates only.  Tree still packs AND still resumes:
              the content key hashes onnx/ and calibration/, which stay.
    inputs    also the export inputs (onnx/, calibration/).  Still packs; a
              recompile would have to re-export first.
    all       the whole tree.

Dry run by default. A command that deletes gigabytes should show its work first,
and `--yes` is one word.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from polima.util import table

LEVELS = ("scratch", "inputs", "all")

#: Kept at every level below `all`: this is what bundle packing reads and what
#: proves a bundle came from somewhere.
PROTECTED = ("models_uncompressed", "bundle.json", "plan.json")


@dataclass
class Plan:
    """What would be removed, and what it buys."""

    paths: list[Path] = field(default_factory=list)
    bytes_freed: int = 0
    kept_elfs: int = 0
    #: Candidates deliberately not removed, with the reason.
    skipped: list[str] = field(default_factory=list)

    def add(self, path: Path) -> None:
        size = _size(path)
        if size:
            self.paths.append(path)
            self.bytes_freed += size


def _size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _elf_names_outside(build_dir: Path) -> set[str]:
    """ELF filenames that exist as plain files outside any `compiled*` tree."""
    found: set[str] = set()
    for elf in build_dir.rglob("*.elf"):
        relative = elf.relative_to(build_dir)
        if not relative.parts[0].startswith("compiled"):
            found.add(elf.name)
    return found


def _first_orphan_elf(candidate: Path, survivors: set[str]) -> str | None:
    """An ELF under `candidate` -- loose or inside an mpk -- with no copy outside."""
    import tarfile

    for elf in candidate.rglob("*.elf"):
        if elf.name not in survivors:
            return elf.name
    for archive in candidate.rglob("*_mpk.tar.gz"):
        try:
            with tarfile.open(archive, "r:gz") as handle:
                for member in handle.getnames():
                    name = member.rsplit("/", 1)[-1]
                    if name.endswith(".elf") and name not in survivors:
                        return f"{name} (inside {archive.name})"
        except (tarfile.TarError, OSError):
            # Unreadable archive: assume it matters rather than delete it.
            return archive.name
    return None


def plan_for(build_dir: Path, level: str) -> Plan:
    plan = Plan()
    if level == "all":
        plan.add(build_dir)
        return plan

    # afe's own output: mpk archives and quantized models. Normally the ELFs
    # were copied out to retained/ and models_uncompressed/ at compile time --
    # but not always, and a blind delete here is how you lose one.
    #
    # SmolVLA's tree has already produced that near-miss once: four denoise
    # expert ELFs, 540 MB, existed ONLY inside .tar.gz archives. So every
    # candidate is checked for an ELF that is not duplicated elsewhere in the
    # tree, and kept if it holds one.
    survivors = _elf_names_outside(build_dir)
    for candidate in sorted(build_dir.glob("compiled*")):
        if not candidate.is_dir():
            continue
        orphan = _first_orphan_elf(candidate, survivors)
        if orphan:
            plan.skipped.append(f"{candidate.name}: holds {orphan}, found nowhere else")
            continue
        plan.add(candidate)

    # Beside each ELF the compiler leaves .mlc images and a graph dump, several
    # times the size of the ELF. Keep the ELF, drop the rest.
    retained = build_dir / "retained"
    if retained.is_dir():
        for graph_dir in sorted(retained.iterdir()):
            if not graph_dir.is_dir():
                continue
            for item in sorted(graph_dir.iterdir()):
                if item.suffix == ".elf":
                    plan.kept_elfs += 1
                    continue
                plan.add(item)

    # onnxsim's output, regenerated on every compile from the source ONNX.
    for prepared in sorted((build_dir / "onnx").glob("*_tensor_prepared.onnx")):
        plan.add(prepared)

    if level == "inputs":
        plan.add(build_dir / "onnx")
        plan.add(build_dir / "calibration")
    return plan


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima clean", description=__doc__)
    parser.add_argument("build_dir", nargs="+", help="build tree(s) to clean")
    parser.add_argument("--level", choices=LEVELS, default="scratch")
    parser.add_argument("--yes", action="store_true", help="actually delete")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    total = 0
    rows = []
    plans: list[tuple[Path, Plan]] = []
    for raw in args.build_dir:
        build_dir = Path(raw).resolve()
        if not build_dir.is_dir():
            print(f"polima clean: not a directory: {build_dir}", file=sys.stderr)
            return 2
        looks_like_build = (
            (build_dir / "retained").is_dir()
            or any(build_dir.glob("compiled*"))
            or (build_dir / "models_uncompressed").is_dir()
        )
        if args.level != "all" and not looks_like_build:
            print(f"polima clean: {build_dir} does not look like a build tree "
                  "(no retained/ or compiled/)", file=sys.stderr)
            return 2
        plan = plan_for(build_dir, args.level)
        plans.append((build_dir, plan))
        total += plan.bytes_freed
        rows.append([build_dir.name, f"{plan.bytes_freed / 1e9:.2f} GB",
                     f"{len(plan.paths)} path(s)",
                     f"{plan.kept_elfs} ELF(s) kept" if plan.kept_elfs else ""])

    if args.json:
        from polima.util.jsonio import dumps

        print(dumps({
            "level": args.level, "bytes": total, "applied": args.yes,
            "trees": [{"path": str(d), "bytes": p.bytes_freed,
                       "paths": [str(x) for x in p.paths]} for d, p in plans],
        }), end="")
    else:
        print(table.section(f"clean ({args.level})"))
        print(table.render(rows, headers=["tree", "frees", "removes", ""]))
        for _dir, plan in plans:
            for reason in plan.skipped:
                print(f"  kept  {reason}")
        print(f"\n  {total / 1e9:.2f} GB total")

    if not args.yes:
        if not args.json:
            print("\n  dry run -- add --yes to delete")
            if args.level == "scratch":
                print("  `scratch` keeps the ELFs, models_uncompressed/, onnx/ and\n"
                      "  calibration/, so the tree still packs and still resumes.")
            elif args.level == "inputs":
                print("  `inputs` also drops onnx/ and calibration/: the tree still\n"
                      "  packs, but a recompile would have to re-export first.")
        return 0

    for build_dir, plan in plans:
        for path in plan.paths:
            # Never step outside the tree being cleaned.
            if not str(path.resolve()).startswith(str(build_dir)):
                print(f"  refusing to remove {path} (outside {build_dir})", file=sys.stderr)
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
    if not args.json:
        print(f"\n  removed {total / 1e9:.2f} GB")
    return 0
