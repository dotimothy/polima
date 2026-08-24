"""Interactive setup for bare `polima compile` and `polima deploy` commands.

Invoked with no arguments, the compile stage used to print four lines telling
you to go and read the flags. But everything it was asking for is already on
disk and discoverable: which checkpoints exist, which build trees have an
`onnx/` waiting, which of those are already compiled and only need packing. So
a bare invocation now offers them.

Two properties keep this from becoming a second code path:

  * it runs ONLY when argv is empty and stdin is a TTY, so cron, CI and every
    explicit `polima compile --build-dir ...` behave exactly as before; and
  * it composes an argv and hands it straight back to the same parser. Nothing
    here compiles anything. The command it prints is literally the one that
    runs, so the session teaches the flags instead of replacing them.

stdlib only, like the rest of the core -- `readline` is imported for line
editing and path completion where the interpreter has it, and simply skipped
where it does not.
"""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from polima.util import table
from polima.util.paths import outputs_root, repo_root

#: Directories a compiler build tree grows once its graphs are built. Any one of
#: them means the tree can be packed with --import-legacy and needs no compiler.
COMPILED_MARKERS = ("models_uncompressed", "models", "retained")

#: Enough to choose from without paging; the newest are what anyone wants.
MAX_LISTED = 12


class Cancelled(Exception):
    """Ctrl-C or EOF at a prompt. Not an error -- the user changed their mind."""


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    run: str
    steps: int
    mtime: float


@dataclass(frozen=True)
class BuildTree:
    path: Path
    graphs: int
    compiled: bool
    mtime: float


@dataclass(frozen=True)
class LocalBundle:
    path: Path
    bundle_id: str
    policy: str
    graphs: int
    elf_bytes: int
    mtime: float


# ------------------------------------------------------------------ discovery


def stack_outputs(spec) -> Path | None:
    """The legacy stack's outputs/ for this policy.

    Derived from `TrainSpec.repo_dir_hint` ("ACT/lerobot" -> ACT) rather than a
    second hardcoded policy->directory map that could drift from the spec.
    """
    hint = getattr(spec.train, "repo_dir_hint", None)
    if not hint:
        return None
    return repo_root() / hint.split("/")[0] / "outputs"


def find_checkpoints(spec, limit: int = MAX_LISTED) -> list[Checkpoint]:
    outputs = stack_outputs(spec)
    if not outputs or not outputs.is_dir():
        return []
    found = []
    seen: set[Path] = set()
    # lerobot writes `checkpoints/last -> 100000`, so the glob returns the same
    # checkpoint twice. Dedupe on the resolved path and read the step count from
    # the target, which is where the number actually is.
    for path in sorted(outputs.glob(f"*/{spec.train.checkpoint_glob}")):
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            steps = int(resolved.parent.name)
        except ValueError:
            steps = 0
        found.append(Checkpoint(path, path.parents[2].name, steps, _mtime(path)))
    found.sort(key=lambda c: (-c.mtime, -c.steps))
    return found[:limit]


def find_build_trees(spec, limit: int = MAX_LISTED) -> list[BuildTree]:
    """Trees holding an `onnx/`, newest first, from both places they land."""
    roots = [stack_outputs(spec), outputs_root() / "build"]
    trees: list[BuildTree] = []
    seen: set[Path] = set()
    for root in roots:
        if not root or not root.is_dir():
            continue
        for onnx in sorted(root.glob("*/onnx")):
            tree = onnx.parent
            if not onnx.is_dir() or tree in seen:
                continue
            seen.add(tree)
            trees.append(BuildTree(
                tree,
                len(list(onnx.glob("*.onnx"))),
                any((tree / marker).is_dir() for marker in COMPILED_MARKERS),
                _mtime(tree),
            ))
    trees.sort(key=lambda t: -t.mtime)
    return trees[:limit]


def find_local_bundles(
    root: str | Path | None = None, limit: int = MAX_LISTED,
) -> list[LocalBundle]:
    """Valid local bundles, newest first, for the deploy session."""
    import json

    bundle_root = Path(root) if root else outputs_root() / "bundles"
    found: list[LocalBundle] = []
    if not bundle_root.is_dir():
        return found
    for manifest in sorted(bundle_root.glob("*/bundle.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            bundle_id = str(data["bundle_id"])
            policy = str(data["policy"])
            graphs = data.get("graphs") or []
            if not isinstance(graphs, list):
                continue
            found.append(LocalBundle(
                path=manifest.parent,
                bundle_id=bundle_id,
                policy=policy,
                graphs=len(graphs),
                elf_bytes=sum(int(graph.get("elf_bytes") or 0) for graph in graphs),
                mtime=_mtime(manifest),
            ))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            # An interrupted pack must not make the whole chooser unusable.
            continue
    found.sort(key=lambda bundle: -bundle.mtime)
    return found[:limit]


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _stamp(mtime: float) -> str:
    import time

    return time.strftime("%Y-%m-%d", time.localtime(mtime)) if mtime else "?"


def default_build_dir(spec, checkpoint: Checkpoint) -> Path:
    """PoLiMa-owned build tree for an external training checkpoint."""
    dataset = checkpoint.run.split(f"_{spec.name}_")[0] or checkpoint.run
    return outputs_root() / "build" / f"polima_{dataset}_{spec.name}_{checkpoint.steps}"


# -------------------------------------------------------------------- prompts


def _enable_line_editing() -> None:
    """Arrow keys, history and tab-completed paths, where readline exists."""
    try:
        import readline
    except ImportError:      # pragma: no cover - not every build ships it
        return

    def complete(text: str, state: int):
        expanded = os.path.expanduser(text)
        matches = [
            str(p) + ("/" if p.is_dir() else "")
            for p in Path(expanded or ".").parent.glob(Path(expanded).name + "*")
        ] if expanded else []
        return matches[state] if state < len(matches) else None

    readline.set_completer_delims(" \t\n")
    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")


def ask(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"  {question}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise Cancelled from None
    return answer or default


def confirm(question: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    answer = ask(f"{question} {hint}").lower()
    if answer in ("y", "yes"):
        return True
    if answer in ("n", "no"):
        return False
    return default


def choose(question: str, options: list[tuple[str, str]], default: int = 1) -> int:
    """Numbered single-select, 1-based, matching polima.data.select's grammar."""
    width = max(len(label) for label, _ in options)
    for index, (label, detail) in enumerate(options, 1):
        print(f"   {index}) {label:<{width}}  {detail}")
    while True:
        raw = ask(question, str(default))
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw)
        print(f"  choose 1-{len(options)}")


# ---------------------------------------------------------------- composition


def compose(specs: dict, *, can_compile: bool) -> list[str] | None:
    """Walk the user to a compile command line. None means they cancelled.

    `specs` is name -> PolicySpec for the policies that actually load here, so
    the session never offers one that `get_policy` would raise on.
    """
    _enable_line_editing()
    print(table.section("polima compile"))
    print("  Nothing specified, so let's put a command together.")
    print("  Ctrl-C to quit; the command is shown before anything runs.\n")

    spec = next(iter(specs.values()))
    if len(specs) > 1:
        options = [(name, s.display_name) for name, s in sorted(specs.items())]
        spec = specs[options[choose("policy?", options) - 1][0]]
        print()

    checkpoints = find_checkpoints(spec)
    trees = find_build_trees(spec)
    packable = [t for t in trees if t.compiled]

    modes: list[tuple[str, str, str]] = []
    if can_compile and checkpoints:
        modes.append(("checkpoint", "export, compile and pack from scratch",
                      f"{len(checkpoints)} found"))
    if can_compile and trees:
        modes.append(("build tree", "compile the onnx/ already exported",
                      f"{len(trees)} found"))
    if packable:
        modes.append(("adopt a built tree", "pack ELFs that are already compiled",
                      f"{len(packable)} found, needs no compiler"))
    modes.append(("type a path", "name a checkpoint or build tree myself", ""))

    if not can_compile:
        print(table.status(
            table.WARN, "no model compiler",
            "only packing is offered; set MODEL_COMPILER_BIN for the rest",
        ))

    pick = choose("what do you want to do?",
                  [(label, f"{detail}  {extra}".strip()) for label, detail, extra in modes])
    mode = modes[pick - 1][0]

    if mode == "checkpoint":
        return _from_checkpoint(spec, checkpoints)
    if mode == "build tree":
        return _from_build_tree(spec, trees)
    if mode == "adopt a built tree":
        return _from_adopted(spec, packable)
    return _from_path(spec, can_compile)


def _from_checkpoint(spec, checkpoints: list[Checkpoint]) -> list[str] | None:
    print()
    print(table.render(
        [[f"{i})", _clip(c.run, 52), c.steps, _stamp(c.mtime)]
         for i, c in enumerate(checkpoints, 1)],
        headers=["#", "run", "steps", "trained"],
    ))
    chosen = checkpoints[_pick_index("checkpoint?", len(checkpoints)) - 1]

    build_dir = ask("build dir?", str(default_build_dir(spec, chosen)))
    argv = ["--policy", spec.name, "--checkpoint", str(chosen.path), "--build-dir", build_dir]
    return _finish(argv, compiling=True, elf_count=len(spec.compile.graphs))


def _from_build_tree(spec, trees: list[BuildTree]) -> list[str] | None:
    print()
    print(table.render(
        [[f"{i})", _clip(t.path.name, 46), t.graphs, "yes" if t.compiled else "no",
          _stamp(t.mtime)]
         for i, t in enumerate(trees, 1)],
        headers=["#", "build tree", "graphs", "built", "touched"],
    ))
    chosen = trees[_pick_index("build tree?", len(trees)) - 1]
    argv = ["--policy", spec.name, "--build-dir", str(chosen.path)]
    if chosen.compiled and confirm("this tree is already built -- reuse unchanged graphs?", False):
        argv.append("--reuse")
    return _finish(argv, compiling=True, elf_count=chosen.graphs)


def _from_adopted(spec, trees: list[BuildTree]) -> list[str] | None:
    print()
    print(table.render(
        [[f"{i})", _clip(t.path.name, 46), t.graphs, _stamp(t.mtime)]
         for i, t in enumerate(trees, 1)],
        headers=["#", "built tree", "graphs", "touched"],
    ))
    chosen = trees[_pick_index("tree to adopt?", len(trees)) - 1]
    return _finish(["--policy", spec.name, "--import-legacy", str(chosen.path)], compiling=False)


def _from_path(spec, can_compile: bool) -> list[str] | None:
    path = Path(os.path.expanduser(ask("path to a checkpoint or build tree"))).resolve()
    if not path.is_dir():
        print(f"  {path} is not a directory")
        return None
    if (path / "onnx").is_dir():
        flag = "--build-dir" if can_compile else "--import-legacy"
        elf_count = len(list((path / "onnx").glob("*.onnx"))) if can_compile else None
        return _finish(
            ["--policy", spec.name, flag, str(path)],
            compiling=can_compile,
            elf_count=elf_count,
        )
    build_dir = ask("build dir for the export?",
                    str(outputs_root() / "build" / f"polima_{path.name}"))
    return _finish(
        ["--policy", spec.name, "--checkpoint", str(path), "--build-dir", build_dir],
        compiling=True,
        elf_count=len(spec.compile.graphs),
    )


def _pick_index(question: str, count: int) -> int:
    while True:
        raw = ask(question, "1")
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw)
        print(f"  choose 1-{count}")


def _finish(
    argv: list[str], *, compiling: bool, elf_count: int | None = None,
) -> list[str] | None:
    """Ask the shared options, show the command, and confirm before returning."""
    if compiling:
        cores = os.cpu_count()
        elves = (
            f"{elf_count} ELF{'s' if elf_count != 1 else ''} to compile"
            if elf_count is not None else "ELF count unknown"
        )
        host = f"{cores} logical CPU core{'s' if cores != 1 else ''}" if cores else "CPU count unknown"
        jobs = ask(f"jobs? ({elves}; {host}; afe needs ~1.6 GB/job)", "1")
        if jobs.isdigit() and int(jobs) > 1:
            argv += ["--jobs", jobs]
        stage = choose(
            "stop after?",
            [("pack", "compile and write a deployable bundle"),
             ("compile", "leave the build tree, do not pack"),
             ("export", "just write onnx/ and check it against PyTorch")],
        )
        stop_after = ("pack", "compile", "export")[stage - 1]
        if stop_after != "pack":
            argv += ["--stop-after", stop_after]

    print(f"\n  polima compile {' '.join(argv)}\n")
    if not confirm("run this?"):
        print("  nothing run")
        return None
    return argv


def compose_deploy(board, bundles_root: str | Path | None = None) -> list[str] | None:
    """Walk the user to a normal ``polima deploy`` argument vector."""
    _enable_line_editing()
    print(table.section("polima deploy"))
    print("  Nothing specified, so let's put a deployment together.")
    print("  Ctrl-C to quit; the command is shown before anything runs.\n")

    bundles = find_local_bundles(bundles_root)
    if bundles:
        rows = [
            (
                _clip(bundle.bundle_id, 62),
                (
                    f"{bundle.policy}, {bundle.graphs} graphs, "
                    f"{bundle.elf_bytes / 1048576:.1f} MiB ELFs"
                ),
            )
            for bundle in bundles
        ]
        rows.append(("type a path", "deploy a bundle outside the output folder"))
        selected = choose("bundle?", rows)
        if selected <= len(bundles):
            bundle_path = str(bundles[selected - 1].path)
            policy = bundles[selected - 1].policy
        else:
            bundle_path = ask("bundle path?")
            policy = ""
    else:
        print(table.status(table.WARN, "no local bundles", "enter a bundle path"))
        bundle_path = ask("bundle path?")
        policy = ""

    if not bundle_path:
        print("  no bundle selected")
        return None

    host = ask("board?", board.host)
    default_port = board.port
    if default_port is None and policy:
        try:
            from polima.policies.registry import get_policy

            default_port = get_policy(policy).wire.default_port
        except (KeyError, AttributeError):
            default_port = None
    port = ask("port?", str(default_port or ""))

    action = choose(
        "after deployment?",
        [
            ("deploy only", "copy, build and activate; leave the server stopped"),
            ("deploy and start", "start serving this bundle after activation"),
        ],
    )

    argv = ["--bundle", bundle_path]
    if host:
        argv += ["--board", host]
    if port:
        argv += ["--port", port]
    if action == 2:
        argv.append("--start")

    if confirm("advanced options?", False):
        if confirm("skip the on-board CMake build?", False):
            argv.append("--no-build")
        if confirm("leave the board's current link unchanged?", False):
            argv.append("--no-activate")
        if confirm("force transfer and rebuild?", False):
            argv.append("--force")
        if action == 2 and confirm("show per-stage server timings?", False):
            argv.append("--verbose-server")

    print(f"\n  polima deploy {shlex.join(argv)}\n")
    if not confirm("run this?"):
        print("  nothing run")
        return None
    return argv


def bare_invocation_is_interactive(argv: list[str]) -> bool:
    """Only a truly bare call, and only at a terminal.

    Both halves matter: a script that pipes input must keep getting the old
    non-zero exit and its four-line explanation, not a hung prompt.
    """
    return not argv and sys.stdin.isatty() and sys.stdout.isatty()
