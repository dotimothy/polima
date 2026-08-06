"""`polima doctor` -- diagnose the three interpreters and the board.

This exists because the legacy stack has several environment landmines that fail
late and confusingly. Each check below corresponds to one of them:

  * `llima-compile` carries the shebang `#!/sdk-extensions/model-compiler/bin/python`,
    an in-container path that does not exist on this host, so exec'ing it dies
    with "required file not found". SmolVLA's ENV_CHECK does exactly that and can
    therefore never pass. Doctor probes it BY IMPORT instead.
  * `~/.codex/skills/sima-model-surgery` and `sima-model-quantize-compile` are
    referenced by all three compile controllers but are not installed. That is a
    warning here, never a hard failure.
  * polima.util / config / policies / bundle / wire must import with stdlib+numpy
    alone, or polima.compile.afe_compile breaks inside the compiler venv. The
    import matrix enforces it.
  * `SmolVLA/lerobot` shows ~974 dirty paths that are pure 100644->100755 mode
    churn, burying 3 real source edits. Doctor suppresses that noise.
  * `$HOME/MLSandbox` is a load-bearing symlink: ACT/lerobot's git remote points
    at its own sibling through it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from polima.config.loader import load
from polima.util import table
from polima.util.paths import repo_root, symlink_report

#: Modules that must import with stdlib + numpy only, in every interpreter.
CORE_MODULES = (
    "polima.util.jsonio",
    "polima.util.hashing",
    "polima.util.paths",
    "polima.config.base",
    "polima.config.loader",
    "polima.policies.base",
    "polima.policies.registry",
    "polima.data.episodes",
    "polima.data.contract",
)

COMPILER_BIN_CANDIDATES = (
    "$MODEL_COMPILER_BIN",
    "~/sima-sdk-extensions/model-compiler/bin",
    "/sdk-extensions/model-compiler/bin",
)

OPTIONAL_SKILLS = (
    "~/.codex/skills/sima-model-surgery/scripts/model_surgery_guard.py",
    "~/.codex/skills/sima-model-quantize-compile/scripts/quantize_compile.py",
)

LEROBOT_CLONES = ("ACT/lerobot", "SmolVLA/lerobot")
PINNED_LEROBOT_SHA = "36b8face988669509272b00f4abe6592d0b17aa0"


class Doctor:
    def __init__(self, *, json_output: bool = False) -> None:
        self.json_output = json_output
        self.results: list[dict] = []
        self.worst = table.OK

    def record(self, kind: str, label: str, detail: str = "", **extra) -> None:
        self.results.append({"status": kind, "check": label, "detail": detail, **extra})
        order = {table.OK: 0, table.SKIP: 0, table.WARN: 1, table.FAIL: 2}
        if order.get(kind, 0) > order.get(self.worst, 0):
            self.worst = kind
        if not self.json_output:
            print(table.status(kind, label, detail))

    def heading(self, title: str) -> None:
        if not self.json_output:
            print(table.section(title))


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima doctor", description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--board", action="store_true", help="only the board checks")
    parser.add_argument("--imports", action="store_true", help="only the import matrix")
    parser.add_argument("--no-board", action="store_true", help="skip board reachability")
    args = parser.parse_args(argv)

    config = load(config_file=getattr(parent, "config", None))
    doc = Doctor(json_output=args.json)
    only_board, only_imports = args.board, args.imports
    everything = not (only_board or only_imports)

    if everything:
        check_role(doc)
        check_host(doc)
        check_repo(doc)
    if everything or only_imports:
        check_import_matrix(doc, config)
    if everything:
        check_compiler(doc, config)
        check_policies(doc)
        check_datasets(doc, config)
    if (everything and not args.no_board) or only_board:
        check_board(doc, config)

    if args.json:
        print(json.dumps({"status": doc.worst, "checks": doc.results}, indent=2))
    else:
        print()
        counts: dict[str, int] = {}
        for item in doc.results:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if doc.worst == table.FAIL else 0


# ----------------------------------------------------------------- host / repo


def check_role(doc: Doctor) -> None:
    from polima import role

    capabilities = role.detect()
    doc.heading(f"install role: {capabilities.role} ({capabilities.machine})")

    stages = (
        ("train", capabilities.can_train, role.HOST),
        ("compile", capabilities.can_compile, role.HOST),
        ("deploy", capabilities.can_deploy, role.HOST),
        ("run", capabilities.can_run, role.BOARD),
        ("robot", capabilities.can_robot, role.BOARD),
    )
    for command, enabled, belongs in stages:
        # A host missing board-only extras is normal, and vice versa -- only
        # report a gap when the stage belongs on THIS machine.
        expected_here = (
            belongs == role.HOST and capabilities.role in (role.HOST, role.BOTH)
        ) or (belongs == role.BOARD and capabilities.role in (role.BOARD, role.BOTH))
        if enabled:
            doc.record(table.OK, f"polima-{command}", "available")
        elif expected_here:
            doc.record(table.WARN, f"polima-{command}", capabilities.explain(command))
        else:
            doc.record(table.SKIP, f"polima-{command}", f"{belongs}-side stage, not needed here")

    if capabilities.missing:
        doc.record(
            table.SKIP, "optional modules",
            ", ".join(f"{m} ({p})" for m, p in sorted(capabilities.missing.items())),
        )


def check_host(doc: Doctor) -> None:
    doc.heading("host")
    doc.record(table.OK, "python", f"{sys.version.split()[0]} at {sys.executable}")
    try:
        import numpy

        doc.record(table.OK, "numpy", numpy.__version__)
    except ImportError:
        doc.record(table.FAIL, "numpy", "missing -- polima's only hard dependency")

    conda = shutil.which("conda")
    doc.record(table.OK if conda else table.WARN, "conda", conda or "not on PATH")
    if conda:
        listing = _capture([conda, "env", "list"])
        names = {
            line.split()[0]
            for line in listing.splitlines()
            if line.strip() and not line.startswith("#")
        }
        for env in ("act", "vla", "groot-n1.6", "groot-n1.6-convert"):
            doc.record(
                table.OK if env in names else table.WARN,
                f"conda env {env}",
                "present" if env in names else "absent",
            )


def check_repo(doc: Doctor) -> None:
    doc.heading("repository")
    report = symlink_report()
    doc.record(
        table.OK if report["consistent"] else table.WARN,
        "$HOME/MLSandbox symlink",
        f"{report['target']} (repo root {report['resolved_repo_root']})"
        if report["exists"]
        else "absent -- ACT/lerobot's git remote resolves through it",
    )

    root = repo_root()
    doc.record(
        table.WARN if not (root / ".git").exists() else table.OK,
        "version control",
        "MLSandbox is not a git repo (planned for Phase 2); "
        "`make check-legacy-intact` is the regression guard until then"
        if not (root / ".git").exists()
        else "git present",
    )

    manifest = root / "polima" / "tests" / "fixtures" / "legacy_manifest.txt"
    if manifest.is_file():
        count = len(manifest.read_text().splitlines())
        result = subprocess.run(
            ["sha256sum", "-c", "--quiet", str(manifest)],
            cwd=root, capture_output=True, text=True,
        )
        doc.record(
            table.OK if result.returncode == 0 else table.FAIL,
            "legacy intact",
            f"{count} files unchanged" if result.returncode == 0
            else result.stdout.strip().splitlines()[:3],
        )
    else:
        doc.record(table.WARN, "legacy intact", "no manifest; run `make snapshot-legacy`")

    backups = root / ".legacy-backups"
    stamps = sorted(
        p.name for p in backups.iterdir()
        if p.is_dir() and not p.is_symlink()   # skip the `latest` convenience link
    ) if backups.is_dir() else []
    if stamps:
        files = len(list((backups / stamps[-1] / "tree").rglob("*")))
        doc.record(
            table.OK, "legacy backups",
            f"{len(stamps)} snapshot(s), newest {stamps[-1]} ({files} paths)",
        )
    else:
        doc.record(table.WARN, "legacy backups", "none; run `make backup-legacy`")

    for clone in LEROBOT_CLONES:
        path = root / clone
        if not (path / ".git").exists():
            doc.record(table.WARN, f"{clone}", "not a git clone")
            continue
        sha = _capture(["git", "-C", str(path), "rev-parse", "HEAD"]).strip()
        # -c core.fileMode=false suppresses the 974-path chmod noise in
        # SmolVLA/lerobot that otherwise hides the 3 real source edits.
        dirty = _capture(
            ["git", "-c", "core.fileMode=false", "-C", str(path), "status", "--porcelain"]
        ).strip()
        changed = len([line for line in dirty.splitlines() if line.strip()])
        pinned = sha == PINNED_LEROBOT_SHA
        doc.record(
            table.OK if pinned else table.WARN,
            f"{clone}",
            f"{sha[:8]}{'' if pinned else ' (NOT the pinned revision)'}, "
            f"{changed} real change(s) [mode churn suppressed]",
        )


# -------------------------------------------------------------- import matrix


def check_import_matrix(doc: Doctor, config) -> None:
    doc.heading("import matrix (core modules must need only stdlib + numpy)")
    interpreters = {"current": sys.executable}

    compiler_bin = _find_compiler_bin(config)
    if compiler_bin:
        interpreters["model-compiler"] = str(Path(compiler_bin) / "python")
    for env in ("act",):
        candidate = _conda_python(env)
        if candidate:
            interpreters[f"conda:{env}"] = candidate

    source_root = str(Path(__file__).resolve().parents[2])
    snippet = (
        "import sys; sys.path.insert(0, %r);\n" % source_root
        + "import importlib\n"
        + f"mods = {list(CORE_MODULES)!r}\n"
        + "bad = []\n"
        + "for m in mods:\n"
        + "    try: importlib.import_module(m)\n"
        + "    except Exception as e: bad.append(f'{m}: {type(e).__name__}: {e}')\n"
        + "print('|'.join(bad))\n"
    )

    for label, python in interpreters.items():
        if not python or not Path(python).exists():
            doc.record(table.SKIP, f"imports [{label}]", "interpreter not found")
            continue
        result = subprocess.run(
            [python, "-c", snippet], capture_output=True, text=True, timeout=120
        )
        failures = result.stdout.strip()
        version = _capture([python, "-c", "import sys;print(sys.version.split()[0])"]).strip()
        if result.returncode != 0:
            doc.record(table.FAIL, f"imports [{label}]", result.stderr.strip()[:160])
        elif failures:
            doc.record(table.FAIL, f"imports [{label}]", failures[:200])
        else:
            doc.record(
                table.OK, f"imports [{label}]", f"py{version}: all {len(CORE_MODULES)} core modules"
            )


# -------------------------------------------------------------- compiler venv


def check_compiler(doc: Doctor, config) -> None:
    doc.heading("model compiler (SiMa ModelSDK)")
    compiler_bin = _find_compiler_bin(config)
    if not compiler_bin:
        doc.record(
            table.FAIL, "compiler bin",
            "not found; tried " + ", ".join(COMPILER_BIN_CANDIDATES),
        )
        return
    doc.record(table.OK, "compiler bin", compiler_bin)

    python = str(Path(compiler_bin) / "python")
    version = _capture([python, "-c", "import sys;print(sys.version.split()[0])"]).strip()
    doc.record(table.OK if version else table.FAIL, "compiler python", version or "unusable")

    afe = _capture(
        [python, "-c",
         "from afe.apis.release_v1 import get_model_sdk_version as v;print(v())"]
    ).strip()
    doc.record(table.OK if afe else table.FAIL, "afe / ModelSDK", afe or "import failed")

    # By IMPORT, never by exec: the llima-compile console script's shebang points
    # at /sdk-extensions/..., which exists only inside the SDK container.
    lmm = _capture(
        [python, "-c",
         "import sima_lmm.host.compile_lmm as m;"
         "print(','.join(n for n in ('main','gen_files') if hasattr(m,n)))"]
    ).strip()
    doc.record(
        table.OK if "gen_files" in lmm else table.FAIL,
        "sima_lmm.host.compile_lmm",
        f"importable ({lmm})" if lmm else "import failed",
    )

    script = Path(compiler_bin) / "llima-compile"
    if script.is_file():
        shebang = script.read_text(errors="replace").splitlines()[0]
        interpreter = shebang.removeprefix("#!").strip().split()[0] if shebang.startswith("#!") else ""
        usable = bool(interpreter) and Path(interpreter).exists()
        doc.record(
            table.OK if usable else table.WARN,
            "llima-compile script",
            "executable" if usable
            else f"shebang {shebang!r} is unresolvable -- invoke "
                 "`$COMPILER_BIN/python -m sima_lmm.host.compile_lmm` instead",
        )

    for skill in OPTIONAL_SKILLS:
        path = Path(os.path.expanduser(skill))
        doc.record(
            table.OK if path.is_file() else table.WARN,
            f"optional {path.name}",
            str(path) if path.is_file() else "absent -- AUDIT stage will be skipped",
        )


# ------------------------------------------------------------ policies / data


def check_policies(doc: Doctor) -> None:
    doc.heading("policies")
    from polima.policies.registry import BUILTIN, load_all

    specs = load_all(strict=False)
    for name in BUILTIN:
        spec = specs.get(name)
        if spec is None:
            doc.record(table.SKIP, f"policy {name}", "not ported yet")
            continue
        try:
            spec.validate()
            doc.record(
                table.OK, f"policy {name}",
                f"{len(spec.compile.graphs)} graphs, port {spec.wire.default_port}, "
                f"magic {spec.wire.magic_ascii}",
            )
        except Exception as exc:  # noqa: BLE001
            doc.record(table.FAIL, f"policy {name}", str(exc)[:160])


def check_datasets(doc: Doctor, config) -> None:
    doc.heading(f"datasets under {config.paths.dataset_parent}")
    from polima.data.contract import validate
    from polima.data.discover import discover
    from polima.policies.registry import iter_specs

    entries = discover(config.paths.dataset_parent)
    if not entries:
        doc.record(table.WARN, "datasets", f"none found under {config.paths.dataset_parent}")
        return

    specs = list(iter_specs())
    for entry in entries:
        verdicts = []
        for spec in specs:
            report = validate(entry.root, spec.dataset)
            verdicts.append(f"{spec.name}={'ok' if report.ok else 'no'}")
        doc.record(
            table.OK, entry.name,
            f"{entry.episodes} eps, {entry.frames} frames, {entry.fps}fps  "
            + " ".join(verdicts),
        )


# ------------------------------------------------------------------- board


def check_board(doc: Doctor, config) -> None:
    board = config.board
    doc.heading(f"board {board.host}")
    probe = (
        "echo VER=$(uname -m); "
        "echo CORES=$(nproc); "
        f"echo FREE=$(df -BG --output=avail {board.root.split('/')[1] and '/media/nvme'} "
        "2>/dev/null | tail -1 | tr -d ' G'); "
        "echo CMAKE=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}'); "
        "echo SIMALMM=$(ls -d /usr/lib/*/cmake/SimaLMM 2>/dev/null | head -1); "
        "echo JSON=$(ls /usr/include/nlohmann/json.hpp 2>/dev/null); "
        f"echo POLIMA=$(ls -d {board.root} 2>/dev/null)"
    )
    result = subprocess.run(
        ["ssh", *board.ssh_options, "-o", f"ConnectTimeout={int(board.connect_timeout_s)}",
         board.host, probe],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        doc.record(
            table.WARN, "reachable",
            f"ssh failed: {result.stderr.strip().splitlines()[-1] if result.stderr else 'no output'}",
        )
        return

    facts = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    arch = facts.get("VER", "")
    doc.record(
        table.OK if arch == "aarch64" else table.FAIL,
        "reachable", f"{arch}, {facts.get('CORES', '?')} cores",
    )
    free = facts.get("FREE", "")
    doc.record(table.OK if free else table.WARN, "free space", f"{free}G on /media/nvme")
    doc.record(
        table.OK if facts.get("CMAKE") else table.FAIL, "cmake", facts.get("CMAKE", "absent")
    )
    doc.record(
        table.OK if facts.get("SIMALMM") else table.FAIL,
        "SimaLMM cmake package", facts.get("SIMALMM") or "absent -- find_package will fail",
    )
    doc.record(
        table.OK if facts.get("JSON") else table.WARN,
        "nlohmann/json", facts.get("JSON") or "absent -- vendor it into native/third_party",
    )
    doc.record(
        table.OK if facts.get("POLIMA") else table.SKIP,
        "polima root", facts.get("POLIMA") or f"{board.root} not created yet (first deploy will)",
    )


# ---------------------------------------------------------------- internals


def _capture(argv: list[str], timeout: float = 60) -> str:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _find_compiler_bin(config) -> str | None:
    explicit = getattr(config.paths, "model_compiler_bin", None)
    candidates = [str(explicit)] if explicit else []
    candidates += [os.environ.get("MODEL_COMPILER_BIN", "")]
    candidates += [os.path.expanduser("~/sima-sdk-extensions/model-compiler/bin")]
    candidates += ["/sdk-extensions/model-compiler/bin"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "python").exists():
            return candidate
    return None


def _conda_python(env: str) -> str | None:
    base = _capture(["conda", "info", "--base"]).strip()
    if not base:
        return None
    candidate = Path(base) / "envs" / env / "bin" / "python"
    return str(candidate) if candidate.exists() else None
