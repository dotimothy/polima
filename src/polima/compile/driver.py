"""Drive the per-graph quantize/compile loop.

This is the generic form of `ACT/scripts/act_modalix_compile_controller.py` and
`SmolVLA/scripts/smolvla_modalix_compile_controller.py`, which implement the same
loop twice with different graph lists hardcoded. Here the graph list comes from
`PolicySpec.compile.graphs`, so a new policy needs no controller at all.

Three behaviours are inherited deliberately:

* **Precision fallback.** ACT and GR00T both try int8 and fall back to bf16.
  `GraphSpec.precisions` encodes the order per graph.
* **ELF presence is the success condition**, not the exit code. A compile can
  return 0 having quietly placed the graph on the APU, which only shows up as a
  load failure on the board.
* **Resume.** Compiles take tens of minutes; a controller that cannot resume
  gets killed and restarted from scratch.

Resume is upgraded from "does the output file exist" to a content key over the
ONNX, the calibration data and the compile flags. That closes the hole the
legacy `--resume` had -- editing the export and re-running kept the stale ELF --
and generalizes SmolVLA's `--reuse-vision-dir`, whose whole purpose is to avoid
recompiling a backbone that did not change. See docs/carry-forward.md.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from polima.compile import mpk
from polima.util import proc
from polima.util.hashing import sha256_file, sha256_text

STATE_FILE = "compile_state.json"


@dataclass
class GraphResult:
    name: str
    status: str                      # compiled | reused | failed | skipped
    precision: str | None = None
    elf: str | None = None
    key: str | None = None
    duration_s: float = 0.0
    attempts: list[dict] = field(default_factory=list)
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("compiled", "reused")

    def summary(self) -> str:
        mark = {"compiled": "built", "reused": "reused", "failed": "FAILED",
                "skipped": "skipped"}[self.status]
        detail = f"{self.precision}" if self.precision else ""
        if self.status == "compiled":
            detail += f"  {self.duration_s:.0f}s"
        if self.note:
            detail += f"  {self.note}"
        return f"  {mark:<8} {self.name:<24} {detail}"


class CompileError(RuntimeError):
    pass


@dataclass
class Driver:
    """Compiles every graph in a policy's `CompilePlan` into `build_dir`.

    The build tree keeps the legacy layout on purpose:

        onnx/<name>.onnx                       input, written by the export stage
        calibration/<name>.npz                 input, written by the export stage
        compiled/<precision>/<name>/           afe's own output
        retained/<name>/<name>_stage1_mla.elf  the ELF the board actually runs
        models_uncompressed/<name>/share/      what bundle packing reads

    Keeping it identical means the bundle packer proven in Phase 1a consumes a
    freshly compiled tree with no changes, and a legacy tree stays importable.
    """

    spec: object                     # PolicySpec
    build_dir: Path
    compiler_python: Path
    env: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    force: bool = False
    sdk_version: str = ""
    #: Graphs compiled at once. afe is single-threaded per graph -- measured at
    #: ~100% of one core and ~1.6 GB RSS for ACT -- so a 6-graph policy uses one
    #: core of a 20-core host and takes the sum of its parts. Independent graphs
    #: have no ordering constraint: each reads its own ONNX and writes its own
    #: retained directory.
    #:
    #: Default 1 because memory, not CPU, is the real limit and it is
    #: policy-dependent. SmolVLA's compile script notes its vision and prefix
    #: stages "each require substantial host RAM" and runs them sequentially for
    #: that reason. Opt in with --jobs once you know the graph sizes.
    jobs: int = 1
    _state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------- paths

    def onnx_path(self, name: str) -> Path:
        return self.build_dir / "onnx" / f"{name}.onnx"

    def calibration_path(self, graph) -> Path | None:
        kind = graph.calibration.kind
        if kind == "random":
            return None
        suffix = "npz" if kind == "npz" else "f32"
        return self.build_dir / "calibration" / f"{graph.name}.{suffix}"

    def retained_dir(self, name: str) -> Path:
        return self.build_dir / "retained" / name

    def elf_path(self, graph) -> Path:
        return self.retained_dir(graph.name) / graph.elf_name

    # -------------------------------------------------------------- keys

    def stage_key(self, graph) -> str:
        """Content key over everything that can change the ELF.

        Deliberately excludes absolute paths, so a build tree relocated or
        rebuilt under a different name still counts as unchanged. Includes the
        SDK version, because an afe upgrade changes code generation while every
        input stays byte-identical.
        """
        parts = [
            f"policy={getattr(self.spec, 'name', '?')}",
            f"graph={graph.name}",
            f"layout={graph.layout}",
            f"precisions={','.join(graph.precisions)}",
            f"tessellation={int(graph.mla_tessellation)}",
            f"compiler={graph.compiler}",
            f"llima_args={','.join(graph.llima_args)}",
            f"sdk={self.sdk_version}",
        ]
        onnx = self.onnx_path(graph.name)
        parts.append(f"onnx={sha256_file(onnx) if onnx.exists() else 'missing'}")
        calibration = self.calibration_path(graph)
        if calibration is not None:
            parts.append(
                f"calib={sha256_file(calibration) if calibration.exists() else 'missing'}"
            )
        return sha256_text("\n".join(parts))

    def _state(self) -> dict:
        # Callers may hold _state_lock already; this only reads.
        path = self.build_dir / STATE_FILE
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}

    def _relative_elf(self, elf: str | None) -> str | None:
        """Record the ELF relative to the build tree.

        An absolute path here would undo the care taken in `stage_key`, which
        deliberately excludes paths so a relocated tree still counts as
        unchanged. The same build tree is routinely seen at two paths -- the
        Palette container mounts it as /workspace/... while the host sees
        ~/SDK/NEAT/workspace/... -- and an absolute path recorded on one side
        simply does not exist on the other, so every stage recompiles despite
        matching keys.
        """
        if not elf:
            return elf
        try:
            return str(Path(elf).resolve().relative_to(self.build_dir.resolve()))
        except ValueError:
            return elf      # outside the tree: keep it absolute, it is all we have

    def _resolve_elf(self, elf: str | None) -> Path | None:
        if not elf:
            return None
        path = Path(elf)
        return path if path.is_absolute() else self.build_dir / path

    def _record(self, result: GraphResult) -> None:
        # Serialized: with --jobs > 1 several graphs finish into the same file,
        # and a lost update means a stage recompiles next run for no reason.
        with self._state_lock:
            self._record_locked(result)

    def _record_locked(self, result: GraphResult) -> None:
        state = self._state()
        state[result.name] = {
            "key": result.key,
            "elf": self._relative_elf(result.elf),
            "precision": result.precision,
            "status": result.status,
            "recorded_at": time.time(),
        }
        path = self.build_dir / STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    # ----------------------------------------------------------- command

    def argv(self, graph, precision: str) -> list[str]:
        build = self.build_dir / "compiled" / precision
        command = [
            str(self.compiler_python), "-m", "polima.compile.tensor",
            "--model-path", str(self.onnx_path(graph.name)),
            "--build-dir", str(build),
            "--precision", precision,
            "--model-layout", graph.layout,
        ]
        calibration = self.calibration_path(graph)
        if calibration is not None:
            flag = "--calibration-npz" if graph.calibration.kind == "npz" else "--calibration-raw-f32"
            command += [flag, str(calibration)]
        if graph.mla_tessellation:
            command += ["--mla-tessellation", "--retain-compile-dir",
                        str(self.retained_dir(graph.name))]
        command += list(graph.llima_args)
        return command

    # ------------------------------------------------------------- elf

    def locate_elf(self, graph, precision: str) -> Path | None:
        """Where the ELF ended up, honouring `GraphSpec.elf_from`.

        `retained` is the direct path the ACT deploy script uses: afe's retained
        temp dir holds the raw `<name>_stage1_mla.elf`. `mpk` goes through the
        packed archive instead, which is what SmolVLA and GR00T do.
        """
        if graph.elf_from == "retained":
            candidate = self.elf_path(graph)
            return candidate if candidate.exists() else None
        archive = mpk.find(self.build_dir / "compiled" / precision, graph.name)
        if archive is None or not mpk.has_elf(archive):
            return None
        unpacked = mpk.unpack(archive, self.build_dir / "models_uncompressed" / graph.name)
        elves = sorted((unpacked / "share").glob("*.elf"))
        return elves[0] if elves else None

    def publish(self, graph, elf: Path) -> Path:
        """Copy the ELF to `models_uncompressed/<name>/share/`, which is where
        both the bundle packer and the legacy deploy script read from."""
        destination = self.build_dir / "models_uncompressed" / graph.name / "share"
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / graph.elf_name
        if elf.resolve() != target.resolve():
            shutil.copy2(elf, target)
        return target

    # ------------------------------------------------------------ compile

    def compile_graph(self, graph) -> GraphResult:
        name = graph.name
        key = self.stage_key(graph)
        started = time.monotonic()

        onnx = self.onnx_path(name)
        if not onnx.exists() and not self.dry_run:
            return GraphResult(name, "failed", key=key,
                               note=f"missing {onnx.relative_to(self.build_dir)}; run export first")

        previous = self._state().get(name, {})
        if not self.force and previous.get("key") == key and previous.get("elf"):
            elf = self._resolve_elf(previous["elf"])
            # A state file written before paths were stored relative holds an
            # absolute path from whichever side compiled it. Rather than
            # recompile nine minutes of identical work, fall back to where the
            # convention says the ELF is -- and re-record it relative, so the
            # tree heals itself the first time it is used from the other mount.
            if (elf is None or not elf.exists()) and self.elf_path(graph).exists():
                elf = self.elf_path(graph)
            if elf is not None and elf.exists():
                self.publish(graph, elf)
                outcome = GraphResult(name, "reused", precision=previous.get("precision"),
                                      elf=str(elf), key=key, note="unchanged")
                if previous.get("elf") != self._relative_elf(str(elf)):
                    self._record(outcome)
                return outcome

        attempts: list[dict] = []
        for precision in graph.precisions:
            # afe writes into the retained directory; a leftover ELF from a
            # failed earlier attempt would otherwise be mistaken for success.
            retained = self.retained_dir(name)
            if retained.exists() and not self.dry_run:
                shutil.rmtree(retained)

            log = self.build_dir / "logs" / f"compile_{name}_{precision}.log"
            result = proc.run(self.argv(graph, precision), env=self.env, log_path=log,
                              echo=False, dry_run=self.dry_run)
            attempts.append({"precision": precision, "returncode": result.returncode,
                             "log": str(log), "duration_s": round(result.duration_s, 1)})

            if self.dry_run:
                return GraphResult(name, "skipped", precision=precision, key=key,
                                   attempts=attempts, note="dry run")

            elf = self.locate_elf(graph, precision) if result.ok else None
            if elf is None:
                # Exit code alone is not the signal: afe can return 0 having put
                # the graph on the APU, which produces no MLA ELF at all.
                attempts[-1]["reason"] = (
                    f"exit {result.returncode}" if not result.ok else "no MLA ELF produced"
                )
                continue

            self.publish(graph, elf)
            outcome = GraphResult(name, "compiled", precision=precision, elf=str(elf),
                                  key=key, duration_s=time.monotonic() - started,
                                  attempts=attempts)
            self._record(outcome)
            return outcome

        return GraphResult(name, "failed", key=key, attempts=attempts,
                           duration_s=time.monotonic() - started,
                           note=f"all precisions failed ({', '.join(graph.precisions)})")

    def run(self, only: Sequence[str] | None = None) -> list[GraphResult]:
        graphs = self.select(only)
        results = (
            self._run_sequential(graphs) if self.jobs <= 1 or len(graphs) == 1
            else self._run_parallel(graphs)
        )
        self.write_manifest(results)
        return results

    def _run_sequential(self, graphs) -> list[GraphResult]:
        results: list[GraphResult] = []
        for graph in graphs:
            result = self.compile_graph(graph)
            print(result.summary(), flush=True)
            results.append(result)
            # Stop on the first failure: the later graphs are usually variations
            # of the same one, so continuing mostly buys more copies of the same
            # error and a longer wait before you see it.
            if result.status == "failed":
                break
        return results

    def _run_parallel(self, graphs) -> list[GraphResult]:
        """Compile independent graphs at once.

        Threads rather than processes: each compile is a subprocess, so the work
        happens outside the interpreter and the GIL is irrelevant. Results are
        printed as they land, so a long graph does not hide the short ones that
        already finished.

        Unlike the sequential path this does not stop early -- the others are
        already running, and killing them would waste work that is nearly done.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        workers = min(self.jobs, len(graphs))
        print(f"  running {workers} graph(s) at a time", flush=True)
        by_name: dict[str, GraphResult] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.compile_graph, g): g for g in graphs}
            for future in as_completed(futures):
                graph = futures[future]
                try:
                    result = future.result()
                except Exception as error:            # noqa: BLE001 - reported below
                    result = GraphResult(graph.name, "failed", note=str(error))
                by_name[graph.name] = result
                print(result.summary(), flush=True)
        # Report in plan order, not completion order, so the output is stable.
        return [by_name[g.name] for g in graphs if g.name in by_name]

    def select(self, only: Sequence[str] | None) -> list:
        graphs = list(self.spec.compile.graphs)
        if not only:
            return graphs
        wanted = set(only)
        unknown = wanted - {g.name for g in graphs}
        if unknown:
            raise CompileError(
                f"unknown graph(s) {sorted(unknown)}; "
                f"have {[g.name for g in graphs]}"
            )
        return [g for g in graphs if g.name in wanted]

    def write_manifest(self, results: Iterable[GraphResult]) -> Path:
        results = list(results)
        path = self.build_dir / "artifact_manifest.json"
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing.update({
            "format": "polima-build-v1",
            "policy": getattr(self.spec, "name", "?"),
            "build_dir": str(self.build_dir),
            "sdk_version": self.sdk_version,
            "compiled_at": time.time(),
            "graphs": [asdict(r) for r in results],
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        return path


def sdk_version(compiler_python: str | Path, env: dict[str, str] | None = None) -> str:
    """afe's version string, which participates in the resume key."""
    result = proc.capture(
        [str(compiler_python), "-c",
         "from afe.apis.release_v1 import get_model_sdk_version as v; print(v())"],
        env=env,
    )
    return result.stdout.strip().splitlines()[-1] if result.ok and result.stdout.strip() else ""
