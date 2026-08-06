"""A pure-Python implementation of the plan interpreter.

Mirrors native/src/plan.cpp opcode for opcode. It exists for three reasons:

  * the host-side opcodes (pack, gather_strided, slice, scale, ...) can be
    verified against the real per-stage golden .f32 files *without an MLA*, so
    plan semantics are proven before anything is built on the board;
  * `polima run --stub` can serve a bundle on a workstation, with graph outputs
    supplied from the goldens, to exercise clients end to end; and
  * when a board result disagrees with expectation, running the same plan here
    localizes the disagreement to a specific step.

It is a reference, not a fast path: no MLA, no bf16 round-trip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from polima.util.jsonio import read_json


class PlanError(RuntimeError):
    pass


@dataclass
class StubPlan:
    """A plan.json loaded for host-side execution."""

    buffers: dict[str, int]
    steps: list[dict]
    result: str
    wire: dict = field(default_factory=dict)
    constants_dir: Path | None = None

    @classmethod
    def load(cls, bundle_root: str | Path) -> "StubPlan":
        root = Path(bundle_root)
        data = read_json(root / "plan.json")
        return cls(
            buffers=dict(data["buffers"]),
            steps=list(data["steps"]),
            result=data["result"],
            wire=data.get("wire", {}),
            constants_dir=root / "constants",
        )

    def run(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        graph_fn: Callable[[str, np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Execute the plan.

        `graph_fn(graph_name, input_array) -> output_array` supplies the MLA.
        Without it, `run_elf` steps raise -- pass one backed by golden .f32 files
        (see `golden_graph_fn`) to run a full pipeline on the host.
        """
        state: dict[str, np.ndarray] = {
            name: np.zeros(size, dtype=np.float32) for name, size in self.buffers.items()
        }
        for name, array in inputs.items():
            if name not in state:
                raise PlanError(f"input {name!r} is not a declared buffer")
            flat = np.asarray(array, dtype=np.float32).reshape(-1)
            if flat.size != state[name].size:
                raise PlanError(
                    f"input {name!r}: expected {state[name].size} elements, got {flat.size}"
                )
            state[name] = flat.copy()

        for index, step in enumerate(self.steps):
            try:
                self._run_step(step, state, graph_fn)
            except Exception as exc:
                raise PlanError(f"step {index} ({step['op']} -> {step['out']}): {exc}") from exc
        return state[self.result]

    # -- opcodes; keep in lockstep with native/src/plan.cpp ------------------

    def _run_step(self, step, state, graph_fn) -> None:
        op = step["op"]
        args = step.get("args", {})
        out = state[step["out"]]

        def source() -> np.ndarray:
            name = args.get("src") or (args.get("in") or [None])[0]
            if name is None:
                raise PlanError("step names no source buffer")
            return state[name]

        if op == "run_elf":
            if graph_fn is None:
                raise PlanError(f"no graph_fn supplied for {args['graph']!r}")
            joined = np.concatenate([state[n] for n in args["in"]])
            produced = np.asarray(graph_fn(args["graph"], joined), dtype=np.float32).reshape(-1)
            if produced.size != out.size:
                raise PlanError(
                    f"graph {args['graph']!r} produced {produced.size} elements, "
                    f"buffer holds {out.size}"
                )
            state[step["out"]] = produced

        elif op == "pack":
            packed = np.zeros(args.get("size", out.size), dtype=np.float32)
            for part in args["parts"]:
                start = part["dst_offset"]
                count = part["count"]
                packed[start:start + count] = state[part["src"]][:count]
            state[step["out"]] = packed

        elif op == "slice":
            offset = args.get("offset", 0)
            count = args.get("count", out.size)
            state[step["out"]] = source()[offset:offset + count].copy()

        elif op == "gather_strided":
            stride, take, count = args["stride"], args["take"], args["count"]
            block = source()[: count * stride].reshape(count, stride)
            state[step["out"]] = block[:, :take].reshape(-1).copy()

        elif op == "scale":
            state[step["out"]] = source() * np.float32(args["scalar"])

        elif op == "matvec":
            x = source()
            weights = self._constant(args["weights"])
            rows = args.get("rows", out.size)
            cols = args.get("cols", x.size)
            result = weights.reshape(rows, cols) @ x[:cols]
            if args.get("bias"):
                result = result + self._constant(args["bias"])[:rows]
            state[step["out"]] = result.astype(np.float32)

        elif op == "sincos_time":
            t = np.float32(args["scalar"])
            half = out.size // 2
            index = np.arange(half, dtype=np.float32)
            period = np.power(np.float32(10000.0), index / max(half, 1))
            state[step["out"]] = np.concatenate(
                [np.sin(t / period), np.cos(t / period)]
            ).astype(np.float32)

        elif op == "euler":
            state[step["out"]] = out - np.float32(args["scalar"]) * source()

        elif op in ("normalize", "denormalize"):
            values = source()
            mean = self._constant(args["mean"])
            deviation = self._constant(args["std"])
            tiled_mean = np.resize(mean, values.size)
            tiled_std = np.resize(deviation, values.size)
            state[step["out"]] = (
                (values - tiled_mean) / tiled_std
                if op == "normalize"
                else values * tiled_std + tiled_mean
            ).astype(np.float32)

        else:
            raise PlanError(f"unknown opcode {op!r}")

    def _constant(self, name: str) -> np.ndarray:
        if self.constants_dir is None:
            raise PlanError(f"plan needs constant {name!r} but no constants dir is set")
        path = self.constants_dir / name
        if not path.is_file():
            raise PlanError(f"missing sidecar {path}")
        return np.fromfile(path, dtype="<f4")


def golden_graph_fn(stages_dir: str | Path) -> Callable[[str, np.ndarray], np.ndarray]:
    """Back `run_elf` with recorded per-graph outputs.

    The ACT build ships `<graph>_output.f32` for every stage plus
    `vision_output_{0,1}.f32`, so the whole pipeline can be replayed on the host
    with no accelerator. Vision is called twice with different images, so calls
    to it are served in order.
    """
    stages = Path(stages_dir)
    calls: dict[str, int] = {}

    def resolve(graph: str, _input: np.ndarray) -> np.ndarray:
        occurrence = calls.get(graph, 0)
        calls[graph] = occurrence + 1
        candidates = [
            stages / f"{graph}_output_{occurrence}.f32",
            stages / f"vision_output_{occurrence}.f32" if graph == "vision_backbone" else None,
            stages / f"{graph}_output.f32",
        ]
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return np.fromfile(candidate, dtype="<f4")
        raise PlanError(f"no golden output for graph {graph!r} (call {occurrence}) in {stages}")

    return resolve


def euler_reference(x: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    """Kept beside the opcode so SmolVLA's integration can be unit-tested."""
    return (x - np.float32(dt) * velocity).astype(np.float32)


__all__ = ["StubPlan", "PlanError", "golden_graph_fn", "euler_reference", "math"]
