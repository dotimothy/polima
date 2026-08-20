"""Assemble a deployable bundle.

One convention, always: <bundle>/models/<graph>/{etc,lib,share}/. The two legacy
conventions (SmolVLA's exploded *_mpk.tar.gz, ACT's hand-copied
models_uncompressed/) both land here, so nothing downstream ever branches on
where an ELF came from.

Everything the board needs is inside the bundle and nothing else is: no ONNX, no
checkpoints, no calibration .npz, and above all no .tar.gz -- the deploy step
refuses to ship archives, because the direct-MLA runtime never opens one.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from polima.bundle.layout import Bundle, GraphArtifact, compute_bundle_id
from polima.bundle.retained import ElfCandidate
from polima.policies.base import PolicySpec, RuntimePlan
from polima.util.hashing import sha256_file
from polima.util.jsonio import write_json
from polima.util.logging import get

log = get("bundle.pack")


@dataclass
class BundleInputs:
    """Everything needed to materialize a bundle."""

    spec: PolicySpec
    elfs: dict[str, ElfCandidate]
    dataset: str
    steps: int
    checkpoint: str = ""
    #: destination-relative path -> source file
    fixtures: dict[str, Path] = field(default_factory=dict)
    constants: dict[str, Path] = field(default_factory=dict)
    #: Board-side LeRobot launcher and policy transport, relative to robot_client/.
    robot_files: dict[str, Path] = field(default_factory=dict)
    #: graph -> extra files to place beside the ELF (etc/*.json, lib/*.so)
    model_extras: dict[str, list[Path]] = field(default_factory=dict)
    source: str = "compile"
    legacy_source_dir: str | None = None
    tool_versions: dict = field(default_factory=dict)
    precisions: dict[str, str] = field(default_factory=dict)


def build_bundle(inputs: BundleInputs, output_root: str | Path) -> Bundle:
    """Write the bundle tree and return its manifest.

    The bundle id is computed from ELF content plus the canonical plan, so an
    identical rebuild produces an identical id and `polima deploy` becomes a
    no-op.
    """
    spec = inputs.spec
    missing = [name for name in spec.compile.names if name not in inputs.elfs]
    if missing:
        raise ValueError(f"missing ELF(s) for {missing}; have {sorted(inputs.elfs)}")

    plan = spec.build_runtime_plan()
    _check_plan_against_elfs(spec, plan, inputs.elfs)

    bundle_id = compute_bundle_id(
        policy=spec.name,
        dataset=inputs.dataset,
        steps=inputs.steps,
        graph_digests={name: candidate.sha256 for name, candidate in inputs.elfs.items()},
        plan=_plan_dict(plan),
    )

    root = Path(output_root).resolve() / bundle_id
    if root.exists():
        log.info("bundle %s already materialized at %s", bundle_id, root)
    root.mkdir(parents=True, exist_ok=True)

    artifacts: list[GraphArtifact] = []
    for graph in spec.compile.graphs:
        candidate = inputs.elfs[graph.name]
        model_dir = root / "models" / graph.name
        for sub in ("etc", "lib", "share"):
            (model_dir / sub).mkdir(parents=True, exist_ok=True)

        destination = model_dir / "share" / f"{graph.name}_stage1_mla.elf"
        _copy(candidate.path, destination)

        for extra in inputs.model_extras.get(graph.name, []):
            sub = {".json": "etc", ".yaml": "etc", ".yml": "etc", ".so": "lib"}.get(
                extra.suffix.lower()
            )
            if sub:
                _copy(extra, model_dir / sub / extra.name)

        artifacts.append(
            GraphArtifact(
                name=graph.name,
                elf=str(destination.relative_to(root)),
                sha256=candidate.sha256,
                elf_bytes=destination.stat().st_size,
                precision=inputs.precisions.get(
                    graph.name,
                    (
                        f"activation={graph.activation_precision or graph.precision},"
                        f"weight={graph.weight_precision or graph.precision}"
                        if graph.activation_precision or graph.weight_precision
                        else graph.precision
                    ),
                ),
                input_elements=sum(t.elements for t in graph.inputs),
                output_elements=sum(t.elements for t in graph.outputs),
                dram_layout=graph.outputs[0].dram_layout,
                logical_width=graph.outputs[0].logical_width,
                logical_channels=graph.outputs[0].logical_channels,
                external_dram_layout=graph.external_dram_layout,
            )
        )

    for relative, source in sorted(inputs.fixtures.items()):
        _copy(source, root / "fixtures" / relative)
    for relative, source in sorted(inputs.constants.items()):
        _copy(source, root / "constants" / relative)
    for relative, source in sorted(inputs.robot_files.items()):
        _copy(source, root / "robot_client" / relative)

    # ACT's proven board client reads this before opening the socket. Keep the
    # canonical fixture copy and expose a root-level compatibility path so the
    # bundled launcher remains byte-for-byte the hardware-tested one.
    client_stats = inputs.fixtures.get("normalization_stats.npz")
    if spec.name == "act" and client_stats:
        _copy(client_stats, root / "normalization_stats.npz")

    # Sidecars are recorded relative to the bundle so the C++ side can resolve
    # them without knowing anything about the host layout.
    sidecars = sorted([
        *(str((root / "constants" / relative).relative_to(root))
          for relative in inputs.constants),
        *(str((root / "robot_client" / relative).relative_to(root))
          for relative in inputs.robot_files),
        *(["normalization_stats.npz"] if spec.name == "act" and client_stats else []),
    ])

    bundle = Bundle(
        root=root,
        policy=spec.name,
        bundle_id=bundle_id,
        checkpoint=inputs.checkpoint,
        graphs=artifacts,
        sidecars=sidecars,
        source=inputs.source,
        legacy_source_dir=inputs.legacy_source_dir,
        tool_versions=inputs.tool_versions,
        smoke=spec.smoke.to_dict(),
    )

    write_json(bundle.plan_path, _plan_dict(plan, wire=spec))
    write_json(bundle.manifest_path, bundle.to_dict())
    _assert_no_archives(root)

    log.info(
        "bundle %s: %d graphs, %.1f MiB of ELF, %d fixture(s)",
        bundle_id, len(artifacts), bundle.total_elf_bytes / 1048576, len(inputs.fixtures),
    )
    return bundle


def _plan_dict(plan: RuntimePlan, wire: PolicySpec | None = None) -> dict:
    """The JSON the native interpreter reads.

    Kept flat and explicit -- plan.cpp parses this with nlohmann/json and must
    never have to infer anything.
    """
    data = {
        "buffers": dict(plan.buffers),
        "steps": [{"op": s.op, "out": s.out, "args": dict(s.args)} for s in plan.steps],
        "result": plan.result,
        "sidecars": list(plan.sidecars),
        "loops": dict(plan.loops),
    }
    if wire is not None:
        data["wire"] = {
            "magic": wire.wire.magic,
            "version": wire.wire.version,
            "default_port": wire.wire.default_port,
            "request_tensors": [
                {"name": t.name, "elements": t.elements, "shape": list(t.shape)}
                for t in wire.wire.request_tensors
            ],
            "response_shape": list(wire.wire.response_shape),
            "response_elements": wire.wire.response_elements,
        }
        # The robot description crosses to the board too. Without it the board
        # would need the Python PolicySpec just to learn that the wrist camera
        # is the Sonix one -- and the board deliberately has no polima Python.
        robot = wire.robot
        data["robot"] = {
            "camera_roles": [list(pair) for pair in robot.camera_roles],
            "camera_hints": dict(robot.camera_hints),
            "camera_fourcc": robot.camera_fourcc,
            "joint_names": list(robot.joint_names),
            "actions_per_chunk": robot.actions_per_chunk,
            "fps": robot.default_fps,
            "max_relative_target": robot.max_relative_target,
            "aggregate_fn": robot.aggregate_fn,
            "calibration_id": robot.calibration_id,
        }
    return data


def _check_plan_against_elfs(
    spec: PolicySpec, plan: RuntimePlan, elfs: dict[str, ElfCandidate]
) -> None:
    """Every accelerator step must name graphs for which ELFs were supplied."""
    for index, step in enumerate(plan.steps):
        if step.op not in ("run_elf", "run_elf_chain"):
            continue
        graphs = (
            (step.args.get("graph"),)
            if step.op == "run_elf" else tuple(step.args.get("graphs", ()))
        )
        for graph in graphs:
            if graph not in elfs:
                raise ValueError(
                    f"plan step {index} runs graph {graph!r} but no ELF was supplied"
                )
            _ = spec.compile.graph(graph)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.exists()
        and destination.stat().st_size == source.stat().st_size
        and sha256_file(destination) == sha256_file(source)
    ):
        return
    shutil.copy2(source, destination)


def _assert_no_archives(root: Path) -> None:
    """A bundle must never contain a .tar.gz.

    Both legacy deploy scripts guard for this on the board; PoLiMa catches it one
    step earlier, at pack time, where the message can point at the offending file.
    """
    archives = [p for p in root.rglob("*.tar.gz")]
    if archives:
        raise RuntimeError(
            "bundle contains archives, which the direct-MLA runtime cannot read: "
            + ", ".join(str(p.relative_to(root)) for p in archives)
        )
