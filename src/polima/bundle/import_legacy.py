"""Adopt a pre-PoLiMa build tree as a bundle, without recompiling.

This is the single biggest de-risking move in the port: it lets Phase 1a
validate the new deploy + runtime + wire + client stack against ELFs that are
*already known to work on the board*, before asking the new compile path to be
correct. If the imported bundle serves correct actions, every failure after that
belongs to the compiler, not the runtime.

Verified against ACT/outputs/modalix_rcwb_f_t_act_100000_llima: the six ELFs
selected here are byte-identical to the six currently running on
sima@192.168.91.211 under /media/nvme/ACT_rcwb_f_t_100000.

The tree is messier than it looks. `retained/` holds 16 ELFs in 11 directories,
six of them abandoned experiments (`vision_backbone_rejected_rank4`,
`decoder_action_tail_v2`, ...), and `input_contract.json` lists 12 graphs because
it counts the `_tensor_prepared` intermediates. Neither can be globbed; both are
resolved against the PolicySpec's declared graph names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from polima.bundle import retained as retained_module
from polima.bundle.layout import Bundle
from polima.bundle.pack import BundleInputs, build_bundle
from polima.bundle.retained import ElfCandidate
from polima.bundle.unpack import find_mpk, from_mpk
from polima.policies.base import PolicySpec
from polima.util.hashing import sha256_file
from polima.util.jsonio import read_json_or, write_json
from polima.util.logging import get

log = get("bundle.import")

#: Suffixes the exporter appends to intermediate ONNX graphs; never real graphs.
INTERMEDIATE_SUFFIXES = ("_tensor_prepared", "_simplified", "_prepared")

#: Directories that hold already-unpacked models, in preference order.
DEPLOYED_TREES = ("models_uncompressed", "models")


@dataclass
class LegacyBuild:
    root: Path
    format: str
    checkpoint: str = ""
    dataset_root: str = ""
    graphs: tuple[str, ...] = ()
    verification: dict | None = None

    @property
    def dataset_name(self) -> str:
        return Path(self.dataset_root).name if self.dataset_root else "unknown"

    @property
    def steps(self) -> int:
        """Training step count, read off the checkpoint path (.../checkpoints/100000/...)."""
        match = re.search(r"/checkpoints/(\d+)/", self.checkpoint)
        return int(match.group(1)) if match else 0


def detect(build_dir: str | Path) -> LegacyBuild:
    """Identify a legacy build tree and read what it declares about itself.

    Detection order matches the three manifest formats found in the tree:
      1. input_contract.json  format "act-modalix-v1"
      2. artifact_manifest.json with an "iterations" key  (SmolVLA controller)
      3. artifact_manifest.json format "act-modalix-build-v1" / "groot-..."
    """
    root = Path(build_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"no such build directory: {root}")

    contract = read_json_or(root / "input_contract.json") or {}
    manifest = read_json_or(root / "artifact_manifest.json") or {}
    report = read_json_or(root / "onnx_verification_report.json")

    fmt = contract.get("format") or manifest.get("format")
    if not fmt and "iterations" in manifest:
        fmt = "smolvla-controller"
    if not fmt:
        # A tree built by the shell scripts rather than a controller has no
        # manifest at all -- SmolVLA's compile_deploy_smolvla_som.sh writes
        # none. If the compiled models are there, that is enough to import; the
        # things a manifest would have supplied (dataset, steps) come from
        # --dataset / --steps, and their absence is reported rather than guessed.
        if _has_models(root):
            fmt = "unlabelled"
        else:
            raise ValueError(
                f"{root} has no recognisable manifest "
                "(expected input_contract.json or artifact_manifest.json) "
                "and no models_uncompressed/, models/ or retained/ to import"
            )

    return LegacyBuild(
        root=root,
        format=fmt,
        checkpoint=contract.get("checkpoint") or manifest.get("checkpoint") or "",
        dataset_root=contract.get("dataset_root", ""),
        graphs=tuple(_declared_graphs(contract)),
        verification=report,
    )


def _has_models(root: Path) -> bool:
    """Whether the tree holds compiled models under any known layout."""
    for tree in (*DEPLOYED_TREES, "retained"):
        directory = root / tree
        if directory.is_dir() and any(directory.iterdir()):
            return True
    return any(root.glob("**/*_mpk.tar.gz"))


def _declared_graphs(contract: dict) -> list[str]:
    """Graph stems from input_contract.json, minus the intermediates.

    The real file lists 12 entries for a 6-graph policy because every graph
    appears twice: `vision_backbone.onnx` and `vision_backbone_tensor_prepared.onnx`.
    """
    names: list[str] = []
    for entry in contract.get("graphs", []):
        stem = Path(str(entry)).stem
        if any(stem.endswith(suffix) for suffix in INTERMEDIATE_SUFFIXES):
            continue
        if stem not in names:
            names.append(stem)
    return names


def resolve_elfs(build: LegacyBuild, spec: PolicySpec) -> dict[str, ElfCandidate]:
    """Find one ELF per graph the policy declares.

    Source order, most-trustworthy first:
      1. models_uncompressed/<graph>/share/  -- what the legacy deploy shipped
      2. models/<graph>/share/               -- an already-deployed tree
      3. retained/<graph>/                   -- the compiler's working dir
      4. **/<graph>_mpk.tar.gz               -- unpack on the fly

    Selection is always by the policy's declared graph name. Globbing `retained/`
    would pick up `*_rejected_*` and `*_v2` experiments.
    """
    declared = set(build.graphs)
    if declared and declared != set(spec.compile.names):
        extra = sorted(declared - set(spec.compile.names))
        absent = sorted(set(spec.compile.names) - declared)
        log.warning(
            "manifest graphs differ from policy %s: extra=%s missing=%s",
            spec.name, extra, absent,
        )

    resolved: dict[str, ElfCandidate] = {}
    for spec_graph in spec.compile.graphs:
        graph = spec_graph.name
        candidate = _resolve_one(build.root, graph)
        # Hand-built trees name directories for the build, not the graph --
        # `vision_llima_bf16` rather than `vision`. Aliases are tried in order
        # and only after the canonical name, so a tree using both is unambiguous.
        for alias in getattr(spec_graph, "legacy_names", ()):
            if candidate is not None:
                break
            candidate = _resolve_one(build.root, alias)
            if candidate is not None:
                log.info("  %-24s matched legacy name %r", graph, alias)
        if candidate is None:
            raise FileNotFoundError(
                f"no ELF for graph {graph!r} under {build.root}; looked in "
                f"{', '.join(DEPLOYED_TREES)}, retained/, and *_mpk.tar.gz"
                + (f" (also tried {list(spec_graph.legacy_names)})"
                   if spec_graph.legacy_names else "")
            )
        resolved[graph] = candidate
        log.info("  %-24s %s (%s)", graph, candidate.sha256[:12], candidate.variant)
    return resolved


def _resolve_one(root: Path, graph: str) -> ElfCandidate | None:
    for tree in DEPLOYED_TREES:
        try:
            return retained_module.from_deployed_tree(root / tree, graph)
        except FileNotFoundError:
            pass

    try:
        return retained_module.select(root / "retained", graph)
    except (FileNotFoundError, ValueError) as exc:
        log.debug("retained/ lookup for %s: %s", graph, exc)

    archive = find_mpk(root, graph)
    if archive is not None:
        destination = root / ".polima_unpacked" / graph
        result = from_mpk(archive, destination)
        elf = next((p for p in result.elfs if p.name.startswith(graph)), result.elfs[0])
        from polima.util.hashing import sha256_file

        return ElfCandidate(
            graph=graph, path=elf, variant=f"mpk:{archive.name}",
            sha256=sha256_file(elf), size=elf.stat().st_size,
        )
    return None


def collect_fixtures(build: LegacyBuild, spec: PolicySpec) -> dict[str, Path]:
    """Gather the golden inputs and expected outputs.

    The ACT build ships per-stage goldens (`<graph>_input.f32` /
    `<graph>_output.f32`) as well as end-to-end ones, which is strictly better
    than the end-to-end-only check the legacy smoke test performs: a wrong ELF
    can be identified by name instead of just failing the final cosine.
    """
    fixtures: dict[str, Path] = {}
    direct = build.root / "direct_inputs"

    if direct.is_dir():
        for index, tensor in enumerate(spec.wire.request_tensors):
            source = direct / f"vision_input_{index}.f32"
            if source.is_file():
                fixtures[f"inputs/{tensor.name}.f32"] = source
        state = direct / "state.f32"
        if state.is_file():
            fixtures["inputs/state.f32"] = state

        expected = direct / "expected_normalized_actions.f32"
        if expected.is_file():
            fixtures["expected/normalized_actions.f32"] = expected

        # Per-graph goldens, for stage-by-stage verification.
        for graph in spec.compile.names:
            for kind in ("input", "output"):
                source = direct / f"{graph}_{kind}.f32"
                if source.is_file():
                    fixtures[f"stages/{graph}_{kind}.f32"] = source
        for index in (0, 1):
            source = direct / f"vision_output_{index}.f32"
            if source.is_file():
                fixtures[f"stages/vision_output_{index}.f32"] = source

    # Newer exporters keep the end-to-end fixture in one portable NPZ.  ACT
    # also writes the legacy direct_inputs files, but SmolVLA intentionally did
    # not, which used to produce a bundle that `polima run --fixture` could not
    # consume.  Materialize the wire tensors here so both freshly compiled and
    # older exported build trees get the same runnable bundle layout.
    fixture_archive = build.root / spec.compile.fixture_file
    if fixture_archive.is_file():
        materialized = build.root / ".polima_fixtures"
        materialized.mkdir(parents=True, exist_ok=True)
        import numpy as np

        normalization: dict[str, np.ndarray] = {}
        stats_path = build.root / "normalization_stats.npz"
        if spec.name == "smolvla" and stats_path.is_file():
            with np.load(stats_path, allow_pickle=False) as stats:
                normalization = {
                    name: np.asarray(stats[name], dtype="<f4").copy()
                    for name in ("state_mean", "state_std", "action_mean", "action_std")
                }

        with np.load(fixture_archive, allow_pickle=False) as archive:
            for tensor in spec.wire.request_tensors:
                if f"inputs/{tensor.name}.f32" in fixtures:
                    continue
                if tensor.name not in archive:
                    raise ValueError(
                        f"{fixture_archive.name} has no wire tensor {tensor.name!r}"
                    )
                values = np.asarray(archive[tensor.name], dtype="<f4")
                # SmolVLA's ONNX vision graph is NCHW, while its TCP client
                # deliberately sends prepared camera frames as HWC.  Fixtures
                # exercise the wire contract, so mirror the client here rather
                # than serializing the exporter tensor's storage order.
                if (
                    spec.name == "smolvla"
                    and tensor.name.startswith("image")
                    and values.ndim == 4
                    and values.shape[0] == 1
                    and values.shape[1] == 3
                ):
                    values = values[0].transpose(1, 2, 0)
                # SmolVLA performs state normalization and action
                # denormalization inside the server plan.  Its exporter records
                # the already-normalized model input, so convert it back to the
                # raw wire value before materializing a runnable fixture.
                if spec.name == "smolvla" and tensor.name == "state":
                    if not normalization:
                        raise FileNotFoundError(
                            f"{stats_path} is required to materialize raw SmolVLA state"
                        )
                    values = (
                        values.reshape(-1) * normalization["state_std"]
                        + normalization["state_mean"]
                    )
                values = values.reshape(-1)
                if values.size != tensor.elements:
                    raise ValueError(
                        f"{fixture_archive.name}:{tensor.name} holds {values.size} "
                        f"floats; wire contract needs {tensor.elements}"
                    )
                destination = materialized / f"{tensor.name}.f32"
                values.tofile(destination)
                fixtures[f"inputs/{tensor.name}.f32"] = destination

            expected_names = (
                ("action", "normalized_actions", "normalized_action")
                if spec.name == "smolvla"
                else ("normalized_actions", "normalized_action")
            )
            expected_key = next((name for name in expected_names if name in archive), None)
            if expected_key and "expected/normalized_actions.f32" not in fixtures:
                expected = np.asarray(archive[expected_key], dtype="<f4").reshape(-1)
                if spec.name == "smolvla" and expected_key != "action":
                    if not normalization:
                        raise FileNotFoundError(
                            f"{stats_path} is required to materialize raw SmolVLA actions"
                        )
                    expected = (
                        expected.reshape(-1, spec.dataset.action_dim)
                        * normalization["action_std"]
                        + normalization["action_mean"]
                    ).reshape(-1)
                if expected.size != spec.wire.response_elements:
                    raise ValueError(
                        f"{fixture_archive.name}:{expected_key} holds {expected.size} "
                        f"floats; wire response needs {spec.wire.response_elements}"
                    )
                destination = materialized / "normalized_actions.f32"
                expected.tofile(destination)
                fixtures["expected/normalized_actions.f32"] = destination

    for name in ("normalization_stats.npz", spec.compile.fixture_file):
        source = build.root / name
        if source.is_file():
            fixtures[name] = source
    return fixtures


def collect_sidecars(build: LegacyBuild, spec: PolicySpec) -> dict[str, Path]:
    """Collect runtime constants from a hand-built policy tree.

    ACT carries normalization as a client-side fixture, but SmolVLA's runtime
    plan consumes eight named server-side buffers. Its legacy tree stores four
    directly as ``*.f32`` and packs the remaining means/stds into one 24-float
    file. Materialize the packed values under the names in ``plan.json`` so a
    legacy import is a genuinely deployable bundle, not just four ELFs.
    """
    names = tuple(spec.build_runtime_plan().sidecars)
    if not names:
        return {}

    constants_dir = build.root / "constants"
    collected: dict[str, Path] = {}
    for name in names:
        for candidate in (constants_dir / name, constants_dir / f"{name}.f32"):
            if candidate.is_file():
                collected[name] = candidate
                break

    normalization_names = ("state_mean", "state_std", "action_mean", "action_std")
    widths = (
        spec.dataset.state_dim,
        spec.dataset.state_dim,
        spec.dataset.action_dim,
        spec.dataset.action_dim,
    )
    missing_normalization = [
        name for name in normalization_names if name in names and name not in collected
    ]
    packed = constants_dir / "normalization_stats.f32"
    if missing_normalization and packed.is_file():
        import numpy as np

        values = np.fromfile(packed, dtype="<f4")
        expected = sum(widths)
        if values.size != expected:
            raise ValueError(
                f"{packed} has {values.size} float32 values; expected {expected} "
                "(state mean/std, action mean/std)"
            )
        staged = build.root / ".polima_sidecars"
        staged.mkdir(parents=True, exist_ok=True)
        offset = 0
        for name, width in zip(normalization_names, widths):
            target = staged / name
            target.write_bytes(values[offset:offset + width].tobytes())
            collected[name] = target
            offset += width

    missing = [name for name in names if name not in collected]
    if missing:
        raise FileNotFoundError(
            f"missing runtime sidecar(s) {missing} under {constants_dir}"
        )
    return collected


def collect_robot_files(spec: PolicySpec) -> dict[str, Path]:
    """The proven board-side LeRobot client for this compiled policy.

    These files used to be copied only by the legacy deploy scripts, which
    meant a PoLiMa bundle could serve inference but could not drive the arm.
    Keeping them inside the bundle makes ``polima robot run`` independent of
    whichever hand-built model trees happen to remain on a particular board.
    """
    from polima.util.paths import repo_root

    stack_name = (spec.train.repo_dir_hint or "").split("/", 1)[0]
    stack = repo_root() / stack_name
    if spec.name == "act":
        candidates = {
            "start.sh": stack / "scripts/start_act_robot_client_on_som.sh",
            "preview_robot_cameras.py": stack / "preview_robot_cameras.py",
            "run_robot_client_with_live_view.py": stack / "run_robot_client_with_live_view.py",
            "scripts/act_som_client.py": stack / "scripts/act_som_client.py",
            "robot_rest_position.json": stack / "robot_rest_position.json",
            "camera_focus_config.json": stack / ".camera_focus_config.json",
            "calibration/so-arm101.json": (
                stack / "calibration/robots/so_follower/so-arm101.json"
            ),
        }
    elif spec.name == "smolvla":
        candidates = {
            "start.sh": stack / "scripts/start_smolvla_robot_client_on_som.sh",
            "preview_robot_cameras.py": stack / "preview_robot_cameras.py",
            "run_robot_client_with_live_view.py": stack / "run_robot_client_with_live_view.py",
            "scripts/smolvla_som_client.py": stack / "scripts/smolvla_som_client.py",
            "robot_rest_position.json": stack / "robot_rest_position.json",
            "camera_focus_config.json": stack / ".camera_focus_config.json",
            "calibration/so-arm101.json": (
                stack / "calibration/robots/so_follower/so-arm101.json"
            ),
        }
    else:
        return {}

    missing = [str(path) for path in candidates.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{spec.name} robot client is incomplete; missing " + ", ".join(missing)
        )
    return candidates


def hydrate_robot_client(bundle: Bundle, spec: PolicySpec) -> bool:
    """Upgrade an already-packed local bundle with the board client.

    Bundle ids intentionally describe model arithmetic, not operational
    launchers. This lets a deploy made with a newer PoLiMa add the completed
    device client without recompiling identical ELFs or changing their id.
    Returns whether the manifest changed.
    """
    import shutil

    files = collect_robot_files(spec)
    recorded = set(bundle.sidecars)
    changed = False
    for relative, source in files.items():
        destination = bundle.root / "robot_client" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            shutil.copy2(source, destination)
            changed = True
        recorded.add(str(destination.relative_to(bundle.root)))

    if spec.name == "act":
        fixture = bundle.root / "fixtures/normalization_stats.npz"
        destination = bundle.root / "normalization_stats.npz"
        if fixture.is_file():
            if not destination.is_file() or sha256_file(destination) != sha256_file(fixture):
                shutil.copy2(fixture, destination)
                changed = True
            recorded.add("normalization_stats.npz")

    sidecars = sorted(recorded)
    if sidecars != bundle.sidecars:
        bundle.sidecars = sidecars
        changed = True
    if changed:
        write_json(bundle.manifest_path, bundle.to_dict())
    return changed


def hydrate_runtime_metadata(bundle: Bundle, spec: PolicySpec) -> bool:
    """Upgrade graph layout metadata without touching content-addressed ELFs.

    Early bundles did not record the compiler's physical output contract.
    PoLiMa copies the current policy declaration into ``bundle.json`` so its
    generic runner decodes each graph exactly as the compiled ELF emits it.
    """
    changed = False
    declared = {graph.name: graph for graph in spec.compile.graphs}
    for artifact in bundle.graphs:
        graph = declared.get(artifact.name)
        if graph is None:
            continue
        output = graph.outputs[0]
        values = {
            "dram_layout": output.dram_layout,
            "logical_width": output.logical_width,
            "logical_channels": output.logical_channels,
            "external_dram_layout": graph.external_dram_layout,
        }
        for field, value in values.items():
            if getattr(artifact, field) != value:
                setattr(artifact, field, value)
                changed = True
    if bundle.smoke != spec.smoke.to_dict():
        bundle.smoke = spec.smoke.to_dict()
        changed = True
    if changed:
        write_json(bundle.manifest_path, bundle.to_dict())
    return changed


def import_legacy(
    build_dir: str | Path,
    spec: PolicySpec,
    *,
    output_root: str | Path,
    dataset: str | None = None,
    steps: int | None = None,
    source: str = "legacy-import",
) -> Bundle:
    """Turn a build tree into a PoLiMa bundle.

    `source` records provenance in bundle.json. It defaults to `legacy-import`
    because that is what this started as, but `polima compile` passes
    `polima-compile` for a tree it built itself -- the packing is identical
    either way (that is the point of keeping the tree layouts the same), and
    mislabelling a fresh build as an import would make the record useless.
    """
    build = detect(build_dir)
    log.info("importing %s build from %s", build.format, build.root)

    elfs = resolve_elfs(build, spec)
    fixtures = collect_fixtures(build, spec)
    constants = collect_sidecars(build, spec)
    robot_files = collect_robot_files(spec)

    tool_versions = {"source_format": build.format}
    if build.verification:
        tool_versions["onnx_max_abs"] = build.verification.get("max_abs")

    bundle = build_bundle(
        BundleInputs(
            spec=spec,
            elfs=elfs,
            dataset=dataset or build.dataset_name,
            steps=steps if steps is not None else build.steps,
            checkpoint=build.checkpoint,
            fixtures=fixtures,
            constants=constants,
            robot_files=robot_files,
            source=source,
            legacy_source_dir=str(build.root),
            tool_versions=tool_versions,
        ),
        output_root,
    )
    return bundle
