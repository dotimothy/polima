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
from polima.util.jsonio import read_json_or
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

    for name in ("normalization_stats.npz", "act_fixture.npz"):
        source = build.root / name
        if source.is_file():
            fixtures[name] = source
    return fixtures


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
            source=source,
            legacy_source_dir=str(build.root),
            tool_versions=tool_versions,
        ),
        output_root,
    )
    return bundle
