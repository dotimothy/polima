"""Bundle identity and on-disk layout.

A bundle is one deployable unit: the compiled ELFs, the runtime plan, the
constants and the fixtures needed to prove it works. Its id is content-addressed,
which is the direct answer to what the board looks like today -- 26 accreted
roots (ACT_rcwb_f_t_100000, SmolVLA_combined_035000{,_e2e,_final,_llima,_packed},
10 sima_mpk_extract* trees, ...) with no way to tell which one is live.

With a content hash:
  * redeploying an identical build is a no-op (`sync-bundle` skips it),
  * two checkpoints of the same run can never collide, and
  * `current -> bundles/<id>` makes activation and rollback an atomic `ln -sfn`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from polima.util.hashing import sha256_file, sha256_text, short
from polima.util.jsonio import dumps_compact

BUNDLE_FORMAT = "polima-bundle-v1"

#: <policy>-<dataset>-<steps>-<sha8>
BUNDLE_ID_RE = re.compile(r"^(?P<policy>[a-z0-9]+)-(?P<dataset>.+)-(?P<steps>\d+)-(?P<sha>[0-9a-f]{8})$")


def slug(value: str) -> str:
    """Filesystem- and ssh-safe token. Keeps dataset names readable."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_.-")
    return cleaned or "unknown"


def compute_bundle_id(
    *,
    policy: str,
    dataset: str,
    steps: int | str,
    graph_digests: dict[str, str],
    plan: dict | None = None,
) -> str:
    """Deterministic id from the *content* of the build.

    The hash covers the sorted (graph_name, sha256(elf)) pairs plus the canonical
    plan JSON, so two builds that produce identical ELFs and an identical
    execution plan get the same id -- even if built on different days from
    different directories.
    """
    parts = [f"{name}={graph_digests[name]}" for name in sorted(graph_digests)]
    if plan is not None:
        parts.append("plan=" + dumps_compact(plan))
    digest = sha256_text("\n".join(parts))
    return f"{slug(policy)}-{slug(dataset)}-{int(steps)}-{short(digest)}"


def parse_bundle_id(bundle_id: str) -> dict:
    match = BUNDLE_ID_RE.match(bundle_id)
    if not match:
        raise ValueError(f"not a PoLiMa bundle id: {bundle_id!r}")
    parts = match.groupdict()
    parts["steps"] = int(parts["steps"])
    return parts


def digest_graphs(elf_paths: dict[str, str | Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in elf_paths.items()}


@dataclass
class GraphArtifact:
    name: str
    elf: str                       # relative to the bundle root
    sha256: str
    elf_bytes: int
    precision: str = "bf16"
    input_elements: int = 0
    output_elements: int = 0
    dram_layout: str = "plain"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "elf": self.elf,
            "sha256": self.sha256,
            "elf_bytes": self.elf_bytes,
            "precision": self.precision,
            "input_elements": self.input_elements,
            "output_elements": self.output_elements,
            "dram_layout": self.dram_layout,
        }


@dataclass
class Bundle:
    """The manifest written to <bundle>/bundle.json.

    Replaces, all at once: input_contract.json, artifact_manifest.json,
    action_runtime_manifest.json, and the ELF paths plus element counts
    hardcoded in act_llima.cpp's ActModel constructor (lines 117-122).
    """

    root: Path
    policy: str
    bundle_id: str
    checkpoint: str = ""
    graphs: list[GraphArtifact] = field(default_factory=list)
    sidecars: list[str] = field(default_factory=list)
    source: str = "compile"                    # compile | legacy-import
    legacy_source_dir: str | None = None
    tool_versions: dict = field(default_factory=dict)
    format: str = BUNDLE_FORMAT

    # ---- on-disk layout; every consumer goes through these, never globs ----

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def constants_dir(self) -> Path:
        return self.root / "constants"

    @property
    def fixtures_dir(self) -> Path:
        return self.root / "fixtures"

    @property
    def manifest_path(self) -> Path:
        return self.root / "bundle.json"

    @property
    def plan_path(self) -> Path:
        return self.root / "plan.json"

    def model_dir(self, graph: str) -> Path:
        return self.models_dir / graph

    def model_elf(self, graph: str) -> Path:
        """Always models/<graph>/share/<graph>_stage1_mla.elf.

        Unifies the two legacy conventions: SmolVLA explodes *_mpk.tar.gz into
        etc/lib/share, ACT copies retained/<g>/<g>_stage1_mla.elf into a
        hand-made models_uncompressed/<g>/share/. Verified on the live board:
        ACT's *deployed* tree already has exactly this shape.
        """
        for artifact in self.graphs:
            if artifact.name == graph:
                return self.root / artifact.elf
        return self.model_dir(graph) / "share" / f"{graph}_stage1_mla.elf"

    def graph(self, name: str) -> GraphArtifact:
        for artifact in self.graphs:
            if artifact.name == name:
                return artifact
        raise KeyError(f"no graph {name!r} in bundle {self.bundle_id}")

    @property
    def graph_names(self) -> list[str]:
        return [artifact.name for artifact in self.graphs]

    @property
    def total_elf_bytes(self) -> int:
        return sum(artifact.elf_bytes for artifact in self.graphs)

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "bundle_id": self.bundle_id,
            "policy": self.policy,
            "checkpoint": self.checkpoint,
            "source": self.source,
            "legacy_source_dir": self.legacy_source_dir,
            "graphs": [artifact.to_dict() for artifact in self.graphs],
            "sidecars": sorted(self.sidecars),
            "tool_versions": self.tool_versions,
            "total_elf_bytes": self.total_elf_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict, root: str | Path) -> "Bundle":
        return cls(
            root=Path(root),
            policy=data["policy"],
            bundle_id=data["bundle_id"],
            checkpoint=data.get("checkpoint", ""),
            graphs=[GraphArtifact(**g) for g in data.get("graphs", [])],
            sidecars=list(data.get("sidecars", [])),
            source=data.get("source", "compile"),
            legacy_source_dir=data.get("legacy_source_dir"),
            tool_versions=data.get("tool_versions", {}),
            format=data.get("format", BUNDLE_FORMAT),
        )
