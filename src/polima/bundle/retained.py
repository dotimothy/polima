"""Source ELFs from a compiler build tree.

The compiler leaves MLA ELFs in two places, and the legacy stacks each picked a
different one:

  retained/<variant>/<graph>_stage1_mla.elf     ACT's compile_deploy path
  <graph>_mpk.tar.gz                            SmolVLA's (see unpack.py)

`retained/` is a *working* directory, not a curated one. The real ACT build
contains 16 ELFs in 11 directories, including abandoned experiments:

    vision_backbone/               <- shipped
    vision_backbone_rank3/         <- identical to the above
    vision_backbone_rejected_rank4/
    decoder_action_tail/           <- shipped
    decoder_action_tail_v2/        <- identical to the above
    decoder_action_tail_rejected_apu/

So a glob over `retained/*` yields decoys. Selection is always by explicit graph
name, and `find_variants` exists to make the ambiguity visible rather than
silently picking the first match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polima.util.hashing import sha256_file

#: Directory-name suffixes that mark an experiment the build did not ship.
REJECTED_MARKERS = ("_rejected", "_reject", "_broken", "_wip", "_old")


@dataclass(frozen=True)
class ElfCandidate:
    graph: str
    path: Path
    variant: str
    sha256: str
    size: int

    @property
    def rejected(self) -> bool:
        return any(marker in self.variant for marker in REJECTED_MARKERS)


def elf_filename(graph: str) -> str:
    return f"{graph}_stage1_mla.elf"


def find_variants(retained_dir: str | Path, graph: str) -> list[ElfCandidate]:
    """Every ELF in retained/ that could be `graph`, newest-looking last.

    A variant directory is any directory whose name starts with the graph name,
    so `vision_backbone_rank3` is a candidate for `vision_backbone` but
    `encoder_layer_01` is not a candidate for `encoder_layer_0`.
    """
    retained_dir = Path(retained_dir)
    if not retained_dir.is_dir():
        return []

    filename = elf_filename(graph)
    candidates: list[ElfCandidate] = []
    for child in sorted(retained_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name != graph and not child.name.startswith(f"{graph}_"):
            continue
        elf = child / filename
        if not elf.is_file():
            continue
        candidates.append(
            ElfCandidate(
                graph=graph,
                path=elf,
                variant=child.name,
                sha256=sha256_file(elf),
                size=elf.stat().st_size,
            )
        )
    return candidates


def select(retained_dir: str | Path, graph: str) -> ElfCandidate:
    """Pick the ELF for `graph`, preferring the exactly-named directory.

    Preference order:
      1. retained/<graph>/          -- the canonical, unsuffixed name
      2. the single non-rejected variant, if there is exactly one
      3. otherwise: raise, listing what was found

    Verified against the real ACT build: rule 1 selects the ELF that is
    byte-identical to what is deployed and running on the board.
    """
    candidates = find_variants(retained_dir, graph)
    if not candidates:
        raise FileNotFoundError(
            f"no {elf_filename(graph)} under {retained_dir} for graph {graph!r}"
        )

    exact = [c for c in candidates if c.variant == graph]
    if exact:
        return exact[0]

    viable = [c for c in candidates if not c.rejected]
    if len(viable) == 1:
        return viable[0]

    raise ValueError(
        f"ambiguous ELF for graph {graph!r} in {retained_dir}: "
        + ", ".join(f"{c.variant}({c.sha256[:8]})" for c in candidates)
        + " -- no directory is named exactly {graph!r}; pass an explicit path"
    )


def from_deployed_tree(models_dir: str | Path, graph: str) -> ElfCandidate:
    """Read an ELF from an already-unpacked models/<graph>/share/ tree.

    This is the layout ACT's deploy script produces (models_uncompressed/ on the
    host, models/ on the board) and is also PoLiMa's own bundle layout, verified
    identical on the live unit.
    """
    models_dir = Path(models_dir)
    elf = models_dir / graph / "share" / elf_filename(graph)
    if not elf.is_file():
        raise FileNotFoundError(f"no ELF at {elf}")
    return ElfCandidate(
        graph=graph,
        path=elf,
        variant="deployed",
        sha256=sha256_file(elf),
        size=elf.stat().st_size,
    )
