"""Path resolution.

The `$HOME/MLSandbox -> SDK/NEAT/workspace/MLSandbox` symlink is load-bearing in
the legacy tree: SmolVLA/train_smolvla_local.sh hardcodes `$HOME/MLSandbox/...`,
and ACT/lerobot's git remote points at `/home/timothydo/MLSandbox/SmolVLA/lerobot`
-- i.e. through the symlink at its own sibling. Deleting the symlink silently
breaks a clone that is otherwise fine.

PoLiMa never stores that literal. Everything resolves from this file's own
location, which is the convention every existing script already follows via
BASH_SOURCE. `polima doctor` reports the resolved root and flags the symlink.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/polima/util/paths.py -> src/polima/util -> src/polima -> src -> polima/
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def package_root() -> Path:
    """The `polima/` project directory (contains pyproject.toml, native/, board/)."""
    return _PACKAGE_ROOT.parent


def repo_root() -> Path:
    """MLSandbox/ -- the parent holding ACT/, SmolVLA/, GR00T-N1.6/, lerobot_sima/."""
    override = os.environ.get("POLIMA_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return package_root().parent


def native_dir() -> Path:
    return package_root() / "native"


def board_scripts_dir() -> Path:
    return package_root() / "board"


def legacy_stack(name: str) -> Path:
    """Path to a legacy stack dir. Read-only for PoLiMa through Phase 5."""
    return repo_root() / name


def outputs_root() -> Path:
    return Path(os.environ.get("POLIMA_OUTPUTS", package_root() / "outputs"))


def default_dataset_parent() -> Path:
    return Path(os.environ.get("POLIMA_DATASET_PARENT", "/ml_datasets"))


def symlink_report() -> dict:
    """Facts about $HOME/MLSandbox for `polima doctor`."""
    link = Path.home() / "MLSandbox"
    resolved = repo_root()
    return {
        "home_symlink": str(link),
        "exists": link.exists(),
        "is_symlink": link.is_symlink(),
        "target": str(link.resolve()) if link.exists() else None,
        "resolved_repo_root": str(resolved),
        "consistent": link.exists() and link.resolve() == resolved,
    }


def expand(path: str | Path) -> Path:
    """`~` and `$VAR` expansion, then resolve. Never resolves through to a literal
    $HOME/MLSandbox in stored config -- callers store what the user typed."""
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
