"""Core configuration objects.

Replaces two things at once:

  * the ~40 `ACT_*` / `SMOLVLA_*` environment variables used as shell->Python IPC
    (ACT_SOM_DIRECT, ACT_PERSPECTIVE_CAMERA, ACT_REST_SNAP_TOLERANCE,
    ACT_AUTO_*, ACT_LIVE_PORT, ...), which exist only because
    run_robot_client_with_live_view.py has no argparse of its own; and
  * the board address / port / path constants hardcoded across ~8 legacy files.

Defaults below are the *verified* values from the live board, not guesses:
  sima@192.168.91.211, aarch64, 16 cores, /media/nvme with 407G free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Verified on the live unit, 2026-08-06.
DEFAULT_BOARD_HOST = "sima@192.168.91.211"
DEFAULT_BOARD_ROOT = "/media/nvme/polima"

# Legacy servers currently occupy 8081 (smolvla_som_server) and 8082 (act_llima).
# PoLiMa deploys default to the 809x band so both stacks can serve side by side
# during the Phase-1 parity proof.
DEFAULT_PORTS = {"act": 8092, "smolvla": 8091, "groot": 8093}


@dataclass
class BoardConfig:
    """A Modalix SoM target."""

    host: str = DEFAULT_BOARD_HOST
    root: str = DEFAULT_BOARD_ROOT
    port: int | None = None                 # None -> PolicySpec.wire.default_port
    ssh_options: tuple[str, ...] = ("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no")
    control_master: bool = True             # one multiplexed connection per deploy
    # The board has 16 cores; both legacy deploy scripts hardcode -j2.
    build_jobs: int = 8
    # Refuse to deploy unless free space is this multiple of the bundle size.
    free_space_factor: float = 1.5
    health_timeout_s: float = 30.0
    connect_timeout_s: float = 10.0

    @property
    def user(self) -> str:
        return self.host.split("@")[0] if "@" in self.host else ""

    @property
    def address(self) -> str:
        return self.host.split("@")[-1]

    def path(self, *parts: str) -> str:
        return "/".join([self.root.rstrip("/"), *(p.strip("/") for p in parts)])

    @property
    def bundles_dir(self) -> str:
        return self.path("bundles")

    @property
    def current_link(self) -> str:
        return self.path("current")

    @property
    def native_src_dir(self) -> str:
        return self.path("src", "native")

    @property
    def bin_dir(self) -> str:
        return self.path("bin")

    @property
    def venv_dir(self) -> str:
        return self.path("venv")

    @property
    def log_dir(self) -> str:
        return self.path("var", "log")

    @property
    def run_dir(self) -> str:
        return self.path("var", "run")


@dataclass
class PathsConfig:
    """Host-side locations. The training host keeps its existing layout --
    /ml_datasets and the conda envs are NOT relocated to /media/nvme; that
    applies to the board only."""

    dataset_parent: Path = Path("/ml_datasets")
    outputs: Path | None = None             # None -> polima/outputs
    build_root: Path | None = None          # None -> <outputs>/build
    # $HOME/sima-sdk-extensions/model-compiler/bin -- overridable by
    # $MODEL_COMPILER_BIN, which the legacy controllers already honour.
    model_compiler_bin: Path | None = None
    lerobot_dir: Path | None = None         # host clone used for training


@dataclass
class SiteConfig:
    """Everything not owned by a single policy."""

    board: BoardConfig = field(default_factory=BoardConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    conda_base: Path | None = None
    offline: bool = False                   # sets the HF_*_OFFLINE trio
    dry_run: bool = False
    log_level: str = "INFO"

    def board_port(self, fallback: int) -> int:
        return self.board.port if self.board.port is not None else fallback
