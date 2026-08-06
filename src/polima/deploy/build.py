"""Build the native runtime on the board, skipping when nothing changed.

The legacy deploy scripts run cmake unconditionally with `-j2`, so every SmolVLA
deploy rebuilds the whole server even when only the model changed. Two things
fix that here:

  * there is now ONE binary instead of four, so a policy change does not touch
    the C++ at all; and
  * the build tree is keyed by a content hash of native/, so an unchanged source
    tree skips configure and compile entirely.

The board has 16 cores (verified), not the 2 both legacy scripts assume.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from polima.config.base import BoardConfig
from polima.deploy.ssh import BoardSession
from polima.util.hashing import sha256_tree, short
from polima.util.logging import get
from polima.util.paths import native_dir

log = get("deploy.build")

BINARIES = ("polima_server", "polima_cli")


@dataclass
class BuildResult:
    source_hash: str
    build_dir: str
    skipped: bool
    binaries: list[str]

    def to_dict(self) -> dict:
        return {
            "source_hash": self.source_hash, "build_dir": self.build_dir,
            "skipped": self.skipped, "binaries": self.binaries,
        }


def source_hash(source_dir=None) -> str:
    """Content hash of native/ -- the build-skip key."""
    return short(
        sha256_tree(
            source_dir or native_dir(),
            patterns=("**/*.cpp", "**/*.hpp", "**/*.h", "**/CMakeLists.txt"),
        ),
        12,
    )


def build_native(
    session: BoardSession,
    board: BoardConfig,
    *,
    force: bool = False,
    jobs: int | None = None,
) -> BuildResult:
    """Sync native/ to the board and build it if the source changed."""
    digest = source_hash()
    build_dir = board.path("build", digest)
    sentinel = f"{build_dir}/.ok"
    jobs = jobs or board.build_jobs

    if not force and session.exists(sentinel):
        log.info("build: skipped (source hash %s unchanged)", digest)
        _link_binaries(session, board, build_dir)
        return BuildResult(digest, build_dir, True, list(BINARIES))

    log.info("build: source hash %s, compiling with -j%d", digest, jobs)
    session.mkdirs(board.native_src_dir, build_dir)
    session.rsync(native_dir(), board.native_src_dir, delete=True, echo=False)

    session.run(
        f"cmake -S {shlex.quote(board.native_src_dir)} -B {shlex.quote(build_dir)} "
        f"-DCMAKE_BUILD_TYPE=Release",
        echo=False,
    )
    session.run(f"cmake --build {shlex.quote(build_dir)} -j{jobs}", echo=True)

    missing = [
        name for name in BINARIES
        if not session.exists(f"{build_dir}/{name}")
    ]
    if missing:
        raise RuntimeError(f"build finished but produced no {', '.join(missing)}")

    session.run(f"touch {shlex.quote(sentinel)}")
    _link_binaries(session, board, build_dir)
    return BuildResult(digest, build_dir, False, list(BINARIES))


def _link_binaries(session: BoardSession, board: BoardConfig, build_dir: str) -> None:
    """bin/<name> -> build/<hash>/<name>, so callers never learn the hash."""
    for name in BINARIES:
        session.run(
            f"ln -sfn {shlex.quote(build_dir + '/' + name)} "
            f"{shlex.quote(board.path('bin', name))}"
        )


def prune_builds(session: BoardSession, board: BoardConfig, keep: int = 2) -> list[str]:
    """Drop old build trees, newest `keep` retained. Never touches `bin` links."""
    listing = session.capture(
        f"ls -1t {shlex.quote(board.path('build'))} 2>/dev/null"
    ).splitlines()
    stale = [name for name in listing[keep:] if name.strip()]
    for name in stale:
        session.run(f"rm -rf {shlex.quote(board.path('build', name))}", check=False)
    if stale:
        log.info("pruned %d old build tree(s)", len(stale))
    return stale
