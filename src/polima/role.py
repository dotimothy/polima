"""Which half of PoLiMa is installed here.

PoLiMa spans two machines with incompatible dependency sets:

    HOST  (x86_64 workstation / VM)      BOARD (Modalix SoM, aarch64)
    ------------------------------      ----------------------------
    polima-train    torch, lerobot      polima         numpy
    polima-compile  afe, onnx, onnxsim  polima-run     numpy
    polima-deploy   ssh, rsync          polima robot   flask, cv2, lerobot 0.4.4

Both halves share one core -- policy specs, wire protocol, bundle layout, config
-- because that core *is* the contract between them: the host writes bundle.json
and plan.json, the board reads them. The core therefore imports with stdlib +
numpy alone, and `polima doctor` proves it in every interpreter.

This module reports what the local install can actually do, so a command that
cannot work here says so plainly instead of dying on an ImportError three frames
deep in a vendored library.
"""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from pathlib import Path

HOST = "host"
BOARD = "board"
BOTH = "both"
UNKNOWN = "unknown"

#: Commands that only make sense on the machine that trains and compiles.
HOST_COMMANDS = ("train", "compile", "deploy")
#: Commands that run where the MLA and the arm are.
BOARD_COMMANDS = ("run", "robot")
#: Commands that work anywhere.
SHARED_COMMANDS = ("doctor", "data", "list")

_CAPABILITY_MODULES = {
    "torch": "train",
    "afe": "compile",
    "onnx": "compile",
    "flask": "robot live view",
    "cv2": "robot cameras",
    "pyarrow": "dataset task validation",
    "yaml": "yaml config files",
}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def is_board() -> bool:
    """True on a Modalix SoM.

    aarch64 alone is not sufficient (an ARM laptop is not a board), so this also
    looks for the MLA runtime headers that only exist on the SoM.
    """
    if platform.machine() != "aarch64":
        return False
    return Path("/usr/include/sima_lmm/mla_model.hpp").exists() or Path(
        "/media/nvme"
    ).is_dir()


@dataclass(frozen=True)
class Capabilities:
    role: str
    machine: str
    can_train: bool
    can_compile: bool
    can_deploy: bool
    can_run: bool
    can_robot: bool
    missing: dict

    def allows(self, command: str) -> bool:
        return {
            "train": self.can_train,
            "compile": self.can_compile,
            "deploy": self.can_deploy,
            "run": self.can_run,
            "robot": self.can_robot,
        }.get(command, True)

    def explain(self, command: str) -> str:
        needs = {
            "train": "torch + lerobot (conda env `act`); install with polima[host]",
            "compile": "the SiMa model-compiler venv (afe, onnx, onnxsim); "
                       "set MODEL_COMPILER_BIN=/path/to/model-compiler/bin",
            "deploy": "ssh and rsync on PATH",
            "run": "numpy (always available) -- this should not fail",
            "robot": "flask + opencv + lerobot; install with polima[robot]",
        }
        return needs.get(command, "")


def check(command: str, argv: list[str], module: object = None) -> str | None:
    """Decide whether `command` can run here. Returns None to allow.

    The capability a command needs depends on *what it is being asked to do*,
    not just its name: `polima-compile --import-legacy` copies already-built
    ELFs and needs nothing but numpy, while `polima-compile` proper needs the
    whole SiMa model-compiler venv. A command module refines this by defining

        def needs_capability(argv) -> str | None

    returning the capability actually required, or None for "no gate".
    """
    resolver = getattr(module, "needs_capability", None)
    capability = resolver(argv) if callable(resolver) else command
    if capability is None:
        return None

    capabilities = detect()
    if capabilities.allows(capability):
        return None
    return (
        f"not available on this machine (role: {capabilities.role} on "
        f"{capabilities.machine}).\n  needs: {capabilities.explain(capability)}"
    )


def detect() -> Capabilities:
    from shutil import which

    from polima.compile.toolchain import find_compiler_bin

    board = is_board()
    can_train = _installed("torch") and _installed("lerobot")
    # Compiling does not need afe *here*: `polima compile` runs the compiler as a
    # subprocess under its own venv, because that venv has afe but no torch and
    # the training env has torch but no afe -- they can never be one interpreter.
    # So the real question is whether that venv is discoverable. Requiring an
    # in-process afe would refuse a compile that works. It is still accepted as
    # an alternative, since inside the SDK container afe *is* importable here.
    can_compile = find_compiler_bin() is not None or (_installed("afe") and _installed("onnx"))
    can_deploy = bool(which("ssh") and which("rsync"))
    can_robot = _installed("flask") and _installed("cv2")

    if board:
        role = BOTH if (can_train or can_compile) else BOARD
    elif can_train or can_compile or can_deploy:
        role = HOST
    else:
        role = UNKNOWN

    missing = {
        module: purpose for module, purpose in _CAPABILITY_MODULES.items()
        if not _installed(module)
    }
    return Capabilities(
        role=role,
        machine=platform.machine(),
        can_train=can_train,
        can_compile=can_compile,
        can_deploy=can_deploy,
        can_run=True,
        can_robot=can_robot,
        missing=missing,
    )
