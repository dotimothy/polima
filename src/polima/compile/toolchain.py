"""Locating and invoking the SiMa model-compiler venv.

The compiler is a separate interpreter from everything else PoLiMa runs in: it
has `afe`, `onnx` and `onnxsim` but no torch, while the training env has torch
but no afe. Neither can import the other's modules, so the compile stage is
always a subprocess, never an import.

`polima doctor` and `polima compile` need the same discovery, so it lives here
rather than in either.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Searched in order. The `/sdk-extensions` entry is where the SiMa container
#: mounts it, and is also the path baked into `llima-compile`'s shebang.
SEARCH_PATH = (
    "$MODEL_COMPILER_BIN",
    "~/sima-sdk-extensions/model-compiler/bin",
    "/sdk-extensions/model-compiler/bin",
)


def find_compiler_bin(config=None) -> Path | None:
    """First candidate that actually contains a `python`."""
    candidates: list[str] = []
    explicit = getattr(getattr(config, "paths", None), "model_compiler_bin", None)
    if explicit:
        candidates.append(str(explicit))
    candidates.append(os.environ.get("MODEL_COMPILER_BIN", ""))
    candidates.append(os.path.expanduser("~/sima-sdk-extensions/model-compiler/bin"))
    candidates.append("/sdk-extensions/model-compiler/bin")
    for candidate in candidates:
        if candidate and (Path(candidate) / "python").exists():
            return Path(candidate)
    return None


def require_compiler_python(config=None) -> Path:
    compiler_bin = find_compiler_bin(config)
    if compiler_bin is None:
        raise FileNotFoundError(
            "SiMa model compiler not found. Searched: "
            + ", ".join(SEARCH_PATH)
            + ".\nSet MODEL_COMPILER_BIN=/path/to/model-compiler/bin"
        )
    return compiler_bin / "python"


def compiler_env(compiler_bin: str | Path, source_root: str | Path | None = None) -> dict[str, str]:
    """Environment for a compiler subprocess.

    Two things matter here, both inherited from the legacy scripts:

    * `PATH` is prefixed with the compiler bin, because afe shells out to its own
      helper executables and finds them by name.
    * `CUDA_VISIBLE_DEVICES` is cleared. The compiler will happily grab a GPU and
      then contend with a training run on the same host; the legacy scripts set
      it empty for exactly that reason. `MODEL_COMPILER_CUDA_VISIBLE_DEVICES`
      overrides it when the GPU is genuinely wanted.
    """
    env = dict(os.environ)
    env["PATH"] = f"{compiler_bin}:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = env.get("MODEL_COMPILER_CUDA_VISIBLE_DEVICES", "")
    if source_root:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{source_root}:{existing}" if existing else str(source_root)
    return env
