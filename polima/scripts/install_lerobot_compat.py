"""Install a self-contained LeRobot wheel with the proven export stack.

The local checkout contains the policy code used by the checkpoints, but its
metadata now requires newer Torch/Transformers versions whose SmolVLA prefix
packing is numerically different. Build from an isolated copy and relax only
those metadata ranges; never modify the training checkout itself.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPLACEMENTS = {
    '"torch>=2.7,<2.12.0"': '"torch>=2.3.1,<2.12.0"',
    '"torchvision>=0.22.0,<0.27.0"': '"torchvision>=0.18.1,<0.27.0"',
    '"numpy>=2.0.0,<2.3.0"': '"numpy>=1.26.4,<2.3.0"',
    '"huggingface-hub>=1.0.0,<2.0.0"': '"huggingface-hub>=0.36.2,<2.0.0"',
    '"packaging>=24.2,<26.0"': '"packaging>=24.2,<27.0"',
    'transformers-dep = ["transformers>=5.4.0,<5.6.0"]': (
        'transformers-dep = ["transformers>=4.57.1,<5.6.0"]'
    ),
    'av-dep = ["av>=15.0.0,<16.0.0"]': 'av-dep = ["av>=15.0.0,<19.0.0"]',
}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: install_lerobot_compat.py SOURCE CONSTRAINTS", file=sys.stderr)
        return 2
    source = Path(sys.argv[1]).resolve()
    constraints = Path(sys.argv[2]).resolve()
    manifest = source / "pyproject.toml"
    if not manifest.is_file():
        print(f"LeRobot checkout not found: {source}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="polima-lerobot-") as temporary:
        staged = Path(temporary) / "lerobot"
        shutil.copytree(
            source,
            staged,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache"),
        )
        path = staged / "pyproject.toml"
        text = path.read_text(encoding="utf-8")
        for old, new in REPLACEMENTS.items():
            if old not in text:
                raise RuntimeError(f"LeRobot metadata changed; missing {old!r}")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
        return subprocess.call([
            sys.executable, "-m", "pip", "install",
            "-c", str(constraints), f"{staged}[smolvla]",
        ])


if __name__ == "__main__":
    raise SystemExit(main())
