"""Numerical smoke tests.

The thresholds are not invented: `cosine >= 0.999` and `mean_abs <= 0.01` are
what ACT/scripts/compile_deploy_act_som.sh asserts in a bash heredoc today.
Promoting them to a function makes them reusable, per-policy, and reportable.

Two levels:

  * `end_to_end` -- the whole pipeline against expected_normalized_actions.f32.
  * `per_stage`  -- each graph against its own recorded golden. The ACT build
    ships `<graph>_input.f32` / `<graph>_output.f32` for every stage, so when the
    end-to-end check fails this says WHICH ELF is wrong instead of just that
    something is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from polima.util.logging import get

log = get("deploy.smoke")

#: From ACT/scripts/compile_deploy_act_som.sh.
DEFAULT_COSINE_MIN = 0.999
DEFAULT_MEAN_ABS_MAX = 0.01


@dataclass
class SmokeResult:
    name: str
    ok: bool
    cosine: float
    mean_abs: float
    max_abs: float
    elements: int
    cosine_min: float = DEFAULT_COSINE_MIN
    mean_abs_max: float = DEFAULT_MEAN_ABS_MAX
    note: str = ""

    def summary(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        return (
            f"{verdict}  {self.name}: cosine={self.cosine:.6f} "
            f"(>= {self.cosine_min}) mean_abs={self.mean_abs:.6f} "
            f"(<= {self.mean_abs_max}) max_abs={self.max_abs:.6f}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name, "ok": self.ok, "cosine": self.cosine,
            "mean_abs": self.mean_abs, "max_abs": self.max_abs,
            "elements": self.elements, "cosine_min": self.cosine_min,
            "mean_abs_max": self.mean_abs_max, "note": self.note,
        }


@dataclass
class SmokeReport:
    results: list[SmokeResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    def add(self, result: SmokeResult) -> SmokeResult:
        self.results.append(result)
        return result

    def to_dict(self) -> dict:
        return {"ok": self.ok, "results": [r.to_dict() for r in self.results]}


def compare(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    cosine_min: float = DEFAULT_COSINE_MIN,
    mean_abs_max: float = DEFAULT_MEAN_ABS_MAX,
) -> SmokeResult:
    """The assertion compile_deploy_act_som.sh makes, as a function."""
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    expected = np.asarray(expected, dtype=np.float32).reshape(-1)

    if actual.size != expected.size:
        return SmokeResult(
            name=name, ok=False, cosine=0.0, mean_abs=float("inf"), max_abs=float("inf"),
            elements=actual.size, cosine_min=cosine_min, mean_abs_max=mean_abs_max,
            note=f"size mismatch: {actual.size} vs {expected.size}",
        )

    denominator = float(np.linalg.norm(actual) * np.linalg.norm(expected))
    cosine = float(actual @ expected / denominator) if denominator else 0.0
    difference = np.abs(actual - expected)
    mean_abs = float(difference.mean())
    max_abs = float(difference.max())

    return SmokeResult(
        name=name,
        ok=cosine >= cosine_min and mean_abs <= mean_abs_max,
        cosine=cosine, mean_abs=mean_abs, max_abs=max_abs, elements=actual.size,
        cosine_min=cosine_min, mean_abs_max=mean_abs_max,
    )


def numerical_smoke(
    actual: np.ndarray,
    expected_path: str | Path,
    *,
    name: str = "end-to-end",
    cosine_min: float = DEFAULT_COSINE_MIN,
    mean_abs_max: float = DEFAULT_MEAN_ABS_MAX,
) -> SmokeResult:
    expected = np.fromfile(Path(expected_path), dtype="<f4")
    return compare(
        name, actual, expected, cosine_min=cosine_min, mean_abs_max=mean_abs_max
    )


def compare_against_reference(
    actual: np.ndarray, reference: np.ndarray, *, name: str = "vs-legacy"
) -> SmokeResult:
    """A/B two servers on the same fixture.

    Held to a far tighter bar than the torch comparison: both sides run the same
    ELFs on the same MLA, so anything but a near-exact match means the host-side
    glue differs.
    """
    return compare(name, actual, reference, cosine_min=0.99999, mean_abs_max=1e-5)
