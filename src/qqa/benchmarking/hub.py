"""Portable Benchmark Hub manifests and paired statistical comparisons."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from enum import Enum
from importlib.resources import files
from pathlib import Path
from statistics import median
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from qqa.runtime.security import validate_portable_payload


class BenchmarkTrack(str, Enum):
    QQA_PRIMAL = "qqa-primal"
    CONSTRAINED = "constrained"
    CERTIFICATION = "certification"
    BLACKBOX = "blackbox"
    GPU = "gpu"
    ROBUST = "robust"


@dataclass(frozen=True, slots=True)
class BenchmarkInstance:
    name: str
    format: str
    source: str
    sha256: str | None = None
    tags: tuple[str, ...] = ()
    reference_objective: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.format or not self.source:
            raise ValueError("Benchmark instance name, format, and source must be non-empty.")
        if self.sha256 is not None and (
            len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("Benchmark instance sha256 must be a lowercase digest.")
        if self.reference_objective is not None and not math.isfinite(self.reference_objective):
            raise ValueError("reference_objective must be finite or None.")


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    name: str
    track: BenchmarkTrack
    instances: tuple[BenchmarkInstance, ...]
    budgets: tuple[float, ...]
    seeds: tuple[int, ...]
    solvers: tuple[str, ...] = ("qqa",)
    metrics: tuple[str, ...] = (
        "feasible",
        "objective",
        "time_to_first_feasible",
        "primal_integral",
    )
    cadence: tuple[str, ...] = ("pr", "nightly", "release")
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "track", BenchmarkTrack(self.track))
        object.__setattr__(self, "instances", tuple(self.instances))
        if not self.name or not self.instances:
            raise ValueError("Benchmark manifest name and instances must be non-empty.")
        if any(not math.isfinite(item) or item <= 0 for item in self.budgets):
            raise ValueError("Benchmark budgets must be finite and positive.")
        if any(isinstance(item, bool) or item < 0 for item in self.seeds):
            raise ValueError("Benchmark seeds must be non-negative integers.")
        if not self.solvers or not self.metrics:
            raise ValueError("Benchmark solvers and metrics must be non-empty.")
        if not self.cadence or not set(self.cadence) <= {"pr", "nightly", "weekly", "release"}:
            raise ValueError("Benchmark cadence must use pr/nightly/weekly/release.")
        validate_portable_payload(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["track"] = self.track.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkManifest:
        values = dict(payload)
        values["instances"] = tuple(BenchmarkInstance(**item) for item in values["instances"])
        for name in ("budgets", "seeds", "solvers", "metrics", "cadence"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    """Load TOML or JSON without retaining the local source path."""
    source = Path(path)
    with source.open("rb") as stream:
        payload = tomllib.load(stream) if source.suffix.lower() == ".toml" else json.load(stream)
    return BenchmarkManifest.from_dict(payload)


def builtin_benchmark_manifest(name: str = "qqa-core") -> BenchmarkManifest:
    """Load a wheel-packaged manifest by public logical name."""
    if not name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in name
    ):
        raise ValueError("Manifest name must use lowercase letters, digits, '-' or '_'.")
    resource = files("qqa.benchmarking").joinpath("manifests", f"{name}.toml")
    with resource.open("rb") as stream:
        return BenchmarkManifest.from_dict(tomllib.load(stream))


@dataclass(frozen=True, slots=True)
class PairedMetricSummary:
    pairs: int
    wins: int
    ties: int
    losses: int
    median_difference: float
    confidence_interval: tuple[float, float]
    sign_test_pvalue: float


def paired_metric_summary(
    candidate: list[float] | tuple[float, ...],
    baseline: list[float] | tuple[float, ...],
    *,
    lower_is_better: bool = True,
    confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> PairedMetricSummary:
    """Compare aligned runs with a deterministic paired bootstrap interval."""
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("candidate and baseline must be aligned and non-empty.")
    if not 0 < confidence < 1 or bootstrap_samples < 100:
        raise ValueError("confidence must be in (0, 1) and bootstrap_samples >= 100.")
    differences = [
        float(left) - float(right) for left, right in zip(candidate, baseline, strict=True)
    ]
    if any(not math.isfinite(item) for item in differences):
        raise ValueError("Paired metrics must be finite.")
    tolerance = 1e-12
    wins = sum(item < -tolerance if lower_is_better else item > tolerance for item in differences)
    losses = sum(item > tolerance if lower_is_better else item < -tolerance for item in differences)
    non_ties = wins + losses
    tail_count = min(wins, losses)
    sign_test = (
        1.0
        if non_ties == 0
        else min(
            1.0,
            2.0
            * sum(math.comb(non_ties, index) for index in range(tail_count + 1))
            / (2**non_ties),
        )
    )
    rng = random.Random(seed)
    bootstrap = sorted(
        median([differences[rng.randrange(len(differences))] for _ in differences])
        for _ in range(bootstrap_samples)
    )
    tail = (1.0 - confidence) / 2.0
    lower = bootstrap[max(0, int(tail * bootstrap_samples))]
    upper = bootstrap[min(bootstrap_samples - 1, int((1.0 - tail) * bootstrap_samples))]
    return PairedMetricSummary(
        len(differences),
        wins,
        len(differences) - wins - losses,
        losses,
        float(median(differences)),
        (float(lower), float(upper)),
        sign_test,
    )


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values for multiple comparisons."""
    if not pvalues or any(not 0.0 <= value <= 1.0 for value in pvalues.values()):
        raise ValueError("pvalues must be a non-empty mapping with values in [0, 1].")
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in pvalues}


__all__ = [
    "BenchmarkInstance",
    "BenchmarkManifest",
    "BenchmarkTrack",
    "PairedMetricSummary",
    "builtin_benchmark_manifest",
    "holm_adjust",
    "load_benchmark_manifest",
    "paired_metric_summary",
]
