"""Strict, shared configuration for the stable solve API."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal

from qqa.schedule import make_schedule

SolverProfile = Literal[
    "fast",
    "balanced",
    "quality",
    "certify",
    "diverse",
    "pareto",
    "reproducible",
]


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Single source of truth for Python, CLI, UI, and benchmark defaults.

    Unknown keys are rejected by :meth:`from_mapping`; integrations must not
    silently drop misspelled or unsupported options.
    """

    profile: SolverProfile = "balanced"
    backend: Literal["qqa", "sa", "pa", "isco"] = "qqa"
    budget: float | None = None
    device: str = "auto"
    seed: int = 0
    replicas: int | None = None
    epochs: int | None = None
    learning_rate: float | None = None
    temperature: float = 0.0
    schedule: str = "cosine"
    min_bg: float = -2.0
    max_bg: float = 0.1
    curve_rate: int = 2
    diversity: float | None = None
    polish: bool = True
    restart_patience: int | None = None
    restart_fraction: float = 0.15
    restart_jitter: float = 0.10
    gradient_clip_norm: float | None = 100.0
    optimizer: Literal["adamw", "lightweight-adamw", "mirror-descent"] = "adamw"
    mixed_precision: Literal["fp32", "bf16"] = "fp32"
    memory_fraction: float = 0.80
    return_population: bool = False
    exact_backend: Literal["auto", "none", "scip", "highs", "cpsat", "cuopt"] = "auto"
    require_certificate: bool = False
    deterministic: bool = False
    compile_core: bool = False
    sparse_kernel: Literal["auto", "torch", "triton"] = "auto"
    cuda_graphs: bool = False
    normalize_loss: bool = True
    robust_scaling: bool = True
    heterogeneous_replicas: bool = True
    replica_exchange_interval: int | None = 100
    factor_preconditioning: bool = True
    curvature_aware_beta: bool = True
    archive_size: int = 64

    def __post_init__(self) -> None:
        if self.profile not in _PROFILE_DEFAULTS:
            raise ValueError(
                f"Unknown profile {self.profile!r}; choose from {sorted(_PROFILE_DEFAULTS)}."
            )
        if self.backend not in {"qqa", "sa", "pa", "isco"}:
            raise ValueError("backend must be qqa, sa, pa, or isco.")
        if self.budget is not None and (
            isinstance(self.budget, bool) or not math.isfinite(self.budget) or self.budget <= 0
        ):
            raise ValueError("budget must be finite and > 0, or None.")
        for name, value, minimum in (
            ("seed", self.seed, 0),
            ("replicas", self.replicas, 1),
            ("epochs", self.epochs, 0),
            ("restart_patience", self.restart_patience, 0),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}, or None.")
        if self.learning_rate is not None and (
            isinstance(self.learning_rate, bool)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and > 0, or None.")
        for name, scalar_value in (
            ("temperature", self.temperature),
            ("restart_jitter", self.restart_jitter),
        ):
            if (
                isinstance(scalar_value, bool)
                or not math.isfinite(scalar_value)
                or scalar_value < 0
            ):
                raise ValueError(f"{name} must be finite and >= 0.")
        if not math.isfinite(self.min_bg) or not math.isfinite(self.max_bg):
            raise ValueError("min_bg and max_bg must be finite.")
        if isinstance(self.curve_rate, bool) or self.curve_rate < 2 or self.curve_rate % 2:
            raise ValueError("curve_rate must be a positive even integer.")
        if self.diversity is not None and (
            not math.isfinite(self.diversity) or not 0 <= self.diversity <= 1
        ):
            raise ValueError("diversity must be in [0, 1], or None.")
        if not math.isfinite(self.restart_fraction) or not 0 < self.restart_fraction < 1:
            raise ValueError("restart_fraction must be in (0, 1).")
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and > 0, or None.")
        if not math.isfinite(self.memory_fraction) or not 0 < self.memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0, 1].")
        if self.mixed_precision not in {"fp32", "bf16"}:
            raise ValueError("mixed_precision must be 'fp32' or 'bf16'.")
        if self.optimizer not in {"adamw", "lightweight-adamw", "mirror-descent"}:
            raise ValueError("optimizer must be 'adamw', 'lightweight-adamw', or 'mirror-descent'.")
        if self.exact_backend not in {"auto", "none", "scip", "highs", "cpsat", "cuopt"}:
            raise ValueError("Unsupported exact_backend.")
        if self.sparse_kernel not in {"auto", "torch", "triton"}:
            raise ValueError("sparse_kernel must be auto, torch, or triton.")
        for name in ("deterministic", "compile_core", "cuda_graphs"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        if self.cuda_graphs and self.heterogeneous_replicas and self.replica_exchange_interval:
            raise ValueError(
                "cuda_graphs cannot be combined with heterogeneous replica exchange; "
                "set replica_exchange_interval=None."
            )
        for name in (
            "normalize_loss",
            "robust_scaling",
            "heterogeneous_replicas",
            "factor_preconditioning",
            "curvature_aware_beta",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        if self.replica_exchange_interval is not None and (
            isinstance(self.replica_exchange_interval, bool)
            or not isinstance(self.replica_exchange_interval, int)
            or self.replica_exchange_interval < 1
        ):
            raise ValueError("replica_exchange_interval must be a positive integer or None.")
        if (
            isinstance(self.archive_size, bool)
            or not isinstance(self.archive_size, int)
            or self.archive_size < 0
        ):
            raise ValueError("archive_size must be a non-negative integer.")
        if self.backend != "qqa" and (
            self.require_certificate or self.exact_backend not in {"auto", "none"}
        ):
            raise ValueError(
                "Exact certification requires backend='qqa'; SA, PA, and iSCO "
                "are standalone baseline backends."
            )
        make_schedule(self.schedule, minimum=self.min_bg, maximum=self.max_bg)

    @classmethod
    def for_profile(cls, profile: SolverProfile = "balanced", **overrides: Any) -> SolverConfig:
        """Create a profile and apply explicit, validated overrides."""
        if profile not in _PROFILE_DEFAULTS:
            raise ValueError(
                f"Unknown profile {profile!r}; choose from {sorted(_PROFILE_DEFAULTS)}."
            )
        return cls(profile=profile, **{**_PROFILE_DEFAULTS[profile], **overrides})

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> SolverConfig:
        """Build from a mapping and reject every unknown key."""
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise TypeError(f"Unknown SolverConfig option(s): {', '.join(unknown)}")
        profile = values.get("profile", "balanced")
        overrides = {key: value for key, value in values.items() if key != "profile"}
        return cls.for_profile(profile, **overrides)

    def resolved(self) -> SolverConfig:
        """Fill profile-dependent optional fields without changing explicit values."""
        defaults = _PROFILE_DEFAULTS[self.profile]
        updates = {
            name: defaults[name]
            for name in ("replicas", "epochs", "learning_rate", "diversity", "restart_patience")
            if getattr(self, name) is None
        }
        return replace(self, **updates)

    def anneal_kwargs(self) -> dict[str, Any]:
        """Translate to the legacy QQA engine at the explicit adapter boundary."""
        resolved = self.resolved()
        return {
            "sol_size": resolved.replicas,
            "num_epochs": resolved.epochs,
            "learning_rate": resolved.learning_rate,
            "temp": resolved.temperature,
            "schedule": make_schedule(
                resolved.schedule,
                minimum=resolved.min_bg,
                maximum=resolved.max_bg,
            ),
            "curve_rate": resolved.curve_rate,
            "div_param": resolved.diversity,
            "time_limit": resolved.budget,
            "device": resolved.device,
            "polish": resolved.polish,
            "return_population": resolved.return_population,
            "restart_patience": resolved.restart_patience or None,
            "restart_fraction": resolved.restart_fraction,
            "restart_jitter": resolved.restart_jitter,
            "gradient_clip_norm": resolved.gradient_clip_norm,
            "optimizer": resolved.optimizer,
            "mixed_precision": resolved.mixed_precision,
            "cuda_graphs": resolved.cuda_graphs,
            "normalize_loss": resolved.normalize_loss,
            "robust_scaling": resolved.robust_scaling,
            "heterogeneous_replicas": resolved.heterogeneous_replicas,
            "replica_exchange_interval": resolved.replica_exchange_interval,
            "factor_preconditioning": resolved.factor_preconditioning,
            "curvature_aware_beta": resolved.curvature_aware_beta,
            "archive_size": resolved.archive_size,
            "verbose": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "fast": {
        "replicas": 32,
        "epochs": 300,
        "learning_rate": 0.10,
        "diversity": 0.0,
        "restart_patience": None,
    },
    "balanced": {
        "replicas": 128,
        "epochs": 1500,
        "learning_rate": 0.05,
        "diversity": 0.01,
        "restart_patience": 250,
    },
    "quality": {
        "replicas": 512,
        "epochs": 5000,
        "learning_rate": 0.03,
        "diversity": 0.03,
        "restart_patience": 400,
    },
    "certify": {
        "replicas": 256,
        "epochs": 2500,
        "learning_rate": 0.04,
        "diversity": 0.01,
        "restart_patience": 300,
        "require_certificate": True,
    },
    "diverse": {
        "replicas": 512,
        "epochs": 2500,
        "learning_rate": 0.04,
        "diversity": 0.15,
        "restart_patience": 250,
        "return_population": True,
    },
    "pareto": {
        "replicas": 512,
        "epochs": 3000,
        "learning_rate": 0.04,
        "diversity": 0.05,
        "restart_patience": 250,
        "return_population": True,
    },
    "reproducible": {
        "replicas": 128,
        "epochs": 1500,
        "learning_rate": 0.05,
        "diversity": 0.0,
        "restart_patience": None,
        "deterministic": True,
    },
}


__all__ = ["SolverConfig", "SolverProfile"]
