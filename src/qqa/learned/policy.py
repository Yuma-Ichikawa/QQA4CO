"""Confidence-gated learned planner with deterministic OOD fallback."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PlannerModelCard:
    name: str
    version: str
    training_snapshot_sha256: str
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.version or len(self.training_snapshot_sha256) != 64:
            raise ValueError("Model card requires a name, version, and SHA-256 snapshot digest.")
        int(self.training_snapshot_sha256, 16)


class OODGate:
    def __init__(self, training_features: torch.Tensor, *, quantile: float = 0.99) -> None:
        values = torch.as_tensor(training_features, dtype=torch.float64)
        if (
            values.ndim != 2
            or len(values) < 2
            or values.shape[1] < 1
            or not torch.isfinite(values).all()
            or not 0.5 < quantile < 1
        ):
            raise ValueError("OOD training features/quantile are invalid.")
        self.mean = values.mean(dim=0)
        centred = values - self.mean
        covariance = centred.T @ centred / max(1, len(values) - 1)
        covariance = covariance + 1e-6 * torch.eye(values.shape[1], dtype=values.dtype)
        self.precision = torch.linalg.pinv(covariance)
        distances = torch.einsum("bi,ij,bj->b", centred, self.precision, centred)
        self.threshold = float(torch.quantile(distances, quantile).item())

    def score(self, features: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(features, dtype=self.mean.dtype)
        if values.shape[-1] != len(self.mean) or not torch.isfinite(values).all():
            raise ValueError("Planner features must be finite and match the training schema.")
        centred = values - self.mean
        return torch.einsum("...i,ij,...j->...", centred, self.precision, centred)

    def in_distribution(self, features: torch.Tensor) -> bool:
        return bool(self.score(features) <= self.threshold)


@dataclass(frozen=True, slots=True)
class GatedDecision:
    action: str
    confidence: float
    used_fallback: bool
    ood_score: float


class ConfidenceGatedPlanner:
    def __init__(
        self,
        policy: Callable[[torch.Tensor], tuple[str, float]],
        fallback: Callable[[torch.Tensor], str],
        gate: OODGate,
        *,
        minimum_confidence: float = 0.7,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must lie in [0, 1].")
        self.policy = policy
        self.fallback = fallback
        self.gate = gate
        self.minimum_confidence = minimum_confidence
        self.decisions = 0
        self.fallbacks = 0

    def decide(self, features: torch.Tensor) -> GatedDecision:
        action, confidence = self.policy(features)
        score = float(self.gate.score(features).item())
        use_fallback = score > self.gate.threshold or confidence < self.minimum_confidence
        if use_fallback:
            action = self.fallback(features)
            self.fallbacks += 1
        self.decisions += 1
        return GatedDecision(str(action), float(confidence), use_fallback, score)

    @staticmethod
    def snapshot_hash(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ConfidenceGatedPlanner",
    "GatedDecision",
    "OODGate",
    "PlannerModelCard",
]
