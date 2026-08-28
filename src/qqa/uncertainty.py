"""Scenario, robust, chance-constrained, and CVaR factor aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import torch

from qqa.model.ir import Factor


def _valid_probability_vector(probabilities: torch.Tensor, size: int, *, positive: bool) -> bool:
    return bool(
        len(probabilities) == size
        and torch.isfinite(probabilities).all()
        and (torch.all(probabilities > 0) if positive else torch.all(probabilities >= 0))
        and torch.isclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    )


@dataclass(frozen=True, slots=True)
class ScenarioFactor:
    """Aggregate matching factors evaluated under multiple scenarios."""

    scenarios: tuple[Factor, ...]
    mode: Literal["mean", "worst", "cvar"] = "mean"
    probabilities: torch.Tensor | None = None
    cvar_alpha: float = 0.95

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenarios)
        if not scenarios:
            raise ValueError("ScenarioFactor requires at least one scenario.")
        if self.mode not in {"mean", "worst", "cvar"}:
            raise ValueError("mode must be mean, worst, or cvar.")
        if not 0 <= self.cvar_alpha < 1:
            raise ValueError("cvar_alpha must be in [0, 1).")
        probabilities = (
            torch.full((len(scenarios),), 1.0 / len(scenarios), dtype=torch.float64)
            if self.probabilities is None
            else torch.as_tensor(self.probabilities, dtype=torch.float64).reshape(-1)
        )
        if not _valid_probability_vector(probabilities, len(scenarios), positive=False):
            raise ValueError("Scenario probabilities must be non-negative and sum to one.")
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "probabilities", probabilities)

    def evaluate_scenarios(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack([factor.evaluate(values) for factor in self.scenarios], dim=-1)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        outcomes = self.evaluate_scenarios(values)
        probabilities = self.probabilities
        if probabilities is None:  # Defensive guard for static type checkers and unsafe mutation.
            raise RuntimeError("Scenario probabilities were not initialised.")
        probabilities = probabilities.to(values)
        if self.mode == "mean":
            return (outcomes * probabilities).sum(dim=-1)
        if self.mode == "worst":
            return outcomes.amax(dim=-1)
        order = torch.argsort(outcomes, dim=-1)
        sorted_outcomes = torch.gather(outcomes, -1, order)
        expanded = probabilities.expand_as(outcomes)
        sorted_probabilities = torch.gather(expanded, -1, order)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        tail = cumulative > self.cvar_alpha
        tail_weights = torch.where(tail, sorted_probabilities, 0.0)
        # Include the boundary scenario fraction so CVaR remains continuous in
        # probability mass rather than dropping the quantile bucket.
        previous = cumulative - sorted_probabilities
        boundary = (previous <= self.cvar_alpha) & tail
        tail_weights = torch.where(
            boundary,
            cumulative - self.cvar_alpha,
            tail_weights,
        )
        return (sorted_outcomes * tail_weights).sum(dim=-1) / max(1e-12, 1 - self.cvar_alpha)


@dataclass(frozen=True, slots=True)
class ChanceConstraintFactor:
    """Penalty when the empirical probability of violation exceeds a limit."""

    scenarios: tuple[Factor, ...]
    allowed_probability: float = 0.05
    temperature: float = 0.05
    weight: float = 1.0
    probabilities: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            not self.scenarios
            or not math.isfinite(self.allowed_probability)
            or not 0 <= self.allowed_probability <= 1
            or not math.isfinite(self.weight)
            or self.weight < 0
        ):
            raise ValueError("Chance constraints require scenarios and probability in [0, 1].")
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive.")
        probabilities = (
            torch.full((len(self.scenarios),), 1.0 / len(self.scenarios), dtype=torch.float64)
            if self.probabilities is None
            else torch.as_tensor(self.probabilities, dtype=torch.float64).reshape(-1)
        )
        if not _valid_probability_vector(probabilities, len(self.scenarios), positive=False):
            raise ValueError("Chance-constraint probabilities must be non-negative and sum to one.")
        object.__setattr__(self, "probabilities", probabilities)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        residuals = torch.stack([factor.evaluate(values) for factor in self.scenarios], dim=-1)
        probabilities = self.probabilities
        assert probabilities is not None
        probability = (torch.sigmoid(residuals / self.temperature) * probabilities.to(values)).sum(
            dim=-1
        )
        return self.weight * (probability - self.allowed_probability).clamp_min(0.0).square()


@dataclass(frozen=True, slots=True)
class DistributionallyRobustChanceFactor:
    """Total-variation ambiguity upper bound for a smoothed chance row."""

    scenarios: tuple[Factor, ...]
    allowed_probability: float = 0.05
    ambiguity_radius: float = 0.0
    temperature: float = 0.05
    weight: float = 1.0
    probabilities: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            not self.scenarios
            or not all(
                math.isfinite(value)
                for value in (
                    self.allowed_probability,
                    self.ambiguity_radius,
                    self.temperature,
                    self.weight,
                )
            )
            or not 0 <= self.allowed_probability <= 1
            or not 0 <= self.ambiguity_radius <= 1
            or self.temperature <= 0
            or self.weight < 0
        ):
            raise ValueError("Invalid distributionally robust chance constraint.")
        probabilities = (
            torch.full((len(self.scenarios),), 1 / len(self.scenarios), dtype=torch.float64)
            if self.probabilities is None
            else torch.as_tensor(self.probabilities, dtype=torch.float64).reshape(-1)
        )
        if not _valid_probability_vector(probabilities, len(self.scenarios), positive=False):
            raise ValueError("Robust chance probabilities must be non-negative and sum to one.")
        object.__setattr__(self, "probabilities", probabilities)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        residuals = torch.stack([factor.evaluate(values) for factor in self.scenarios], dim=-1)
        assert self.probabilities is not None
        nominal = (torch.sigmoid(residuals / self.temperature) * self.probabilities.to(values)).sum(
            dim=-1
        )
        worst_probability = (nominal + self.ambiguity_radius).clamp_max(1.0)
        return self.weight * (worst_probability - self.allowed_probability).clamp_min(0).square()


@dataclass(frozen=True, slots=True)
class WassersteinDROFactor:
    """Lipschitz-certified Wasserstein worst-case expectation."""

    scenarios: tuple[Factor, ...]
    radius: float
    lipschitz_constant: float
    probabilities: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("Wasserstein DRO requires at least one scenario.")
        if (
            not math.isfinite(self.radius)
            or not math.isfinite(self.lipschitz_constant)
            or self.radius < 0
            or self.lipschitz_constant < 0
        ):
            raise ValueError("Wasserstein radius and Lipschitz constant must be non-negative.")
        probabilities = (
            torch.full((len(self.scenarios),), 1 / len(self.scenarios), dtype=torch.float64)
            if self.probabilities is None
            else torch.as_tensor(self.probabilities, dtype=torch.float64).reshape(-1)
        )
        if not _valid_probability_vector(probabilities, len(self.scenarios), positive=False):
            raise ValueError("Scenario probabilities must be non-negative and sum to one.")
        object.__setattr__(self, "probabilities", probabilities)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        outcomes = torch.stack([factor.evaluate(values) for factor in self.scenarios], dim=-1)
        assert self.probabilities is not None
        nominal = (outcomes * self.probabilities.to(values)).sum(dim=-1)
        return nominal + self.radius * self.lipschitz_constant


@dataclass(frozen=True, slots=True)
class PhiDivergenceDROFactor:
    """KL or chi-square ambiguity-set robust expectation."""

    scenarios: tuple[Factor, ...]
    radius: float
    kind: Literal["kl", "chi2"] = "kl"
    temperature: float = 0.1
    probabilities: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            not self.scenarios
            or not math.isfinite(self.radius)
            or self.radius < 0
            or self.kind not in {"kl", "chi2"}
        ):
            raise ValueError("Invalid phi-divergence ambiguity set.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        probabilities = (
            torch.full((len(self.scenarios),), 1 / len(self.scenarios), dtype=torch.float64)
            if self.probabilities is None
            else torch.as_tensor(self.probabilities, dtype=torch.float64).reshape(-1)
        )
        if not _valid_probability_vector(probabilities, len(self.scenarios), positive=True):
            raise ValueError("Phi-divergence probabilities must be positive and sum to one.")
        object.__setattr__(self, "probabilities", probabilities)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        outcomes = torch.stack([factor.evaluate(values) for factor in self.scenarios], dim=-1)
        assert self.probabilities is not None
        probabilities = self.probabilities.to(values)
        if self.kind == "chi2":
            mean = (outcomes * probabilities).sum(dim=-1)
            variance = ((outcomes - mean.unsqueeze(-1)).square() * probabilities).sum(dim=-1)
            return mean + torch.sqrt((self.radius * variance).clamp_min(0))
        eta = outcomes.new_tensor(self.temperature)
        return eta * self.radius + eta * torch.logsumexp(
            probabilities.log() + outcomes / eta, dim=-1
        )


@dataclass(frozen=True, slots=True)
class MomentAmbiguityDROFactor:
    """One-sided moment-ambiguity bound using a mean/std safety factor."""

    scenarios: tuple[Factor, ...]
    confidence: float = 0.95
    probabilities: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not self.scenarios or not math.isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("Moment ambiguity requires scenarios and confidence in (0, 1).")
        probabilities = (
            torch.full((len(self.scenarios),), 1 / len(self.scenarios), dtype=torch.float64)
            if self.probabilities is None
            else torch.as_tensor(self.probabilities, dtype=torch.float64).reshape(-1)
        )
        if not _valid_probability_vector(probabilities, len(self.scenarios), positive=False):
            raise ValueError("Moment-ambiguity probabilities must be non-negative and sum to one.")
        object.__setattr__(self, "probabilities", probabilities)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        outcomes = torch.stack([factor.evaluate(values) for factor in self.scenarios], dim=-1)
        assert self.probabilities is not None
        probability = self.probabilities.to(values)
        mean = (outcomes * probability).sum(dim=-1)
        variance = ((outcomes - mean.unsqueeze(-1)).square() * probability).sum(dim=-1)
        factor = math.sqrt(self.confidence / (1.0 - self.confidence))
        return mean + factor * variance.clamp_min(0).sqrt()


def sample_average_confidence_interval(
    outcomes: torch.Tensor,
    *,
    confidence: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normal-approximation confidence interval along the scenario axis."""
    values = torch.as_tensor(outcomes)
    if values.ndim < 1 or values.shape[-1] < 2 or not 0 < confidence < 1:
        raise ValueError("At least two scenarios and confidence in (0, 1) are required.")
    if not torch.isfinite(values).all():
        raise ValueError("Scenario outcomes must be finite.")
    mean = values.mean(dim=-1)
    standard_error = values.std(dim=-1, correction=1) / math.sqrt(values.shape[-1])
    quantile = NormalDist().inv_cdf(0.5 + confidence / 2)
    return mean - quantile * standard_error, mean + quantile * standard_error


def validate_out_of_sample(
    factor: Factor,
    solutions: torch.Tensor,
    *,
    tolerance: float = 0.0,
) -> dict[str, float]:
    """Evaluate held-out scenario cost/violation without modifying a solution."""
    if not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite.")
    outcomes = factor.evaluate(torch.as_tensor(solutions)).detach().to(torch.float64)
    if outcomes.ndim != 1 or not len(outcomes) or not torch.isfinite(outcomes).all():
        raise ValueError("Held-out factor must return one finite value per solution.")
    lower, upper = sample_average_confidence_interval(outcomes)
    return {
        "mean": float(outcomes.mean().item()),
        "maximum": float(outcomes.max().item()),
        "violation_rate": float((outcomes > tolerance).to(torch.float64).mean().item()),
        "confidence_lower": float(lower.item()),
        "confidence_upper": float(upper.item()),
    }


def reduce_scenarios(
    features: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy probability-weighted k-medoids scenario reduction."""
    features = torch.as_tensor(features)
    probabilities = torch.as_tensor(
        probabilities, device=features.device, dtype=features.dtype
    ).reshape(-1)
    if (
        features.ndim != 2
        or len(features) != len(probabilities)
        or isinstance(count, bool)
        or not 1 <= count <= len(features)
    ):
        raise ValueError("features/probabilities must align and count must be valid.")
    if (
        not torch.isfinite(features).all()
        or not torch.isfinite(probabilities).all()
        or torch.any(probabilities < 0)
        or not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.0))
    ):
        raise ValueError("probabilities must be non-negative and sum to one.")
    distances = torch.cdist(features, features)
    selected = [int(torch.argmax(probabilities).item())]
    while len(selected) < count:
        nearest = distances[:, selected].amin(dim=1)
        score = probabilities * nearest
        score[selected] = -1
        selected.append(int(torch.argmax(score).item()))
    medoids = torch.as_tensor(selected, device=features.device)
    assignment = distances[:, medoids].argmin(dim=1)
    reduced_probability = probabilities.new_zeros(count).scatter_add_(0, assignment, probabilities)
    return features[medoids], reduced_probability


__all__ = [
    "ChanceConstraintFactor",
    "DistributionallyRobustChanceFactor",
    "MomentAmbiguityDROFactor",
    "PhiDivergenceDROFactor",
    "ScenarioFactor",
    "WassersteinDROFactor",
    "reduce_scenarios",
    "sample_average_confidence_interval",
    "validate_out_of_sample",
]
