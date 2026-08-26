"""Scenario, robust, chance-constrained, and CVaR factor aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from qqa.model.ir import Factor


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
        if (
            len(probabilities) != len(scenarios)
            or torch.any(probabilities < 0)
            or not torch.isclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
        ):
            raise ValueError("Scenario probabilities must be non-negative and sum to one.")
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "probabilities", probabilities)

    def evaluate_scenarios(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack([factor.evaluate(values) for factor in self.scenarios], dim=-1)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        outcomes = self.evaluate_scenarios(values)
        probabilities = self.probabilities.to(values)
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

    def __post_init__(self) -> None:
        if not self.scenarios or not 0 <= self.allowed_probability <= 1:
            raise ValueError("Chance constraints require scenarios and probability in [0, 1].")
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive.")

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        residuals = torch.stack([factor.evaluate(values) for factor in self.scenarios], dim=-1)
        probability = torch.sigmoid(residuals / self.temperature).mean(dim=-1)
        return self.weight * (probability - self.allowed_probability).clamp_min(0.0).square()


__all__ = ["ChanceConstraintFactor", "ScenarioFactor"]
