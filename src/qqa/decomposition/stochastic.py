"""Progressive-hedging controller for scenario-decomposable decisions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ProgressiveHedgingResult:
    consensus: torch.Tensor
    scenario_solutions: torch.Tensor
    residual: float
    iterations: int
    converged: bool


def progressive_hedging(
    scenario_solvers: Sequence[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]],
    initial_consensus: torch.Tensor,
    *,
    probabilities: torch.Tensor | None = None,
    rho: float = 1.0,
    tolerance: float = 1e-5,
    maximum_iterations: int = 100,
) -> ProgressiveHedgingResult:
    """Coordinate independent scenario oracles through nonanticipativity."""
    solvers = tuple(scenario_solvers)
    consensus = torch.as_tensor(initial_consensus, dtype=torch.float64).detach().clone()
    if not solvers or consensus.ndim != 1 or not torch.isfinite(consensus).all():
        raise ValueError("Scenario solvers and a finite vector consensus are required.")
    if rho <= 0 or tolerance <= 0 or maximum_iterations < 1:
        raise ValueError("rho, tolerance, and maximum_iterations must be positive.")
    probability = (
        torch.full((len(solvers),), 1 / len(solvers), dtype=consensus.dtype)
        if probabilities is None
        else torch.as_tensor(probabilities, dtype=consensus.dtype).reshape(-1)
    )
    if (
        len(probability) != len(solvers)
        or torch.any(probability < 0)
        or not torch.isclose(probability.sum(), probability.new_tensor(1.0))
    ):
        raise ValueError("Scenario probabilities must be non-negative and sum to one.")
    multipliers = torch.zeros((len(solvers), len(consensus)), dtype=consensus.dtype)
    scenario_values = consensus.expand(len(solvers), -1).clone()
    residual = float("inf")
    for iteration in range(1, maximum_iterations + 1):
        scenario_values = torch.stack(
            [
                torch.as_tensor(solver(consensus, multipliers[index], rho), dtype=consensus.dtype)
                for index, solver in enumerate(solvers)
            ]
        )
        if scenario_values.shape != multipliers.shape or not torch.isfinite(scenario_values).all():
            raise ValueError("Every scenario oracle must return one aligned finite vector.")
        consensus = (scenario_values * probability[:, None]).sum(dim=0)
        difference = scenario_values - consensus
        multipliers = multipliers + rho * difference
        residual = float(torch.linalg.vector_norm(difference, dim=1).max().item())
        if residual <= tolerance:
            return ProgressiveHedgingResult(consensus, scenario_values, residual, iteration, True)
    return ProgressiveHedgingResult(
        consensus,
        scenario_values,
        residual,
        maximum_iterations,
        False,
    )


__all__ = ["ProgressiveHedgingResult", "progressive_hedging"]
