"""Generic Benders and column-generation controllers with explicit contracts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class BendersCut:
    coefficients: torch.Tensor
    rhs: float
    kind: str = "optimality"


@dataclass(frozen=True, slots=True)
class BendersResult:
    first_stage: torch.Tensor
    objective: float
    lower_bound: float
    iterations: int
    cuts: tuple[BendersCut, ...]
    converged: bool


def benders_decompose(
    master_solver: Callable[[tuple[BendersCut, ...]], tuple[torch.Tensor, float]],
    subproblem_solver: Callable[[torch.Tensor], tuple[float, BendersCut | None]],
    *,
    maximum_iterations: int = 100,
    tolerance: float = 1e-6,
) -> BendersResult:
    """Coordinate a master and scenario/recourse oracle until the gap closes."""
    if maximum_iterations < 1 or tolerance <= 0:
        raise ValueError("maximum_iterations and tolerance must be positive.")
    cuts: list[BendersCut] = []
    incumbent = torch.empty(0)
    upper = float("inf")
    lower = -float("inf")
    for iteration in range(1, maximum_iterations + 1):
        candidate, master_bound = master_solver(tuple(cuts))
        candidate = torch.as_tensor(candidate).detach().clone()
        recourse, cut = subproblem_solver(candidate)
        lower = max(lower, float(master_bound))
        upper = min(upper, float(master_bound) + float(recourse))
        incumbent = candidate
        if cut is not None:
            cuts.append(cut)
        if upper - lower <= tolerance * max(1.0, abs(upper), abs(lower)):
            return BendersResult(incumbent, upper, lower, iteration, tuple(cuts), True)
        if cut is None:
            break
    return BendersResult(incumbent, upper, lower, iteration, tuple(cuts), False)


@dataclass(frozen=True, slots=True)
class ColumnGenerationResult:
    columns: tuple[torch.Tensor, ...]
    objective: float
    iterations: int
    converged: bool


def column_generation(
    restricted_master: Callable[[tuple[torch.Tensor, ...]], tuple[float, torch.Tensor]],
    pricing: Callable[[torch.Tensor], tuple[torch.Tensor, float]],
    initial_columns: Sequence[torch.Tensor],
    *,
    maximum_iterations: int = 100,
    reduced_cost_tolerance: float = 1e-8,
) -> ColumnGenerationResult:
    columns = [torch.as_tensor(column).detach().clone() for column in initial_columns]
    if not columns:
        raise ValueError("initial_columns must be non-empty.")
    objective = float("inf")
    for iteration in range(1, maximum_iterations + 1):
        objective, dual = restricted_master(tuple(columns))
        column, reduced_cost = pricing(torch.as_tensor(dual))
        if reduced_cost >= -reduced_cost_tolerance:
            return ColumnGenerationResult(tuple(columns), float(objective), iteration, True)
        columns.append(torch.as_tensor(column).detach().clone())
    return ColumnGenerationResult(tuple(columns), float(objective), maximum_iterations, False)


__all__ = [
    "BendersCut",
    "BendersResult",
    "ColumnGenerationResult",
    "benders_decompose",
    "column_generation",
]
