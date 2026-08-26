"""Elastic feasibility repair for callable mixed-variable models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from qqa.mixed.augmented_lagrangian import normalised_residuals
from qqa.mixed.problem import MixedProblem


@dataclass(frozen=True, slots=True)
class RepairResult:
    solution: torch.Tensor
    objective: float
    maximum_violation: float
    total_violation: float
    feasible: bool
    iterations: int


def _merit(
    problem: MixedProblem, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residuals, equality = normalised_residuals(problem, values)
    violations = torch.where(equality, residuals.abs(), residuals.clamp_min(0.0))
    maximum = violations.amax(dim=1) if violations.shape[1] else values.new_zeros(len(values))
    total = violations.sum(dim=1)
    objective = problem.objective_values(values)
    return maximum, total, objective


def repair_mixed_solution(
    problem: MixedProblem,
    candidate: torch.Tensor,
    *,
    max_steps: int = 150,
    learning_rate: float = 0.03,
    elastic_weight: float = 1000.0,
    proximity_weight: float = 1e-3,
    objective_weight: float = 1e-6,
) -> RepairResult:
    """Fix discrete coordinates and elastically repair continuous coordinates."""
    if not isinstance(problem, MixedProblem):
        raise TypeError("problem must be a MixedProblem.")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer.")
    for name, value in (
        ("learning_rate", learning_rate),
        ("elastic_weight", elastic_weight),
        ("proximity_weight", proximity_weight),
        ("objective_weight", objective_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0.")
    if learning_rate == 0:
        raise ValueError("learning_rate must be > 0.")

    physical = candidate.detach().to(dtype=problem.dtype).reshape(-1)
    problem.space.validate(physical)
    latent = problem.space.encode(physical).clone().requires_grad_(True)
    reference = latent.detach().clone()
    real_mask = torch.tensor(
        [kind == "real" for kind in problem.space.kinds],
        device=latent.device,
        dtype=latent.dtype,
    )
    optimizer = torch.optim.Adam([latent], lr=learning_rate)

    with torch.no_grad():
        initial = problem.space.project(latent).unsqueeze(0)
        best_max, best_total, best_objective = _merit(problem, initial)
        best = initial[0].detach().clone()
    iterations = 0
    for step in range(max_steps):
        iterations = step + 1
        optimizer.zero_grad(set_to_none=True)
        values = problem.space.decode(latent).unsqueeze(0)
        residuals, equality = normalised_residuals(problem, values)
        violations = torch.where(equality, residuals.abs(), residuals.clamp_min(0.0))
        elastic = violations.sum()
        proximity = ((latent - reference) * real_mask).square().sum()
        objective = problem.objective_values(values).sum()
        loss = (
            elastic_weight * elastic + proximity_weight * proximity + objective_weight * objective
        )
        loss.backward()
        with torch.no_grad():
            if latent.grad is not None:
                latent.grad.mul_(real_mask)
        optimizer.step()
        with torch.no_grad():
            latent.clamp_(0.0, 1.0)
            latent.mul_(real_mask).add_(reference * (1.0 - real_mask))
            projected = problem.space.project(latent).unsqueeze(0)
            maximum, total, objective_value = _merit(problem, projected)
            better = maximum[0] < best_max[0] - 1e-12 or (
                abs(float(maximum[0] - best_max[0])) <= 1e-12
                and (
                    total[0] < best_total[0] - 1e-12
                    or (
                        abs(float(total[0] - best_total[0])) <= 1e-12
                        and objective_value[0] < best_objective[0]
                    )
                )
            )
            if better:
                best = projected[0].detach().clone()
                best_max = maximum.detach().clone()
                best_total = total.detach().clone()
                best_objective = objective_value.detach().clone()
            if float(best_max[0]) <= 1e-8 and step >= 10:
                break
    return RepairResult(
        solution=best,
        objective=float(best_objective[0]),
        maximum_violation=float(best_max[0]),
        total_violation=float(best_total[0]),
        feasible=float(best_max[0]) <= 1e-6,
        iterations=iterations,
    )


__all__ = ["RepairResult", "repair_mixed_solution"]
