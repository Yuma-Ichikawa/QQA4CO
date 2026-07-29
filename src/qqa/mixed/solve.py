"""High-level entry point for mixed-variable optimisation."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch

from qqa.annealing import AnnealResult, anneal
from qqa.mixed.problem import MixedProblem
from qqa.utils import resolve_device


def _prefer_feasible_incumbent(problem: MixedProblem, result: AnnealResult) -> None:
    """Select the best feasible final replica before a penalized incumbent."""
    if not problem.constraints or result.final_population is None:
        return
    candidates = torch.cat(
        [
            result.best_sol.detach().reshape(1, -1),
            result.final_population.detach().reshape(-1, problem.num_vars),
        ],
        dim=0,
    )
    with torch.no_grad():
        violations = problem.constraint_violations(candidates)
        feasible = torch.ones(len(candidates), dtype=torch.bool, device=candidates.device)
        for constraint in problem.constraints:
            feasible &= violations[constraint.name] <= constraint.tolerance
        if not feasible.any():
            return
        indices = torch.where(feasible)[0]
        objective = problem.objective_values(candidates)
        selected = indices[torch.argmin(objective[indices])]
        best_sol = candidates[selected].detach().clone()
        best_obj = float(problem.loss_fn(best_sol.unsqueeze(0))[0].item())
    result.best_sol = best_sol
    result.best_obj = best_obj
    result.score = problem.score_summary(best_sol)


def solve_mixed(
    problem: MixedProblem,
    *,
    calibrate_penalty: bool = True,
    calibration_points: int = 256,
    penalty_safety_factor: float = 50.0,
    max_penalty_multiplier: float = 1e8,
    **kwargs: Any,
) -> AnnealResult:
    """Solve a :class:`MixedProblem` with conservative mixed-domain defaults.

    Explicit keyword arguments always win. The wrapper exists so a first-time
    user can call ``problem.solve()`` without knowing binary-QUBO defaults.
    """
    if not isinstance(problem, MixedProblem):
        raise TypeError(f"problem must be a MixedProblem, got {type(problem).__name__}.")
    if not isinstance(calibration_points, int) or calibration_points < 2:
        raise ValueError("calibration_points must be an integer >= 2.")
    if not math.isfinite(penalty_safety_factor) or penalty_safety_factor <= 0:
        raise ValueError("penalty_safety_factor must be finite and > 0.")
    if not math.isfinite(max_penalty_multiplier) or max_penalty_multiplier < 1:
        raise ValueError("max_penalty_multiplier must be finite and >= 1.")
    defaults = {
        "sol_size": 128,
        "learning_rate": 0.05,
        "min_bg": -0.5,
        "max_bg": 1.0,
        "curve_rate": 2,
        "num_epochs": 1500,
        "polish": False,
        "gradient_clip_norm": 100.0,
        "restart_patience": 250,
        "restart_fraction": 0.15,
    }
    defaults.update(kwargs)
    defaults["device"] = resolve_device(defaults.get("device", "cpu"))
    return_population = bool(defaults.get("return_population", False))
    defaults["return_population"] = True

    # Calibration belongs to one solve, not to the reusable problem object.
    # A shallow copy preserves immutable variable metadata and callables while
    # making concurrent/repeated solves independent.
    solving_problem = copy.copy(problem)
    solving_problem.penalty_multiplier = 1.0
    if calibrate_penalty and problem.constraints:
        device = defaults.get("device", "cpu")
        engine = torch.quasirandom.SobolEngine(
            problem.num_vars,
            scramble=True,
            seed=0,
        )
        latent = engine.draw(calibration_points).to(device=device, dtype=problem.dtype)
        projected = problem.relaxation.project(latent)
        with torch.no_grad():
            objective = problem.objective_values(projected)
            penalty = problem.constraint_penalty(projected)
            positive = penalty[penalty > 1e-12]
            # Use a robust central range rather than the absolute objective
            # level (which breaks under a harmless constant offset) or MAD
            # (which can materially understate the trade-off span).  The
            # 10–90 % Sobol range ignores isolated nonlinear outliers while
            # representing the cost that feasibility must dominate.
            quantiles = torch.quantile(
                objective,
                torch.tensor([0.1, 0.9], device=device, dtype=objective.dtype),
            )
            objective_scale = float((quantiles[1] - quantiles[0]).abs().clamp_min(1.0).item())
            penalty_scale = float(positive.median().item()) if positive.numel() else 1.0
        solving_problem.penalty_multiplier = min(
            max_penalty_multiplier,
            max(1.0, penalty_safety_factor * objective_scale / penalty_scale),
        )
    result = anneal(solving_problem, **defaults)
    _prefer_feasible_incumbent(solving_problem, result)
    result.diagnostics["penalty_multiplier"] = float(solving_problem.penalty_multiplier)
    if not return_population:
        result.final_population = None
    return result
