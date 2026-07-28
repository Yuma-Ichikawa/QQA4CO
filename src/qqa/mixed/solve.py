"""High-level entry point for mixed-variable optimisation."""

from __future__ import annotations

import math
from typing import Any

import torch

from qqa.annealing import AnnealResult, anneal
from qqa.mixed.problem import MixedProblem


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
    }
    defaults.update(kwargs)
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
            objective = problem.objective_values(projected).abs()
            penalty = problem.constraint_penalty(projected)
            positive = penalty[penalty > 1e-12]
            objective_scale = float(objective.median().clamp_min(1.0).item())
            penalty_scale = float(positive.median().item()) if positive.numel() else 1.0
        problem.penalty_multiplier = min(
            max_penalty_multiplier,
            max(1.0, penalty_safety_factor * objective_scale / penalty_scale),
        )
    return anneal(problem, **defaults)
