"""High-level entry point for mixed-variable optimisation."""

from __future__ import annotations

from typing import Any

from qqa.annealing import AnnealResult, anneal
from qqa.mixed.problem import MixedProblem


def solve_mixed(problem: MixedProblem, **kwargs: Any) -> AnnealResult:
    """Solve a :class:`MixedProblem` with conservative mixed-domain defaults.

    Explicit keyword arguments always win. The wrapper exists so a first-time
    user can call ``problem.solve()`` without knowing binary-QUBO defaults.
    """
    if not isinstance(problem, MixedProblem):
        raise TypeError(f"problem must be a MixedProblem, got {type(problem).__name__}.")
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
    return anneal(problem, **defaults)
