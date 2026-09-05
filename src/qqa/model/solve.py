"""Constraint-aware execution of canonical :class:`~qqa.model.ModelIR`."""

from __future__ import annotations

from typing import Any

import torch

from qqa.annealing import AnnealResult, anneal
from qqa.mixed.augmented_lagrangian import (
    AdaptiveALCallback,
    AdaptiveAugmentedLagrangian,
    ConstraintArchive,
    ConstraintArchiveCallback,
)
from qqa.model.problem import ModelIRProblem


def _prefer_feasible(problem: ModelIRProblem, result: AnnealResult) -> None:
    if not problem.constraints or result.final_population is None:
        return
    candidates = torch.cat([result.best_sol.unsqueeze(0), result.final_population], dim=0)
    with torch.no_grad():
        keys = problem.incumbent_keys(candidates)
        order = torch.arange(len(candidates), device=candidates.device)
        for column in range(keys.shape[1] - 1, -1, -1):
            order = order[torch.argsort(keys[order, column], stable=True)]
        selected = order[0]
        result.best_sol = candidates[selected].detach().clone()
        result.best_obj = float(problem.internal_objective(result.best_sol.unsqueeze(0))[0])
        result.score = problem.score_summary(result.best_sol)


def solve_model_ir(problem: ModelIRProblem, **kwargs: Any) -> AnnealResult:
    """Run QQA with scaled, constraint-wise augmented Lagrangian state.

    The state is created per solve, so reusing one immutable ModelIR from
    multiple callers never leaks multipliers between campaigns.
    """
    if not isinstance(problem, ModelIRProblem):
        raise TypeError("problem must be a ModelIRProblem.")
    options = dict(kwargs)
    requested_population = bool(options.get("return_population", False))
    callbacks = list(options.pop("callbacks", ()))
    archive = ConstraintArchive() if problem.constraints else None
    controller = None
    if archive is not None:
        controller = AdaptiveAugmentedLagrangian.for_problem(problem)
        problem._augmented_lagrangian = controller
        update_interval = max(1, min(50, int(options.get("num_epochs", 1000) or 1)))
        callbacks.extend(
            [
                ConstraintArchiveCallback(archive, update_interval=min(10, update_interval)),
                AdaptiveALCallback(update_interval=update_interval, controller=controller),
            ]
        )
        options["return_population"] = True
    if callbacks:
        options["callbacks"] = callbacks
    try:
        result = anneal(problem, **options)
    finally:
        if controller is not None:
            problem._augmented_lagrangian = None
    if archive is not None and result.final_population is not None:
        archive.update(
            problem,
            torch.cat([result.best_sol.unsqueeze(0), result.final_population], dim=0),
        )
        extra = archive.candidates()
        if extra:
            result.final_population = torch.cat(
                [result.final_population, torch.stack(extra).to(result.final_population)], dim=0
            )
        _prefer_feasible(problem, result)
        assert controller is not None
        result.diagnostics["augmented_lagrangian"] = controller.diagnostics()
        result.diagnostics["constraint_archive"] = archive.diagnostics()
    if not requested_population:
        result.final_population = None
    return result


__all__ = ["solve_model_ir"]
