"""High-level entry point for mixed-variable optimisation."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch

from qqa.annealing import AnnealResult, anneal
from qqa.callbacks import Callback
from qqa.mixed.augmented_lagrangian import (
    AdaptiveALCallback,
    AdaptiveAugmentedLagrangian,
    ConstraintArchive,
    ConstraintArchiveCallback,
)
from qqa.mixed.problem import MixedProblem
from qqa.mixed.repair import repair_mixed_solution
from qqa.utils import resolve_device


def _prefer_feasible_incumbent(problem: MixedProblem, result: AnnealResult) -> None:
    """Select by feasibility, maximum/total violation, then objective."""
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
        normalised = []
        for constraint in problem.constraints:
            feasible &= violations[constraint.name] <= constraint.tolerance
            normalised.append(violations[constraint.name] / constraint.scale)
        matrix = torch.stack(normalised, dim=1)
        maximum = matrix.amax(dim=1)
        total = matrix.sum(dim=1)
        objective = problem.objective_values(candidates)
        ranked = sorted(
            range(len(candidates)),
            key=lambda index: (
                not bool(feasible[index]),
                0.0 if bool(feasible[index]) else float(maximum[index]),
                0.0 if bool(feasible[index]) else float(total[index]),
                float(objective[index]),
            ),
        )
        selected = ranked[0]
        best_sol = candidates[selected].detach().clone()
        # ``AnnealResult.best_obj`` remains the internal search energy for
        # every candidate.  The original mathematical objective is exposed
        # separately through ``score['value']`` and the stable SolveResult.
        # Never switch this field's meaning based on feasibility.
        best_obj = float(problem.loss_fn(best_sol.unsqueeze(0))[0].item())
    result.best_sol = best_sol
    result.best_obj = best_obj
    result.score = problem.score_summary(best_sol)


def _snap_nearby_real_bounds(problem: MixedProblem, result: AnnealResult) -> int:
    """Snap numerical boundary residue iff feasibility and objective are preserved."""
    real_indices = [index for index, kind in enumerate(problem.space.kinds) if kind == "real"]
    if not real_indices:
        return 0
    incumbent = result.best_sol.detach().clone()
    encoded = problem.space.encode(incumbent)
    lower = problem.space.decode(torch.zeros_like(encoded))
    upper = problem.space.decode(torch.ones_like(encoded))
    tolerance = math.sqrt(torch.finfo(encoded.dtype).eps)
    with torch.no_grad():
        incumbent_objective = float(problem.objective_values(incumbent.unsqueeze(0))[0])
        snaps = 0
        for index in real_indices:
            boundary = None
            if float(encoded[index]) <= tolerance:
                boundary = lower[index]
            elif float(1.0 - encoded[index]) <= tolerance:
                boundary = upper[index]
            if boundary is None or bool(incumbent[index] == boundary):
                continue
            candidate = incumbent.clone()
            candidate[index] = boundary
            violations = problem.constraint_violations(candidate.unsqueeze(0))
            feasible = all(
                float(violations[constraint.name][0]) <= constraint.tolerance
                for constraint in problem.constraints
            )
            candidate_objective = float(problem.objective_values(candidate.unsqueeze(0))[0])
            if feasible and candidate_objective <= incumbent_objective:
                incumbent = candidate
                incumbent_objective = candidate_objective
                snaps += 1
        if snaps:
            result.best_sol = incumbent
            result.best_obj = float(problem.loss_fn(incumbent.unsqueeze(0))[0])
            result.score = problem.score_summary(incumbent)
    return snaps


def _repair_candidates(
    problem: MixedProblem,
    result: AnnealResult,
    *,
    max_candidates: int,
    max_steps: int,
) -> dict[str, int | float]:
    if not problem.constraints or result.final_population is None:
        return {"attempted": 0, "feasible": 0, "improved": 0}
    if not any(kind == "real" for kind in problem.space.kinds):
        return {"attempted": 0, "feasible": 0, "improved": 0}
    candidates = torch.cat(
        [result.best_sol.detach().reshape(1, -1), result.final_population.detach()],
        dim=0,
    )
    with torch.no_grad():
        violations = problem.constraint_violations(candidates)
        matrix = torch.stack(
            [violations[row.name] / row.scale for row in problem.constraints], dim=1
        )
        maximum = matrix.amax(dim=1)
        total = matrix.sum(dim=1)
        objective = problem.objective_values(candidates)
    ranked = sorted(
        range(len(candidates)),
        key=lambda index: (float(maximum[index]), float(total[index]), float(objective[index])),
    )
    repaired = []
    feasible_count = 0
    improved_count = 0
    for index in ranked[:max_candidates]:
        repair = repair_mixed_solution(problem, candidates[index], max_steps=max_steps)
        repaired.append(repair.solution)
        feasible_count += int(repair.feasible)
        if repair.maximum_violation < float(maximum[index]) - 1e-10 or (
            repair.feasible
            and float(maximum[index]) <= 1e-6
            and repair.objective < float(objective[index]) - 1e-10
        ):
            improved_count += 1
    if repaired:
        result.final_population = torch.cat(
            [result.final_population, torch.stack(repaired).to(result.final_population)],
            dim=0,
        )
    return {
        "attempted": len(repaired),
        "feasible": feasible_count,
        "improved": improved_count,
    }


def solve_mixed(
    problem: MixedProblem,
    *,
    calibrate_penalty: bool = True,
    calibration_points: int = 256,
    penalty_safety_factor: float = 50.0,
    max_penalty_multiplier: float = 1e8,
    adaptive_augmented_lagrangian: bool = True,
    al_update_interval: int = 50,
    al_rho_growth: float = 2.0,
    al_maximum_rho: float = 1e10,
    repair: bool = True,
    repair_candidates: int = 4,
    repair_steps: int = 150,
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
    if not isinstance(adaptive_augmented_lagrangian, bool):
        raise TypeError("adaptive_augmented_lagrangian must be boolean.")
    if not isinstance(repair, bool):
        raise TypeError("repair must be boolean.")
    for name, value in (
        ("al_update_interval", al_update_interval),
        ("repair_candidates", repair_candidates),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(repair_steps, bool) or not isinstance(repair_steps, int) or repair_steps < 0:
        raise ValueError("repair_steps must be a non-negative integer.")
    defaults: dict[str, Any] = {
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
    controller = None
    archive = ConstraintArchive() if problem.constraints else None
    callbacks: list[Callback] = list(defaults.get("callbacks", ()))
    if archive is not None:
        callbacks.append(
            ConstraintArchiveCallback(
                archive,
                update_interval=min(10, al_update_interval),
            )
        )
    if adaptive_augmented_lagrangian and problem.constraints:
        controller = AdaptiveAugmentedLagrangian.for_problem(
            solving_problem,
            penalty_multiplier=solving_problem.penalty_multiplier,
            rho_growth=al_rho_growth,
            maximum_rho=al_maximum_rho,
        )
        solving_problem._augmented_lagrangian = controller
        callbacks.append(AdaptiveALCallback(update_interval=al_update_interval))
    if callbacks:
        defaults["callbacks"] = callbacks
    result = anneal(solving_problem, **defaults)
    if archive is not None and result.final_population is not None:
        archive.update(
            solving_problem,
            torch.cat(
                [result.best_sol.detach().reshape(1, -1), result.final_population.detach()],
                dim=0,
            ),
        )
        archived = archive.candidates()
        if archived:
            result.final_population = torch.cat(
                [result.final_population, torch.stack(archived).to(result.final_population)],
                dim=0,
            )
    repair_diagnostics = (
        _repair_candidates(
            solving_problem,
            result,
            max_candidates=repair_candidates,
            max_steps=repair_steps,
        )
        if repair
        else {"attempted": 0, "feasible": 0, "improved": 0}
    )
    _prefer_feasible_incumbent(solving_problem, result)
    real_bound_snaps = _snap_nearby_real_bounds(solving_problem, result)
    result.diagnostics["penalty_multiplier"] = float(solving_problem.penalty_multiplier)
    result.diagnostics["adaptive_augmented_lagrangian"] = (
        controller.diagnostics() if controller is not None else None
    )
    result.diagnostics["repair"] = repair_diagnostics
    result.diagnostics["real_bound_snaps"] = real_bound_snaps
    result.diagnostics["constraint_archive"] = (
        archive.diagnostics() if archive is not None else None
    )
    result.diagnostics["objective_value"] = float(result.score.get("value", result.best_obj))
    result.diagnostics["internal_energy"] = float(result.best_obj)
    if not return_population:
        result.final_population = None
    return result
