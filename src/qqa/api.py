"""Stable one-entry solve, inspect, and plan API."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from qqa.algebraic import AlgebraicModel
from qqa.annealing import anneal
from qqa.config import SolverConfig
from qqa.model import ModelIR, ObjectiveSense
from qqa.model.adapters import algebraic_to_model_ir
from qqa.model.presolve import PresolveResult, presolve_model
from qqa.model.problem import ModelIRProblem
from qqa.portfolio import ModelInspection, SolverPlan, build_plan, inspect_model
from qqa.result import (
    ConstraintReport,
    ConstraintViolation,
    Provenance,
    ResourceReport,
    SolveResult,
    SolveStatus,
    TimingReport,
)
from qqa.utils import fix_seed, resolve_device, safe_score_summary


def _load_model(model: Any) -> Any:
    if not isinstance(model, (str, Path)):
        return model
    path = Path(model).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path.name}")
    name = path.name.lower()
    if name.endswith(".qplib"):
        from qqa.io import load_qplib

        return load_qplib(path)
    if any(name.endswith(suffix) for suffix in (".mps", ".mps.gz", ".mps.bz2", ".lp", ".lp.gz")):
        from qqa.io import load_mps

        return load_mps(path)
    from qqa.io.formats import load_portable_model

    return load_portable_model(path)


def _resolve_config(
    *,
    profile: str,
    budget: float | None,
    device: str,
    config: SolverConfig | None,
    overrides: dict[str, Any],
) -> SolverConfig:
    if config is None:
        values = {"profile": profile, "budget": budget, "device": device, **overrides}
        return SolverConfig.from_mapping(values)
    if not isinstance(config, SolverConfig):
        raise TypeError("config must be a SolverConfig or None.")
    explicit = dict(overrides)
    if profile != "balanced":
        explicit["profile"] = profile
    if budget is not None:
        explicit["budget"] = budget
    if device != "auto":
        explicit["device"] = device
    return SolverConfig.from_mapping({**config.to_dict(), **explicit})


def inspect(model: Any) -> ModelInspection:
    """Load if necessary and return solver-independent structural features."""
    return inspect_model(_load_model(model))


def plan(
    model: Any,
    *,
    profile: str = "balanced",
    budget: float | None = None,
    device: str = "auto",
    config: SolverConfig | None = None,
    **overrides: Any,
) -> SolverPlan:
    """Build an explainable plan without executing the solver."""
    loaded = _load_model(model)
    resolved = _resolve_config(
        profile=profile,
        budget=budget,
        device=device,
        config=config,
        overrides=overrides,
    )
    resolved = replace(resolved, device=str(resolve_device(resolved.device)))
    return build_plan(loaded, resolved)


def _constraint_report(score: dict[str, Any], *, use_raw: bool = False) -> ConstraintReport:
    extra = score.get("extra", {}) if isinstance(score, dict) else {}
    constraints = extra.get("constraints", {}) if isinstance(extra, dict) else {}
    rows = []
    if isinstance(constraints, dict):
        for name, record in constraints.items():
            if not isinstance(record, dict):
                continue
            violation = float(record.get("violation", 0.0))
            scale = float(record.get("scale", 1.0))
            scaled = float(record.get("scaled_violation", violation / max(scale, 1e-30)))
            tolerance = float(record.get("tolerance", 1e-6))
            rows.append(
                ConstraintViolation(
                    str(name),
                    violation,
                    scaled,
                    tolerance,
                    bool(record.get("feasible", violation <= tolerance)),
                )
            )
    feasible = bool(score.get("feasible", True)) if isinstance(score, dict) else True
    if use_raw and "raw_feasible" in extra:
        feasible = bool(extra["raw_feasible"])
    if not rows and not feasible:
        rows.append(ConstraintViolation("model_feasibility", 1.0, 1.0, 0.0, False))
    return ConstraintReport(tuple(rows))


def _score_value(score: dict[str, Any], fallback: float) -> float:
    value = score.get("value", fallback)
    if torch.is_tensor(value):
        value = value.reshape(-1)[0].item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(fallback)
    return result if math.isfinite(result) else float(fallback)


def _objective_value(
    problem: Any,
    solution: torch.Tensor,
    score: dict[str, Any],
    fallback: float,
) -> float:
    """Prefer the model's original objective over a repaired display score."""
    evaluator = getattr(problem, "objective_values", None)
    if callable(evaluator):
        values = solution.unsqueeze(0) if solution.ndim == 1 else solution
        with torch.no_grad():
            objective = evaluator(values).reshape(-1)[0]
        value = float(objective.item())
        if math.isfinite(value):
            return value
    return _score_value(score, fallback)


def _model_ir_score(model: ModelIR, solution: torch.Tensor) -> dict[str, Any]:
    values = solution.unsqueeze(0)
    objective = float(model.objective_values(values)[0].item())
    violations = model.constraint_violations(values)
    rows = {}
    feasible = True
    for row in model.constraints:
        violation = float(violations[row.name][0].item())
        satisfied = violation <= row.tolerance
        feasible &= satisfied
        rows[row.name] = {
            "lhs": float(row.expression.evaluate(values)[0].item()),
            "sense": row.sense,
            "rhs": row.rhs,
            "violation": violation,
            "scaled_violation": violation / row.scale,
            "tolerance": row.tolerance,
            "feasible": satisfied,
        }
    return {
        "label": "objective",
        "value": objective,
        "unit": "",
        "feasible": feasible,
        "extra": {"constraints": rows, "sense": ObjectiveSense(model.sense).value},
    }


def _legacy_status(result: Any, feasible: bool) -> tuple[SolveStatus, bool]:
    scip_status = str(getattr(result, "scip_status", "")).lower()
    if scip_status == "optimal":
        return SolveStatus.OPTIMAL, True
    if "infeasible" in scip_status:
        return SolveStatus.INFEASIBLE, False
    deadline = bool(getattr(result, "diagnostics", {}).get("deadline_reached", False))
    if "timelimit" in scip_status or deadline:
        return SolveStatus.TIME_LIMIT, False
    return (SolveStatus.FEASIBLE if feasible else SolveStatus.UNKNOWN), False


def solve(
    model: Any,
    *,
    profile: str = "balanced",
    budget: float | None = None,
    device: str = "auto",
    config: SolverConfig | None = None,
    initial_solution: torch.Tensor | None = None,
    **overrides: Any,
) -> SolveResult:
    """Solve any supported model through one QQA-centred, strict API.

    Exact backends are opt-in through ``profile='certify'`` or
    ``exact_backend=...``.  The default path remains pure QQA.
    """
    loaded = _load_model(model)
    resolved = _resolve_config(
        profile=profile,
        budget=budget,
        device=device,
        config=config,
        overrides=overrides,
    )
    resolved = replace(resolved, device=str(resolve_device(resolved.device))).resolved()
    fix_seed(resolved.seed)
    torch.use_deterministic_algorithms(resolved.deterministic, warn_only=True)
    solver_plan = build_plan(loaded, resolved)
    resolved = replace(resolved, replicas=solver_plan.replicas)

    original_ir = None
    presolved: PresolveResult | None = None
    if isinstance(loaded, AlgebraicModel):
        original_ir = algebraic_to_model_ir(loaded)
        presolved = presolve_model(original_ir)
        problem = ModelIRProblem(presolved.model)
    elif isinstance(loaded, ModelIR):
        original_ir = loaded
        if original_ir.structured_block is None:
            presolved = presolve_model(original_ir)
            problem = ModelIRProblem(presolved.model)
        else:
            problem = ModelIRProblem(original_ir)
    else:
        problem = loaded

    is_cuda = resolved.device.startswith("cuda") and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(torch.device(resolved.device))
    started = perf_counter()
    kwargs = resolved.anneal_kwargs()
    kwargs["sol_size"] = solver_plan.replicas
    kwargs["compile_core"] = resolved.compile_core
    if initial_solution is not None:
        if resolved.backend != "qqa":
            raise ValueError("initial_solution is currently supported by the QQA backend only.")
        if not torch.is_tensor(initial_solution):
            raise TypeError("initial_solution must be a torch.Tensor or None.")
        initial = initial_solution.detach().clone()
        kwargs["initial_state"] = presolved.reduce(initial) if presolved is not None else initial

    if solver_plan.exact_backend is not None and not isinstance(loaded, AlgebraicModel):
        raise NotImplementedError(
            "Stable exact completion currently requires an algebraic MPS, LP, or "
            "QPLIB model; no requested certificate was silently skipped."
        )
    if solver_plan.exact_backend is not None and isinstance(loaded, AlgebraicModel):
        # QQA remains the primal generator.  The optional exact backend then
        # receives its incumbent and spends the remainder on completion and
        # certification.
        from qqa.hybrid import solve_exact_algebraic
        from qqa.model.solve import solve_model_ir

        total_budget = float(resolved.budget or 60.0)
        qqa_kwargs = dict(kwargs)
        qqa_kwargs["time_limit"] = min(total_budget * 0.35, kwargs.get("time_limit") or math.inf)
        qqa_kwargs["return_population"] = True
        warm_result = solve_model_ir(problem, **qqa_kwargs)
        qqa_elapsed = perf_counter() - started
        remaining = max(1e-3, total_budget - qqa_elapsed)
        legacy = solve_exact_algebraic(
            loaded,
            solver_plan.exact_backend,
            time_limit=remaining,
            warm_start=(
                presolved.restore(warm_result.best_sol)
                if presolved is not None
                else warm_result.best_sol
            ),
            threads=1,
            relative_gap=0.0,
            verbose=False,
        )
        legacy.runtime = perf_counter() - started
        legacy.final_population = warm_result.final_population
        legacy.diagnostics.update(
            {
                "qqa_warm_start_time": qqa_elapsed,
                "certification_time": max(0.0, legacy.runtime - qqa_elapsed),
                "qqa_warm_start": True,
            }
        )
    elif resolved.backend == "sa":
        from qqa.sa import simulated_annealing

        legacy = simulated_annealing(
            problem,
            sol_size=solver_plan.replicas,
            num_sweeps=int(resolved.epochs or 0),
            seed=resolved.seed,
            device=resolved.device,
            polish=resolved.polish,
            verbose=False,
        )
    elif resolved.backend == "pa":
        from qqa.pa import population_annealing

        temperature_steps = max(2, int(math.sqrt(int(resolved.epochs or 0))))
        sweeps = max(1, int(math.ceil(int(resolved.epochs or 0) / temperature_steps)))
        legacy = population_annealing(
            problem,
            sol_size=solver_plan.replicas,
            num_temps=temperature_steps,
            sweeps_per_temp=sweeps,
            seed=resolved.seed,
            device=resolved.device,
            polish=resolved.polish,
            verbose=False,
        )
    elif resolved.backend == "isco":
        from qqa.isco import discrete_langevin

        legacy = discrete_langevin(
            problem,
            sol_size=solver_plan.replicas,
            num_steps=int(resolved.epochs or 0),
            seed=resolved.seed,
            device=resolved.device,
            polish=resolved.polish,
            verbose=False,
        )
    elif solver_plan.exact_backend == "scip" and hasattr(problem, "generate_qubo_matrix"):
        from qqa.hybrid import solve_qqa_scip

        total_budget = resolved.budget or 60.0
        legacy = solve_qqa_scip(
            problem,
            qqa_kwargs=kwargs,
            time_limit=total_budget,
            verbose=False,
        )
    elif solver_plan.exact_backend not in {None, "scip"}:
        raise NotImplementedError(
            f"Exact backend {solver_plan.exact_backend!r} does not support this model route."
        )
    else:
        sparse_qubo = getattr(problem, "sparse_qubo", None)
        if isinstance(problem, ModelIRProblem):
            from qqa.model.solve import solve_model_ir

            legacy = solve_model_ir(problem, **kwargs)
        elif sparse_qubo is not None and len(sparse_qubo.connected_components()) > 1:
            from qqa.engines.qqa import anneal_components

            legacy = anneal_components(problem, **kwargs)
        else:
            legacy = anneal(problem, **kwargs)
    total_time = perf_counter() - started

    reduced_raw_solution = legacy.best_sol.detach().clone()
    raw_solution = (
        presolved.restore(reduced_raw_solution) if presolved is not None else reduced_raw_solution
    )
    merit_value = float(legacy.best_obj)
    raw_score = (
        _model_ir_score(original_ir, raw_solution)
        if original_ir is not None
        else safe_score_summary(problem, raw_solution, fallback_obj=merit_value)
    )
    raw_objective = (
        float(original_ir.objective_values(raw_solution)[0].item())
        if original_ir is not None
        else _objective_value(problem, raw_solution, raw_score, merit_value)
    )
    internal_energy = (
        float(original_ir.internal_energy(raw_solution)[0].item())
        if original_ir is not None
        else merit_value
    )
    raw_report = _constraint_report(raw_score, use_raw=True)

    repaired_solution = None
    repaired_objective = None
    final_score = raw_score
    repair_started = perf_counter()
    repair_function: Callable[[torch.Tensor], Any] | None
    if original_ir is not None:
        from qqa.repair import repair_model_ir

        def repair_function(candidate):
            return repair_model_ir(original_ir, candidate)

    else:
        repair_function = getattr(problem, "repair_solution", None)
    if callable(repair_function):
        candidate = repair_function(raw_solution.detach().clone())
        if torch.is_tensor(candidate) and not torch.equal(candidate, raw_solution):
            repaired_solution = candidate.detach().clone()
            final_score = (
                _model_ir_score(original_ir, repaired_solution)
                if original_ir is not None
                else safe_score_summary(problem, repaired_solution, fallback_obj=merit_value)
            )
            repaired_objective = (
                float(original_ir.objective_values(repaired_solution)[0].item())
                if original_ir is not None
                else _objective_value(problem, repaired_solution, final_score, merit_value)
            )
    repair_time = perf_counter() - repair_started if repaired_solution is not None else 0.0
    final_report = _constraint_report(final_score)
    final_feasible = final_report.feasible
    status, proven = _legacy_status(legacy, final_feasible)
    best_bound = getattr(legacy, "dual_bound", None)
    relative_gap = getattr(legacy, "gap", None)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(torch.device(resolved.device))) if is_cuda else None
    )
    certification_time = float(getattr(legacy, "diagnostics", {}).get("certification_time", 0.0))
    search_time = max(0.0, total_time - repair_time - certification_time)
    final_population = getattr(legacy, "final_population", getattr(legacy, "final_x", None))
    if presolved is not None and final_population is not None:
        final_population = presolved.restore(final_population)
    return SolveResult(
        status=status,
        raw_solution=raw_solution,
        repaired_solution=repaired_solution,
        objective_value=raw_objective,
        repaired_objective_value=repaired_objective,
        internal_energy=internal_energy,
        merit_value=merit_value,
        feasible=final_feasible,
        violations=final_report,
        best_bound=best_bound,
        relative_gap=relative_gap,
        proven_optimal=proven,
        population=final_population,
        timings=TimingReport(
            total_time,
            search=search_time,
            repair=repair_time,
            certification=certification_time,
        ),
        resources=ResourceReport(
            resolved.device,
            resolved.mixed_precision,
            peak_device_memory_bytes=peak_memory,
        ),
        provenance=Provenance(
            backend=(
                f"{solver_plan.primary_engine}+{solver_plan.exact_backend}"
                if solver_plan.exact_backend is not None
                else solver_plan.primary_engine
            ),
            seed=resolved.seed,
            profile=resolved.profile,
            config=resolved.to_dict(),
            transformations=(
                tuple(record.operation for record in presolved.model.transformations)
                if presolved is not None
                else ()
            ),
        ),
        plan=solver_plan,
        score=final_score,
        diagnostics={
            **dict(getattr(legacy, "diagnostics", {})),
            "raw_feasible": raw_report.feasible,
        },
        legacy_result=legacy,
    )


__all__ = ["inspect", "plan", "solve"]
