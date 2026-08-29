"""Stable one-entry solve, inspect, and plan API."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

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
    CertificateMetadata,
    ConstraintReport,
    ConstraintViolation,
    Provenance,
    ResourceReport,
    SolveResult,
    SolveStatus,
    TimingReport,
)
from qqa.runtime.events import EventKind, EventRecorder
from qqa.runtime.population import WarmStateBundle
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


SolveGoal = Literal["best", "feasible", "prove", "diverse", "pareto"]


def _parse_budget(value: float | str | None) -> float | None:
    if value is None or isinstance(value, (int, float)):
        return None if value is None else float(value)
    if not isinstance(value, str):
        raise TypeError("budget must be seconds, a duration string, or None.")
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)\s*", value.lower())
    if match is None:
        raise ValueError("budget strings must use ms, s, m, or h, for example '60s'.")
    amount = float(match.group(1))
    multiplier = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}[match.group(2)]
    return amount * multiplier


def _profile_for_goal(profile: str, goal: SolveGoal) -> str:
    if goal not in {"best", "feasible", "prove", "diverse", "pareto"}:
        raise ValueError("goal must be best, feasible, prove, diverse, or pareto.")
    if profile != "balanced" or goal == "best":
        return profile
    return {
        "feasible": "fast",
        "prove": "certify",
        "diverse": "diverse",
        "pareto": "pareto",
    }[goal]


def _resolve_config(
    *,
    profile: str,
    budget: float | str | None,
    device: str,
    config: SolverConfig | None,
    overrides: dict[str, Any],
) -> SolverConfig:
    if profile == "prove":
        profile = "certify"
    budget = _parse_budget(budget)
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


def doctor(model: Any, *, replicas: int = 128):
    """Run deterministic model, route, scaling, and resource diagnostics."""
    from qqa.model.doctor import diagnose_model

    return diagnose_model(_load_model(model), replicas=replicas)


def plan(
    model: Any,
    *,
    profile: str = "balanced",
    budget: float | str | None = None,
    device: str = "auto",
    config: SolverConfig | None = None,
    goal: SolveGoal = "best",
    **overrides: Any,
) -> SolverPlan:
    """Build an explainable plan without executing the solver."""
    loaded = _load_model(model)
    profile = _profile_for_goal(profile, goal)
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
    declared_feasible = score.get("feasible") if isinstance(score, dict) else None
    evaluated = bool(rows) or declared_feasible is not None
    feasible = bool(declared_feasible) if declared_feasible is not None else False
    if use_raw and "raw_feasible" in extra:
        feasible = bool(extra["raw_feasible"])
    if not rows and not feasible:
        rows.append(ConstraintViolation("model_feasibility", 1.0, 1.0, 0.0, False))
    return ConstraintReport(tuple(rows), evaluated=evaluated)


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
            "search_weight": row.weight,
            "priority": row.priority,
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
    compact_status = re.sub(r"[^a-z0-9]+", "", scip_status)
    if compact_status == "optimal":
        return (SolveStatus.OPTIMAL, True) if feasible else (SolveStatus.UNKNOWN, False)
    if compact_status == "inforunbd" or (
        "infeasible" in compact_status and "unbounded" in compact_status
    ):
        return SolveStatus.INFEASIBLE_OR_UNBOUNDED, False
    if "infeasible" in compact_status:
        return SolveStatus.INFEASIBLE_PROVEN, False
    if "unbounded" in compact_status:
        return SolveStatus.UNBOUNDED_PROVEN, False
    if "invalid" in compact_status:
        return SolveStatus.MODEL_INVALID, False
    if any(token in compact_status for token in ("numerical", "nan", "singular")):
        return SolveStatus.NUMERICAL_FAILURE, False
    if any(token in compact_status for token in ("interrupt", "cancel")):
        return SolveStatus.INTERRUPTED, False
    deadline = bool(getattr(result, "diagnostics", {}).get("deadline_reached", False))
    limit_reached = deadline or any(
        token in compact_status
        for token in (
            "timelimit",
            "nodelimit",
            "memlimit",
            "gaplimit",
            "sollimit",
            "restartlimit",
            "iterationlimit",
        )
    )
    if limit_reached:
        return (
            SolveStatus.LIMIT_REACHED_WITH_INCUMBENT
            if getattr(result, "best_sol", None) is not None
            else SolveStatus.LIMIT_REACHED_NO_INCUMBENT,
            False,
        )
    if getattr(result, "best_sol", None) is None:
        return SolveStatus.NO_SOLUTION_FOUND, False
    return (SolveStatus.FEASIBLE if feasible else SolveStatus.UNKNOWN), False


def _diverse_algebraic_warm_starts(
    model: AlgebraicModel,
    candidates: list[torch.Tensor],
    *,
    count: int = 16,
) -> tuple[torch.Tensor, ...]:
    """Rank exact warm starts by feasibility/quality, then retain diversity."""
    rows = []
    seen = set()
    for candidate in candidates:
        tensor = torch.as_tensor(candidate).detach().cpu().to(torch.float64)
        tensor = tensor.reshape(1, -1) if tensor.ndim == 1 else tensor.reshape(-1, tensor.shape[-1])
        for row in tensor:
            if row.numel() != model.num_variables or not torch.isfinite(row).all():
                continue
            digest = row.numpy().tobytes()
            if digest in seen:
                continue
            seen.add(digest)
            evaluation = model.evaluate(row.numpy())
            canonical = (
                evaluation.objective
                if model.objective_sense == "minimize"
                else -evaluation.objective
            )
            rows.append((row.clone(), evaluation.maximum_infeasibility, canonical))
    if not rows:
        return ()
    rows.sort(key=lambda item: (item[1] > 1e-6, item[1], item[2]))
    shortlist = rows[: max(count, min(len(rows), count * 8))]
    selected = [0]
    values = torch.stack([item[0] for item in shortlist])
    scale = values.amax(dim=0) - values.amin(dim=0)
    scale = scale.clamp_min(1.0)
    normalised = values / scale
    while len(selected) < min(count, len(shortlist)):
        remaining = [index for index in range(len(shortlist)) if index not in selected]
        separation = torch.cdist(normalised[remaining], normalised[selected], p=1).amin(dim=1)
        quality = torch.tensor(
            [1.0 - rank / max(1, len(shortlist) - 1) for rank in remaining],
            dtype=separation.dtype,
        )
        chosen = int(torch.argmax(separation + 0.05 * quality).item())
        selected.append(remaining[chosen])
    return tuple(shortlist[index][0] for index in selected)


def _retain_verified_warm_incumbent(
    model: AlgebraicModel,
    exact_result: Any,
    candidate: torch.Tensor | None,
) -> bool:
    """Keep a verified QQA incumbent when an interrupted exact run returns none."""
    if getattr(exact_result, "best_sol", None) is not None or candidate is None:
        return False
    compact_status = re.sub(
        r"[^a-z0-9]+", "", str(getattr(exact_result, "scip_status", "")).lower()
    )
    if (
        compact_status == "optimal"
        or compact_status == "inforunbd"
        or any(token in compact_status for token in ("infeasible", "unbounded", "invalid"))
    ):
        return False
    values = torch.as_tensor(candidate).detach().reshape(-1).cpu().to(torch.float64)
    if values.numel() != model.num_variables or not torch.isfinite(values).all():
        return False
    evaluation = model.evaluate(values.numpy())
    if not evaluation.feasible:
        return False
    exact_result.best_sol = values
    exact_result.best_obj = (
        evaluation.objective if model.objective_sense == "minimize" else -evaluation.objective
    )
    exact_result.diagnostics["verified_warm_incumbent_retained"] = True
    return True


def solve(
    model: Any,
    *,
    profile: str = "balanced",
    budget: float | str | None = None,
    device: str = "auto",
    config: SolverConfig | None = None,
    initial_solution: torch.Tensor | None = None,
    warm_states: WarmStateBundle | None = None,
    goal: SolveGoal = "best",
    checkpoint_path: str | Path | None = None,
    checkpoint_interval: int | None = None,
    resume_from: str | Path | None = None,
    **overrides: Any,
) -> Any:
    """Solve any supported model through one QQA-centred, strict API.

    Exact backends are opt-in through ``profile='certify'`` or
    ``exact_backend=...``.  The default path remains pure QQA.
    """
    loaded = _load_model(model)
    profile = _profile_for_goal(profile, goal)
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
    from qqa.multiobjective import MultiObjectiveProblem, pareto_anneal

    if isinstance(loaded, MultiObjectiveProblem):
        if goal != "pareto" and resolved.profile != "pareto":
            raise TypeError("MultiObjectiveProblem requires goal='pareto' or profile='pareto'.")
        return pareto_anneal(
            loaded,
            sol_size=int(resolved.replicas or 1),
            num_epochs=int(resolved.epochs or 0),
            learning_rate=float(resolved.learning_rate or 0.05),
            temp=resolved.temperature,
            min_bg=resolved.min_bg,
            max_bg=resolved.max_bg,
            curve_rate=resolved.curve_rate,
            div_param=float(resolved.diversity or 0.0),
            archive_size=max(1, resolved.archive_size),
            gradient_clip_norm=resolved.gradient_clip_norm,
            seed=resolved.seed,
            device=resolved.device,
            time_limit=resolved.budget,
            verbose=False,
        )
    solver_plan = build_plan(loaded, resolved)
    resolved = replace(resolved, replicas=solver_plan.replicas)

    original_ir = None
    presolved: PresolveResult | None = None
    if isinstance(loaded, AlgebraicModel):
        original_ir = algebraic_to_model_ir(loaded)
        presolved = presolve_model(original_ir)
        try:
            problem = ModelIRProblem(presolved.model)
        except (NotImplementedError, ValueError):
            if solver_plan.exact_backend is None:
                raise
            # Exact solvers may legitimately support infinite bounds or a
            # factor with no differentiable QQA representation. In that case
            # certification proceeds without fabricating a QQA warm start.
            problem = None
    elif isinstance(loaded, ModelIR):
        original_ir = loaded
        if original_ir.structured_block is None:
            presolved = presolve_model(original_ir)
            problem = ModelIRProblem(presolved.model)
        else:
            problem = ModelIRProblem(original_ir)
    else:
        problem = loaded

    if getattr(problem, "sparse_qubo", None) is not None:
        dynamic_problem: Any = problem
        dynamic_problem.sparse_kernel = resolved.sparse_kernel

    is_cuda = resolved.device.startswith("cuda") and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(torch.device(resolved.device))
    started = perf_counter()
    kwargs = resolved.anneal_kwargs()
    kwargs["sol_size"] = solver_plan.replicas
    kwargs["compile_core"] = resolved.compile_core
    event_recorder = EventRecorder(stride=max(1, int(resolved.epochs or 1) // 200))
    event_recorder.emit(
        EventKind.SOLVE_STARTED,
        {"backend": solver_plan.primary_engine, "profile": resolved.profile},
        elapsed_seconds=0.0,
    )
    kwargs["callbacks"] = (*kwargs.get("callbacks", ()), event_recorder)
    if any(value is not None for value in (checkpoint_path, checkpoint_interval, resume_from)):
        if resolved.backend != "qqa":
            raise ValueError("Checkpoint/resume is currently supported by the QQA backend only.")
        kwargs["checkpoint_path"] = None if checkpoint_path is None else str(checkpoint_path)
        kwargs["checkpoint_interval"] = checkpoint_interval
        kwargs["resume_from"] = None if resume_from is None else str(resume_from)
    if initial_solution is not None and warm_states is not None:
        raise ValueError("Supply initial_solution or warm_states, not both.")
    if warm_states is not None:
        if resolved.backend != "qqa":
            raise ValueError("warm_states is currently supported by the QQA backend only.")
        if not isinstance(warm_states, WarmStateBundle):
            raise TypeError("warm_states must be a WarmStateBundle or None.")
        if presolved is None:
            kwargs["initial_state"] = warm_states
        else:
            kwargs["initial_state"] = WarmStateBundle(
                incumbent=(
                    None
                    if warm_states.incumbent is None
                    else presolved.reduce(warm_states.incumbent)
                ),
                lp_primal=(
                    None
                    if warm_states.lp_primal is None
                    else presolved.reduce(warm_states.lp_primal)
                ),
                conflict_avoiding=(
                    None
                    if warm_states.conflict_avoiding is None
                    else presolved.reduce(warm_states.conflict_avoiding)
                ),
            )
    if initial_solution is not None:
        if resolved.backend != "qqa":
            raise ValueError("initial_solution is currently supported by the QQA backend only.")
        if not torch.is_tensor(initial_solution):
            raise TypeError("initial_solution must be a torch.Tensor or None.")
        initial = initial_solution.detach().clone()
        kwargs["initial_state"] = presolved.reduce(initial) if presolved is not None else initial

    lp_relaxation = None
    if (
        isinstance(loaded, AlgebraicModel)
        and problem is not None
        and resolved.backend == "qqa"
        and resume_from is None
        and loaded.objective.is_linear
        and all(row.expression.is_linear for row in loaded.constraints)
    ):
        from qqa.dual import solve_lp_relaxation

        lp_budget = min(2.0, max(1e-4, float(resolved.budget or 20.0) * 0.1))
        lp_relaxation = solve_lp_relaxation(
            loaded,
            device=(resolved.device if resolved.device.startswith(("cpu", "cuda")) else "cpu"),
            max_iterations=2000,
            tolerance=1e-5,
            time_limit=lp_budget,
        )
        lp_primal = lp_relaxation.primal_solution
        if presolved is not None:
            lp_primal = presolved.reduce(lp_primal)
        existing = kwargs.get("initial_state")
        if isinstance(existing, WarmStateBundle):
            kwargs["initial_state"] = WarmStateBundle(
                incumbent=existing.incumbent,
                lp_primal=lp_primal,
                conflict_avoiding=existing.conflict_avoiding,
            )
        else:
            kwargs["initial_state"] = WarmStateBundle(
                incumbent=existing if torch.is_tensor(existing) else None,
                lp_primal=lp_primal,
            )
        event_recorder.emit(
            EventKind.RELAXATION_UPDATED,
            {
                "engine": "pdhg",
                "iterations": lp_relaxation.iterations,
                "kkt_residual": lp_relaxation.kkt_residual,
            },
        )
        if resolved.budget is not None:
            kwargs["time_limit"] = max(1e-4, resolved.budget - (perf_counter() - started))

    legacy: Any
    if solver_plan.exact_backend == "cpsat" and isinstance(loaded, ModelIR):
        from qqa.exact import solve_cp_model_ir

        legacy = solve_cp_model_ir(
            loaded,
            time_limit=resolved.budget,
            random_seed=resolved.seed,
            workers=1,
        )
        # CP-SAT returns the original variable order directly.
        presolved = None
    elif solver_plan.exact_backend is not None and not isinstance(loaded, AlgebraicModel):
        raise NotImplementedError(
            "Stable exact completion currently requires an algebraic MPS, LP, or "
            "QPLIB model; no requested certificate was silently skipped."
        )
    elif solver_plan.exact_backend is not None and isinstance(loaded, AlgebraicModel):
        # QQA remains the primal generator.  The optional exact backend then
        # receives its incumbent and spends the remainder on completion and
        # certification.
        from qqa.hybrid import solve_exact_algebraic
        from qqa.model.solve import solve_model_ir

        total_budget = float(resolved.budget or 60.0)
        warm_result = None
        qqa_elapsed = 0.0
        if problem is not None:
            qqa_started = perf_counter()
            qqa_kwargs = dict(kwargs)
            qqa_stage = solver_plan.stage("qqa-primal")
            qqa_fraction = 0.35 if qqa_stage is None else qqa_stage.budget_fraction
            qqa_kwargs["time_limit"] = min(
                total_budget * qqa_fraction, kwargs.get("time_limit") or math.inf
            )
            qqa_kwargs["return_population"] = True
            warm_result = solve_model_ir(problem, **qqa_kwargs)
            qqa_elapsed = perf_counter() - qqa_started
        remaining = max(1e-3, total_budget - (perf_counter() - started))
        exact_candidates = [] if initial_solution is None else [initial_solution]
        if warm_result is not None:
            qqa_best = (
                presolved.restore(warm_result.best_sol)
                if presolved is not None
                else warm_result.best_sol
            )
            exact_candidates.append(qqa_best)
            if warm_result.final_population is not None:
                qqa_population = (
                    presolved.restore(warm_result.final_population)
                    if presolved is not None
                    else warm_result.final_population
                )
                exact_candidates.append(qqa_population)
        exact_warm_starts = _diverse_algebraic_warm_starts(loaded, exact_candidates)
        exact_warm_start = exact_warm_starts[0] if exact_warm_starts else None
        legacy = solve_exact_algebraic(
            loaded,
            solver_plan.exact_backend,
            time_limit=remaining,
            warm_start=exact_warm_start,
            warm_starts=exact_warm_starts[1:],
            threads=1,
            relative_gap=0.0,
            verbose=False,
        )
        _retain_verified_warm_incumbent(loaded, legacy, exact_warm_start)
        legacy.runtime = perf_counter() - started
        legacy.final_population = None if warm_result is None else warm_result.final_population
        legacy.diagnostics.update(
            {
                "qqa_warm_start_time": qqa_elapsed,
                "qqa_budget_fraction": qqa_fraction,
                "certification_time": max(0.0, legacy.runtime - qqa_elapsed),
                "qqa_warm_start": warm_result is not None,
                "qqa_warm_start_candidates": len(exact_warm_starts),
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
    search_finished = perf_counter()

    raw_candidate = getattr(legacy, "best_sol", None)
    if not torch.is_tensor(raw_candidate) or raw_candidate.numel() == 0:
        no_incumbent_status, _ = _legacy_status(legacy, False)
        total_time = perf_counter() - started
        event_recorder.emit(
            EventKind.SOLVE_FINISHED,
            {"status": no_incumbent_status.value, "feasible": "unknown"},
            elapsed_seconds=total_time,
        )
        infeasibility_certificate = (
            CertificateMetadata(
                proof_system=str(solver_plan.exact_backend),
                status="solver-reported-infeasible",
                verifier=str(getattr(legacy, "backend", solver_plan.exact_backend)),
            )
            if no_incumbent_status is SolveStatus.INFEASIBLE_PROVEN
            else None
        )
        return SolveResult(
            status=no_incumbent_status,
            raw_solution=None,
            objective_value=None,
            internal_energy=None,
            merit_value=None,
            feasible=False,
            violations=ConstraintReport.unknown(),
            best_bound=getattr(legacy, "dual_bound", None),
            timings=TimingReport(total_time, search=total_time),
            resources=ResourceReport(resolved.device, resolved.mixed_precision),
            provenance=Provenance(
                backend=f"{solver_plan.primary_engine}+{solver_plan.exact_backend}",
                seed=resolved.seed,
                profile=resolved.profile,
                config=resolved.to_dict(),
            ),
            plan=solver_plan,
            events=tuple(event_recorder.events),
            certificate=infeasibility_certificate,
            diagnostics=dict(getattr(legacy, "diagnostics", {})),
            legacy_result=legacy,
        )
    reduced_raw_solution = raw_candidate.detach().clone()
    raw_solution = (
        presolved.restore(reduced_raw_solution) if presolved is not None else reduced_raw_solution
    )
    legacy_best_objective = getattr(legacy, "best_obj", None)
    if legacy_best_objective is None:
        raise RuntimeError("Solver returned a primal incumbent without an objective value.")
    merit_value = float(legacy_best_objective)
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
    if repaired_solution is not None:
        event_recorder.emit(
            EventKind.CANDIDATE_REPAIRED,
            {"objective": repaired_objective, "feasible": final_score.get("feasible")},
        )
    final_report = _constraint_report(final_score)
    final_feasible = final_report.feasible
    status, proven = _legacy_status(legacy, final_feasible)
    certificate: CertificateMetadata | None = (
        CertificateMetadata(
            proof_system=str(solver_plan.exact_backend),
            status="solver-reported-optimal",
            verifier=str(getattr(legacy, "backend", solver_plan.exact_backend)),
        )
        if proven and solver_plan.exact_backend is not None
        else None
    )
    best_bound = getattr(legacy, "dual_bound", None)
    if best_bound is None and lp_relaxation is not None:
        best_bound = lp_relaxation.dual_bound
    relative_gap = getattr(legacy, "gap", None)
    if relative_gap is None and best_bound is not None and final_feasible:
        reported_objective = repaired_objective if repaired_objective is not None else raw_objective
        relative_gap = abs(reported_objective - best_bound) / max(
            1.0, abs(reported_objective), abs(best_bound)
        )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(torch.device(resolved.device))) if is_cuda else None
    )
    certification_time = float(getattr(legacy, "diagnostics", {}).get("certification_time", 0.0))
    total_time = perf_counter() - started
    if best_bound is not None:
        event_recorder.emit(
            EventKind.DUAL_BOUND_IMPROVED,
            {"bound": float(best_bound), "relative_gap": relative_gap},
        )
    event_recorder.emit(
        EventKind.SOLVE_FINISHED,
        {
            "status": status.value,
            "objective": repaired_objective if repaired_objective is not None else raw_objective,
            "feasible": final_report.status.value,
        },
        elapsed_seconds=total_time,
    )
    search_time = max(0.0, search_finished - started - certification_time)
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
        archive=getattr(legacy, "archive", None),
        events=tuple(event_recorder.events),
        certificate=certificate,
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
            "pdhg": (
                None
                if lp_relaxation is None
                else {
                    "iterations": lp_relaxation.iterations,
                    "runtime": lp_relaxation.runtime,
                    "primal_residual": lp_relaxation.primal_residual,
                    "dual_residual": lp_relaxation.dual_residual,
                    "kkt_residual": lp_relaxation.kkt_residual,
                    "converged": lp_relaxation.converged,
                }
            ),
        },
        legacy_result=legacy,
    )


__all__ = ["SolveGoal", "doctor", "inspect", "plan", "solve"]
