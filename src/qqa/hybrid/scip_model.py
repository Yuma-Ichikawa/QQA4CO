"""Exact SCIP refinement for safe TeX/JSON mixed nonlinear models."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from numbers import Real
from time import perf_counter
from typing import Any

import torch

from qqa.annealing import AnnealResult
from qqa.tex.schema import ModelSpec


class SCIPExpressionError(ValueError):
    """Raised when a validated Torch expression cannot be represented by SCIP."""


@dataclass(slots=True)
class SCIPModelResult:
    """QQA warm-start exploration followed by an exact SCIP MINLP solve."""

    best_sol: torch.Tensor
    best_obj: float
    objective_value: float
    solver_loss: float
    runtime: float
    qqa_result: AnnealResult
    scip_runtime: float
    scip_status: str
    dual_bound: float | None
    gap: float | None
    n_warm_starts: int
    score: dict = field(default_factory=dict)

    @property
    def proven_optimal(self) -> bool:
        return self.scip_status.lower() == "optimal"

    @property
    def history(self) -> dict:
        return self.qqa_result.history


def _require_scip():
    try:
        import pyscipopt
        from pyscipopt import Model, quicksum
        from pyscipopt.recipes.nonlinear import set_nonlinear_objective
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "SCIP model solving requires the optional dependency. "
            "Install it with `pip install 'qqa[scip]'`."
        ) from exc
    return pyscipopt, Model, quicksum, set_nonlinear_objective


def _compile_scip_expression(source: str, variables: dict[str, list], pyscipopt, quicksum):
    """Translate the already validated safe AST into PySCIPOpt expressions."""
    tree = ast.parse(source, mode="eval")

    def scalar(value, label: str):
        if isinstance(value, list):
            raise SCIPExpressionError(f"{label} requires a scalar; use sum() or an index.")
        return value

    def visit(node: ast.AST):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            values = variables[node.id]
            return values[0] if len(values) == 1 else values
        if isinstance(node, ast.Subscript):
            return variables[node.value.id][node.slice.value]
        if isinstance(node, ast.UnaryOp):
            value = scalar(visit(node.operand), "unary operation")
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = scalar(visit(node.left), "binary operation")
            right = scalar(visit(node.right), "binary operation")
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise SCIPExpressionError(f"Unsupported operator {type(node.op).__name__}.")
        if isinstance(node, ast.Call):
            name = node.func.id
            arguments = [visit(argument) for argument in node.args]
            if name == "sum":
                value = arguments[0]
                return quicksum(value) if isinstance(value, list) else value
            arguments = [scalar(value, f"{name}()") for value in arguments]
            if name == "square":
                return arguments[0] * arguments[0]
            if name == "abs":
                return abs(arguments[0])
            if name in {"sqrt", "exp", "log", "sin", "cos"}:
                return getattr(pyscipopt, name)(arguments[0])
            if name == "tanh":
                doubled = 2.0 * arguments[0]
                exponential = pyscipopt.exp(doubled)
                return (exponential - 1.0) / (exponential + 1.0)
            if name == "minimum":
                return 0.5 * (arguments[0] + arguments[1] - abs(arguments[0] - arguments[1]))
            if name == "maximum":
                return 0.5 * (arguments[0] + arguments[1] + abs(arguments[0] - arguments[1]))
            raise SCIPExpressionError(f"Function {name!r} is not supported by SCIP.")
        raise SCIPExpressionError(f"Unsupported expression node {type(node).__name__}.")

    return visit(tree)


def _candidate_starts(problem, qqa_result: AnnealResult, max_starts: int) -> list[torch.Tensor]:
    candidates = [qqa_result.best_sol.detach().cpu().reshape(-1)]
    if qqa_result.final_population is not None:
        candidates.extend(row.detach().cpu().reshape(-1) for row in qqa_result.final_population)
    unique: dict[bytes, torch.Tensor] = {}
    for candidate in candidates:
        canonical = problem.space.project(problem.space.encode(candidate.to(torch.float64)))
        key = bytes(canonical.contiguous().numpy().tobytes())
        unique.setdefault(key, canonical)
    values = list(unique.values())
    if not values:
        return []
    stacked = torch.stack(values)
    with torch.no_grad():
        typed = stacked.to(dtype=problem.dtype)
        objective = problem.objective_values(typed).detach().cpu()
        if problem.constraints:
            raw_violations = problem.constraint_violations(typed)
            normalised = torch.stack(
                [
                    raw_violations[constraint.name].detach().cpu() / constraint.scale
                    for constraint in problem.constraints
                ],
                dim=1,
            )
            # SCIP enforces the mathematical constraints, not the reporting
            # tolerances. Prefer exactly feasible starts, then the smallest
            # maximum and total scaled residual before comparing objectives.
            maximum_violation = normalised.amax(dim=1)
            total_violation = normalised.sum(dim=1)
        else:
            maximum_violation = torch.zeros(len(stacked), dtype=stacked.dtype)
            total_violation = torch.zeros_like(maximum_violation)
    ranked = sorted(
        range(len(values)),
        key=lambda index: (
            maximum_violation[index].item() > 1e-8,
            float(maximum_violation[index].item()),
            float(total_violation[index].item()),
            float(objective[index].item()),
        ),
    )
    return [values[index] for index in ranked[:max_starts]]


def _exactly_feasible(problem, solution: torch.Tensor) -> bool:
    if not problem.constraints:
        return True
    with torch.no_grad():
        violations = problem.constraint_violations(solution.to(dtype=problem.dtype))
    return all(
        float(violations[constraint.name].reshape(-1)[0].item()) <= 1e-7
        for constraint in problem.constraints
    )


def solve_spec_scip(
    spec: ModelSpec | dict,
    *,
    qqa_kwargs: dict[str, Any] | None = None,
    time_limit: float = 60.0,
    relative_gap: float = 0.0,
    max_warm_starts: int = 32,
    threads: int = 1,
    verbose: bool = False,
) -> SCIPModelResult:
    """Solve a single-objective safe model with QQA exploration and SCIP.

    Unlike :func:`solve_qqa_scip`, this path preserves declared binary,
    integer, and real domains plus every nonlinear constraint from
    :class:`~qqa.tex.ModelSpec`. SCIP receives multiple diverse QQA primal
    starts and returns a proof status, dual bound, and optimality gap.
    """
    if isinstance(spec, dict):
        spec = ModelSpec.from_dict(spec)
    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a ModelSpec or dict.")
    if len(spec.objectives) != 1:
        raise ValueError("solve_spec_scip currently requires exactly one objective.")
    if (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, Real)
        or not math.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and > 0.")
    if (
        isinstance(relative_gap, bool)
        or not isinstance(relative_gap, Real)
        or not math.isfinite(relative_gap)
        or relative_gap < 0
    ):
        raise ValueError("relative_gap must be finite and >= 0.")
    if (
        isinstance(max_warm_starts, bool)
        or not isinstance(max_warm_starts, int)
        or max_warm_starts < 1
    ):
        raise ValueError("max_warm_starts must be a positive integer.")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")

    started = perf_counter()
    # Resolve the optional dependency and compile the full SCIP model before
    # spending time on QQA exploration. Unsupported expressions and missing
    # extras must fail immediately, not after a potentially long GPU run.
    pyscipopt, Model, quicksum, set_nonlinear_objective = _require_scip()
    model = Model(f"qqa-scip-{spec.name}")
    if not verbose:
        model.hideOutput()
    model.setRealParam("limits/time", float(time_limit))
    model.setRealParam("limits/gap", float(relative_gap))
    model.setIntParam("parallel/maxnthreads", threads)

    scip_variables: dict[str, list] = {}
    flat_variables = []
    for declaration in spec.variables:
        variable_type = {"binary": "B", "integer": "I", "real": "C"}[declaration.kind]
        values = [
            model.addVar(
                name=declaration.name if declaration.size == 1 else f"{declaration.name}_{index}",
                vtype=variable_type,
                lb=declaration.lower,
                ub=declaration.upper,
            )
            for index in range(declaration.size)
        ]
        scip_variables[declaration.name] = values
        flat_variables.extend(values)

    objective = spec.objectives[0]
    objective_expression = _compile_scip_expression(
        objective.expression,
        scip_variables,
        pyscipopt,
        quicksum,
    )
    set_nonlinear_objective(
        model,
        objective_expression,
        "minimize" if objective.direction == "min" else "maximize",
    )
    for declaration in spec.constraints:
        expression = _compile_scip_expression(
            declaration.expression,
            scip_variables,
            pyscipopt,
            quicksum,
        )
        if declaration.sense == "<=":
            constraint = expression <= declaration.rhs
        elif declaration.sense == ">=":
            constraint = expression >= declaration.rhs
        else:
            constraint = expression == declaration.rhs
        model.addCons(constraint, name=declaration.name)

    from qqa.tex.compiler import problem_from_spec

    problem = problem_from_spec(spec)
    defaults: dict[str, Any] = {
        "sol_size": max(64, 2 * max_warm_starts),
        "num_epochs": 1000,
        "verbose": verbose,
        "return_population": True,
    }
    if qqa_kwargs:
        defaults.update(qqa_kwargs)
    defaults["return_population"] = True
    qqa_result = problem.solve(**defaults)
    starts = _candidate_starts(problem, qqa_result, max_warm_starts)

    accepted = 0
    for start in starts:
        solution = model.createSol()
        for variable, value in zip(flat_variables, start.tolist(), strict=True):
            model.setSolVal(solution, variable, float(value))
        if model.addSol(solution):
            accepted += 1

    scip_started = perf_counter()
    model.optimize()
    scip_runtime = perf_counter() - scip_started
    status = str(model.getStatus())
    best = model.getBestSol()

    qqa_sol = qqa_result.best_sol.detach().cpu().to(dtype=problem.dtype)
    best_sol = qqa_sol
    if best is not None:
        scip_sol = torch.tensor(
            [model.getSolVal(best, variable) for variable in flat_variables],
            dtype=problem.dtype,
        )
        scip_sol = problem.space.project(problem.space.encode(scip_sol))
        # Projection restores integer domains after SCIP's floating-point
        # result extraction, but can perturb a tight nonlinear constraint.
        # Never replace a feasible incumbent with a projected infeasible point.
        with torch.no_grad():
            qqa_solver_objective = float(problem.objective_values(qqa_sol)[0].item())
            scip_solver_objective = float(problem.objective_values(scip_sol)[0].item())
        qqa_feasible = _exactly_feasible(problem, qqa_sol)
        scip_feasible = _exactly_feasible(problem, scip_sol)
        if scip_feasible and (
            not qqa_feasible or scip_solver_objective <= qqa_solver_objective + 1e-8
        ):
            best_sol = scip_sol
    solver_loss = float(problem.loss_fn(best_sol.unsqueeze(0))[0].item())
    score = problem.score_summary(best_sol)
    objective_value = float(score["value"])

    try:
        dual_bound = float(model.getDualbound())
        if not math.isfinite(dual_bound):
            dual_bound = None
    except Exception:  # pragma: no cover - status dependent
        dual_bound = None
    try:
        gap = float(model.getGap())
        if not math.isfinite(gap):
            gap = None
    except Exception:  # pragma: no cover - status dependent
        gap = None

    return SCIPModelResult(
        best_sol=best_sol,
        best_obj=solver_loss,
        objective_value=objective_value,
        solver_loss=solver_loss,
        runtime=perf_counter() - started,
        qqa_result=qqa_result,
        scip_runtime=scip_runtime,
        scip_status=status,
        dual_bound=dual_bound,
        gap=gap,
        n_warm_starts=accepted,
        score=score,
    )


__all__ = [
    "SCIPExpressionError",
    "SCIPModelResult",
    "solve_spec_scip",
]
