"""Execute a validated :class:`OptimizationPlan` through one public API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from qqa.blackbox import BlackBoxConstraint, BlackBoxProblem
from qqa.mixed.variables import VariableSpace
from qqa.natural_language.planner import OptimizationPlan, compile_natural_language, plan_spec
from qqa.tex.client import OpenAICompatibleClient
from qqa.tex.compiler import problem_from_spec
from qqa.tex.expressions import compile_expression
from qqa.tex.schema import ModelSpec


def blackbox_from_spec(spec: ModelSpec | dict) -> BlackBoxProblem:
    """Adapt one safe symbolic objective to an opaque point-evaluation API."""
    if isinstance(spec, dict):
        spec = ModelSpec.from_dict(spec)
    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a ModelSpec or dict.")
    if len(spec.objectives) != 1:
        raise ValueError("blackbox_from_spec requires exactly one objective.")
    variables = spec.variable_specs
    variable_map = {variable.name: variable for variable in variables}
    space = VariableSpace(variables)

    def scalar_adapter(expression: str):
        compiled = compile_expression(expression, variable_map)

        def evaluate(point: dict) -> float:
            packed = space.pack(point, dtype=torch.float64).unsqueeze(0)
            named = space.unpack(packed)
            with torch.no_grad():
                value = compiled(named)
            tensor = torch.as_tensor(value)
            if tensor.numel() != 1 or tensor.is_complex() or tensor.dtype == torch.bool:
                raise TypeError("Compiled black-box expressions must return one real scalar.")
            if not torch.isfinite(tensor).all():
                raise FloatingPointError("Compiled black-box expression returned NaN or infinity.")
            return float(tensor.item())

        return evaluate

    objective = spec.objectives[0]
    constraints = [
        BlackBoxConstraint(
            scalar_adapter(item.expression),
            sense=item.sense,
            rhs=item.rhs,
            tolerance=item.tolerance,
            scale=item.scale,
            name=item.name,
        )
        for item in spec.constraints
    ]
    problem = BlackBoxProblem(
        variables,
        scalar_adapter(objective.expression),
        constraints=constraints,
        direction=objective.direction,
        name=spec.name,
    )
    dynamic_problem: Any = problem
    dynamic_problem.model_spec = spec
    return problem


@dataclass(slots=True)
class AskResult:
    """Plan, compiled problem, and numerical result returned by :func:`ask`."""

    plan: OptimizationPlan
    problem: Any
    result: Any

    @property
    def solver(self) -> str:
        return self.plan.selected_solver


def execute_plan(
    plan: OptimizationPlan,
    *,
    device: str | torch.device = "auto",
    seed: int = 0,
    sol_size: int = 256,
    num_epochs: int = 1500,
    budget: int = 96,
    batch_size: int = 8,
    workers: int = 4,
    scip_time_limit: float = 60.0,
    scip_gap: float = 0.0,
    scip_threads: int = 1,
    scip_warm_starts: int = 32,
    verbose: bool = False,
    solver_kwargs: dict[str, Any] | None = None,
) -> AskResult:
    """Execute an already validated plan with workflow-aware defaults."""
    if not isinstance(plan, OptimizationPlan):
        raise TypeError("plan must be an OptimizationPlan.")
    from qqa.utils import fix_seed, resolve_device

    fix_seed(seed)
    resolved_device = resolve_device(device)
    extra = {} if solver_kwargs is None else dict(solver_kwargs)
    problem: Any
    if plan.selected_solver == "blackbox":
        problem = blackbox_from_spec(plan.spec)
        options = {
            "budget": budget,
            "batch_size": batch_size,
            "workers": workers,
            "device": resolved_device,
            "seed": seed,
            "verbose": verbose,
        }
        options.update(extra)
        result = problem.solve(**options)
    else:
        problem = problem_from_spec(plan.spec)
        if plan.selected_solver == "pareto":
            options = {
                "sol_size": sol_size,
                "num_epochs": num_epochs,
                "device": resolved_device,
                "seed": seed,
                "verbose": verbose,
            }
            options.update(extra)
            result = problem.solve_pareto(**options)
        elif plan.selected_solver == "qqa-scip":
            from qqa.hybrid import solve_spec_scip

            qqa_options = {
                "sol_size": sol_size,
                "num_epochs": num_epochs,
                "device": resolved_device,
                "verbose": verbose,
            }
            qqa_options.update(extra)
            result = solve_spec_scip(
                plan.spec,
                qqa_kwargs=qqa_options,
                time_limit=scip_time_limit,
                relative_gap=scip_gap,
                max_warm_starts=scip_warm_starts,
                threads=scip_threads,
                verbose=verbose,
            )
        else:
            options = {
                "sol_size": sol_size,
                "num_epochs": num_epochs,
                "device": resolved_device,
                "verbose": verbose,
            }
            options.update(extra)
            result = problem.solve(**options)
    return AskResult(plan=plan, problem=problem, result=result)


def ask(
    source: str | ModelSpec | dict | OptimizationPlan,
    *,
    client: OpenAICompatibleClient | None = None,
    solver: str = "auto",
    **kwargs: Any,
) -> AskResult:
    """Plan and solve a natural-language request through one safe entry point.

    Passing a :class:`ModelSpec`, dictionary, or pre-reviewed plan skips the
    external API. Plain text is translated by the configured compatible API,
    validated locally, routed deterministically, and then solved.
    """
    if isinstance(source, OptimizationPlan):
        plan = source
    elif isinstance(source, (ModelSpec, dict)):
        plan = plan_spec(source, solver=solver)
    else:
        plan = compile_natural_language(source, client=client, solver=solver)
    return execute_plan(plan, **kwargs)


__all__ = ["AskResult", "ask", "blackbox_from_spec", "execute_plan"]
