"""Independent factor-component execution for sparse QUBOs."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from qqa.annealing import AnnealResult, anneal
from qqa.engines.qqa.sparse import SparseQUBOProblem
from qqa.utils import safe_score_summary


def anneal_components(problem: Any, **kwargs: Any) -> AnnealResult:
    """Solve disconnected QUBO components independently and decode exactly."""
    qubo = problem.sparse_qubo
    components = qubo.connected_components()
    if len(components) <= 1:
        return anneal(problem, **kwargs)
    started = perf_counter()
    best = torch.zeros(qubo.num_variables, device=qubo.linear.device, dtype=qubo.linear.dtype)
    keep_population = bool(kwargs.get("return_population", False))
    population = None
    total_budget = kwargs.get("time_limit")
    component_results = []
    for index, variables in enumerate(components):
        subproblem = SparseQUBOProblem(
            qubo.subqubo(variables, include_constant=index == 0),
            name=f"{getattr(problem, 'name', type(problem).__name__)}-component-{index}",
        )
        component_kwargs = dict(kwargs)
        initial = component_kwargs.get("initial_state")
        if initial is not None:
            component_kwargs["initial_state"] = initial[..., variables]
        if total_budget is not None:
            elapsed = perf_counter() - started
            remaining = max(1e-6, float(total_budget) - elapsed)
            component_kwargs["time_limit"] = remaining / (len(components) - index)
        result = anneal(subproblem, **component_kwargs)
        component_results.append(result)
        best[variables] = result.best_sol.to(best)
        if keep_population and result.final_population is not None:
            if population is None:
                population = torch.zeros(
                    (result.final_population.shape[0], qubo.num_variables),
                    device=best.device,
                    dtype=best.dtype,
                )
            population[:, variables] = result.final_population.to(population)
    objective = float(problem.loss_fn(best.unsqueeze(0))[0].item())
    return AnnealResult(
        best_sol=best,
        best_obj=objective,
        runtime=perf_counter() - started,
        score=safe_score_summary(problem, best, fallback_obj=objective),
        final_population=population,
        diagnostics={
            "component_decomposition": True,
            "components": len(components),
            "component_sizes": [int(values.numel()) for values in components],
            "component_runtimes": [result.runtime for result in component_results],
            "deadline_reached": any(
                bool(result.diagnostics.get("deadline_reached", False))
                for result in component_results
            ),
        },
    )


__all__ = ["anneal_components"]
