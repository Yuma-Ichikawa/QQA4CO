"""Proof-aware CP-SAT lowering for bounded integral :class:`ModelIR` models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import torch

from qqa.model.ir import (
    AllDifferentFactor,
    AssignmentFactor,
    CardinalityFactor,
    ClauseFactor,
    LinearFactor,
    ModelIR,
    ObjectiveSense,
    VariableDomain,
)
from qqa.model.native import CumulativeResourceFactor, NoOverlapFactor, PrecedenceFactor


@dataclass(slots=True)
class CPResult:
    best_sol: torch.Tensor | None
    best_obj: float | None
    runtime: float
    status: str
    dual_bound: float | None
    gap: float | None
    proven: bool
    backend: str = "ortools-cpsat"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    final_population: torch.Tensor | None = None

    @property
    def scip_status(self) -> str:
        """Compatibility status consumed by the backend-neutral result adapter."""
        return self.status


def _integer(value: float, label: str) -> int:
    rounded = int(round(float(value)))
    if not math.isfinite(float(value)) or abs(float(value) - rounded) > 1e-9:
        raise ValueError(f"CP-SAT requires integral {label}; got {value!r}.")
    return rounded


def _linear_expression(expression, variables):
    terms: list[Any] = []
    for factor in expression.factors:
        if not isinstance(factor, LinearFactor):
            return None
        terms.extend(
            _integer(weight, "linear coefficient") * variables[int(index)]
            for index, weight in zip(factor.indices.tolist(), factor.weights.tolist(), strict=True)
        )
    return sum(terms, _integer(expression.constant, "expression constant"))


def _enforce_zero_penalty(cp_model, factor, variables, variable_bounds) -> None:
    if isinstance(factor, CardinalityFactor):
        if abs(factor.weight) <= 0:
            return
        cp_model.add(
            sum(variables[int(index)] for index in factor.indices.tolist())
            == _integer(factor.target, "cardinality target")
        )
        return
    if isinstance(factor, AllDifferentFactor):
        cp_model.add_all_different([variables[int(index)] for index in factor.indices.tolist()])
        return
    if isinstance(factor, AssignmentFactor):
        indices = factor.indices.tolist()
        for row in indices:
            cp_model.add_exactly_one([variables[int(index)] for index in row])
        for column in zip(*indices, strict=True):
            cp_model.add_exactly_one([variables[int(index)] for index in column])
        return
    if isinstance(factor, ClauseFactor):
        for indices, signs in zip(factor.indices.tolist(), factor.signs.tolist(), strict=True):
            literals = [
                variables[int(index)] if int(sign) > 0 else variables[int(index)].not_()
                for index, sign in zip(indices, signs, strict=True)
            ]
            cp_model.add_bool_or(literals)
        return
    if isinstance(factor, PrecedenceFactor):
        for before, after, duration in zip(
            factor.before.tolist(), factor.after.tolist(), factor.durations.tolist(), strict=True
        ):
            cp_model.add(
                variables[int(before)] + _integer(duration, "duration") <= variables[int(after)]
            )
        return
    if isinstance(factor, NoOverlapFactor):
        intervals = []
        for position, (index, duration) in enumerate(
            zip(factor.starts.tolist(), factor.durations.tolist(), strict=True)
        ):
            length = _integer(duration, "duration")
            start = variables[int(index)]
            lower, upper = variable_bounds[int(index)]
            end = cp_model.new_int_var(
                lower + length,
                upper + length,
                f"end_{index}_{position}",
            )
            intervals.append(
                cp_model.new_interval_var(start, length, end, f"interval_{index}_{position}")
            )
        cp_model.add_no_overlap(intervals)
        return
    if isinstance(factor, CumulativeResourceFactor):
        intervals = []
        demands = []
        for position, (index, duration, demand) in enumerate(
            zip(
                factor.starts.tolist(),
                factor.durations.tolist(),
                factor.demands.tolist(),
                strict=True,
            )
        ):
            length = _integer(duration, "duration")
            start = variables[int(index)]
            lower, upper = variable_bounds[int(index)]
            end = cp_model.new_int_var(
                lower + length,
                upper + length,
                f"resource_end_{index}_{position}",
            )
            intervals.append(
                cp_model.new_interval_var(
                    start, length, end, f"resource_interval_{index}_{position}"
                )
            )
            demands.append(_integer(demand, "resource demand"))
        cp_model.add_cumulative(intervals, demands, _integer(factor.capacity, "capacity"))
        return
    raise NotImplementedError(f"No CP-SAT lowering is registered for {type(factor).__name__}.")


def solve_cp_model_ir(
    model: ModelIR,
    *,
    time_limit: float | None = None,
    random_seed: int = 0,
    workers: int = 1,
) -> CPResult:
    """Solve bounded binary/integer linear and scheduling models with CP-SAT."""
    if not isinstance(model, ModelIR):
        raise TypeError("model must be a ModelIR.")
    if workers < 1 or random_seed < 0:
        raise ValueError("workers must be positive and random_seed non-negative.")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be finite and positive or None.")
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError("Install `qqa[cpsat]` to use the CP scheduling runtime.") from exc

    cp = cp_model.CpModel()
    variables: list[Any] = []
    variable_bounds: list[tuple[int, int]] = []
    for block in model.variables:
        if block.domain not in {VariableDomain.BINARY, VariableDomain.INTEGER}:
            raise NotImplementedError("CP-SAT lowering supports binary and integer blocks only.")
        lower: torch.Tensor | None
        upper: torch.Tensor | None
        if block.domain is VariableDomain.BINARY:
            lower = (
                torch.zeros(block.size)
                if block.lower is None
                else torch.as_tensor(block.lower).reshape(-1).expand(block.size)
            )
            upper = (
                torch.ones(block.size)
                if block.upper is None
                else torch.as_tensor(block.upper).reshape(-1).expand(block.size)
            )
        else:
            lower = (
                None
                if block.lower is None
                else torch.as_tensor(block.lower).reshape(-1).expand(block.size)
            )
            upper = (
                None
                if block.upper is None
                else torch.as_tensor(block.upper).reshape(-1).expand(block.size)
            )
        if (
            lower is None
            or upper is None
            or not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all()
        ):
            raise ValueError(f"CP-SAT requires finite bounds for variable block {block.name!r}.")
        for local, (lo, hi) in enumerate(zip(lower.tolist(), upper.tolist(), strict=True)):
            name = block.name if block.size == 1 else f"{block.name}[{local}]"
            if block.domain is VariableDomain.BINARY:
                if (
                    _integer(lo, "binary lower bound") != 0
                    or _integer(hi, "binary upper bound") != 1
                ):
                    raise ValueError("Binary CP-SAT variables must have bounds [0, 1].")
                variables.append(cp.new_bool_var(name))
                variable_bounds.append((0, 1))
            else:
                integer_lower = _integer(lo, "lower bound")
                integer_upper = _integer(hi, "upper bound")
                variables.append(cp.new_int_var(integer_lower, integer_upper, name))
                variable_bounds.append((integer_lower, integer_upper))

    objective = _linear_expression(model.objective, variables)
    if objective is None:
        raise NotImplementedError("CP-SAT objective lowering currently requires linear factors.")
    if ObjectiveSense(model.sense) is ObjectiveSense.MINIMIZE:
        cp.minimize(objective)
    else:
        cp.maximize(objective)

    for row in model.constraints:
        linear = _linear_expression(row.expression, variables)
        if linear is not None:
            rhs = _integer(row.rhs, f"constraint {row.name} rhs")
            if row.sense == "<=":
                cp.add(linear <= rhs)
            elif row.sense == ">=":
                cp.add(linear >= rhs)
            else:
                cp.add(linear == rhs)
            continue
        if row.sense != "<=" or abs(row.rhs) > row.tolerance or row.expression.constant != 0:
            raise NotImplementedError(
                "Native CP factors must be expressed as a non-negative penalty <= 0."
            )
        for factor in row.expression.factors:
            _enforce_zero_penalty(cp, factor, variables, variable_bounds)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(random_seed)
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = float(time_limit)
    started = perf_counter()
    status_code = solver.solve(cp)
    runtime = perf_counter() - started
    status_name = solver.status_name(status_code).lower()
    infeasible = status_code == cp_model.INFEASIBLE
    has_solution = status_code in {cp_model.OPTIMAL, cp_model.FEASIBLE}
    if infeasible:
        return CPResult(
            None,
            None,
            runtime,
            "infeasible_proven",
            None,
            None,
            True,
            diagnostics={"cp_status": status_name},
        )
    if not has_solution:
        raise RuntimeError(f"CP-SAT returned no incumbent ({status_name}).")
    solution = torch.tensor([solver.value(variable) for variable in variables], dtype=torch.float64)
    objective_value = float(model.objective_values(solution)[0].item())
    bound = float(solver.best_objective_bound)
    gap = abs(objective_value - bound) / max(1.0, abs(objective_value), abs(bound))
    proven = status_code == cp_model.OPTIMAL
    return CPResult(
        solution,
        objective_value,
        runtime,
        "optimal" if proven else "timelimit",
        bound,
        gap,
        proven,
        diagnostics={"cp_status": status_name, "branches": solver.num_branches},
    )


__all__ = ["CPResult", "solve_cp_model_ir"]
