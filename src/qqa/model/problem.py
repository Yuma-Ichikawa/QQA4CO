"""Executable QQA adapter for canonical ModelIR models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from qqa.mixed.relaxation import MixedRelaxation
from qqa.mixed.variables import BinaryVariable, IntegerVariable, RealVariable, VariableSpace
from qqa.model import (
    BlackBoxFactor,
    ModelIR,
    ObjectiveSense,
    PairwisePottsFactor,
    VariableDomain,
)
from qqa.model.capabilities import require_qqa_capabilities
from qqa.problems.base import COProblem
from qqa.relaxation import (
    BinaryRelaxation,
    SinkhornRelaxation,
    SoftmaxCategoricalRelaxation,
    SpinRelaxation,
)


def _bound_vector(
    value: torch.Tensor | float | None,
    *,
    size: int,
    block: str,
    side: str,
) -> torch.Tensor:
    if value is None:
        raise ValueError(
            f"Pure QQA requires an explicit finite {side} bound for variable block {block!r}."
        )
    flat = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    if flat.numel() not in {1, size}:
        raise ValueError(f"{side} bound for variable block {block!r} does not match its size.")
    flat = flat.expand(size)
    if not torch.isfinite(flat).all():
        raise ValueError(
            f"Pure QQA requires a finite {side} bound for variable block {block!r}; "
            "no implicit [-10, 10] replacement is performed."
        )
    return flat


def _bound_runs(lower: torch.Tensor, upper: torch.Tensor):
    """Yield contiguous equal-bound runs without changing column order."""
    start = 0
    for stop in range(1, len(lower) + 1):
        boundary = stop == len(lower) or (
            lower[stop] != lower[start] or upper[stop] != upper[start]
        )
        if boundary:
            yield start, stop, float(lower[start]), float(upper[start])
            start = stop


@dataclass(frozen=True, slots=True)
class _FixedVariable:
    """Internal zero-width-domain coordinate retained when all values are fixed."""

    name: str
    lower: float | int
    upper: float | int
    size: int
    kind: str


class ModelIRProblem(COProblem):
    """Evaluate a canonical model with automatic scaled constraint penalties."""

    def __init__(self, model: ModelIR, *, dtype: torch.dtype = torch.float32) -> None:
        require_qqa_capabilities(model)
        self.model_ir = model
        self.name = model.metadata.name
        self.num_nodes = model.num_variables
        self.num_vars = model.num_variables
        self.dtype = dtype
        self.constraints = model.constraints
        domains = {block.domain for block in model.variables}
        structured = model.structured_block
        if structured is not None:
            unsupported = {
                type(factor).__name__
                for expression in (model.objective, *(row.expression for row in model.constraints))
                for factor in expression.factors
                if not isinstance(factor, (PairwisePottsFactor, BlackBoxFactor))
            }
            if unsupported:
                raise NotImplementedError(
                    "Structured ModelIR execution supports PairwisePottsFactor and "
                    f"BlackBoxFactor; unsupported factors: {sorted(unsupported)}."
                )
            self.num_node = structured.size
            self.num_category = int(structured.categories or 0)
            self.relaxation = (
                SinkhornRelaxation()
                if structured.domain is VariableDomain.PERMUTATION
                else SoftmaxCategoricalRelaxation()
            )
            self.space = None
        elif domains == {VariableDomain.BINARY}:
            self.relaxation = BinaryRelaxation(
                shape_fn=lambda population, _problem: (population, model.num_variables)
            )
            self.space = None
        elif domains == {VariableDomain.SPIN}:
            self.relaxation = SpinRelaxation()
            self.space = None
        else:
            variables: list[Any] = []
            for index, block in enumerate(model.variables):
                name = f"block_{index}"
                if block.domain is VariableDomain.BINARY:
                    variables.append(BinaryVariable(name, block.size))
                elif block.domain is VariableDomain.INTEGER:
                    lower = _bound_vector(
                        block.lower, size=block.size, block=block.name, side="lower"
                    ).ceil()
                    upper = _bound_vector(
                        block.upper, size=block.size, block=block.name, side="upper"
                    ).floor()
                    if torch.any(lower > upper):
                        raise ValueError(
                            f"Integer bounds for variable block {block.name!r} contain an empty domain."
                        )
                    for run, (start, stop, lo, hi) in enumerate(_bound_runs(lower, upper)):
                        variable_name = f"{name}_{run}"
                        variables.append(
                            _FixedVariable(variable_name, int(lo), int(hi), stop - start, "integer")
                            if lo == hi
                            else IntegerVariable(variable_name, int(lo), int(hi), stop - start)
                        )
                elif block.domain is VariableDomain.REAL:
                    lower = _bound_vector(
                        block.lower, size=block.size, block=block.name, side="lower"
                    )
                    upper = _bound_vector(
                        block.upper, size=block.size, block=block.name, side="upper"
                    )
                    for run, (start, stop, lo, hi) in enumerate(_bound_runs(lower, upper)):
                        variable_name = f"{name}_{run}"
                        variables.append(
                            _FixedVariable(variable_name, lo, hi, stop - start, "real")
                            if lo == hi
                            else RealVariable(variable_name, lo, hi, stop - start)
                        )
                else:
                    raise NotImplementedError(
                        "Categorical/permutation ModelIR execution requires a native "
                        "categorical adapter; use the existing structured problem class."
                    )
            self.space = VariableSpace(tuple(variables))
            self.relaxation = MixedRelaxation(self.space)
        self.penalty_multiplier = 1.0
        self._augmented_lagrangian: Any | None = None

    def objective_values(self, values: torch.Tensor) -> torch.Tensor:
        return self.model_ir.objective_values(values)

    def internal_objective(self, values: torch.Tensor) -> torch.Tensor:
        return self.model_ir.internal_energy(values)

    def ranking_objective(self, values: torch.Tensor) -> torch.Tensor:
        """Canonical minimisation objective used only for incumbent ranking."""
        return self.internal_objective(values)

    def incumbent_keys(self, values: torch.Tensor) -> torch.Tensor:
        """Return device-resident feasibility-first lexicographic keys."""
        values = self.model_ir._validate_values(values)
        domain = self.model_ir.domain_violations(values)
        residuals = [domain]
        feasible = domain <= 1e-6
        violations = self.constraint_violations(values)
        for row in self.constraints:
            violation = violations[row.name]
            residuals.append(violation / row.scale)
            feasible &= violation <= row.tolerance
        matrix = torch.stack(residuals, dim=1)
        maximum = matrix.amax(dim=1)
        total = matrix.sum(dim=1)
        internal = self.internal_objective(values)
        zero = torch.zeros_like(maximum)
        return torch.stack(
            (
                (~feasible).to(internal.dtype),
                torch.where(feasible, zero, maximum),
                torch.where(feasible, zero, total),
                internal,
            ),
            dim=1,
        )

    def constraint_values(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.model_ir._validate_values(values)
        return {row.name: row.expression.evaluate(values) for row in self.constraints}

    def normalised_constraint_residuals(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.model_ir._validate_values(values)
        if not self.constraints:
            return (
                values.new_zeros((values.shape[0], 0)),
                torch.zeros(0, dtype=torch.bool, device=values.device),
            )
        residuals = [row.canonical_residual(values) / row.scale for row in self.constraints]
        equality = torch.tensor(
            [row.sense == "==" for row in self.constraints],
            dtype=torch.bool,
            device=values.device,
        )
        return torch.stack(residuals, dim=1), equality

    def constraint_violations(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model_ir.constraint_violations(values)

    def loss_fn(self, values: torch.Tensor) -> torch.Tensor:
        internal = self.internal_objective(values)
        if not self.model_ir.constraints:
            return internal
        controller = self._augmented_lagrangian
        if controller is not None:
            return internal + controller.penalty(self, self.model_ir._validate_values(values))
        penalty = torch.zeros_like(internal)
        for row in self.model_ir.constraints:
            penalty = penalty + row.weight * (row.violation(values) / row.scale).square()
        return internal + self.penalty_multiplier * penalty

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        structured = self.model_ir.structured_block is not None
        values = x_disc.unsqueeze(0) if x_disc.ndim == (2 if structured else 1) else x_disc
        objective = self.objective_values(values)[0]
        violations = self.constraint_violations(values)
        rows: dict[str, dict[str, Any]] = {}
        domain_violation = float(self.model_ir.domain_violations(values)[0].item())
        feasible = domain_violation <= 1e-6
        if not feasible:
            rows["variable_domains"] = {
                "violation": domain_violation,
                "scaled_violation": domain_violation,
                "tolerance": 1e-6,
                "feasible": False,
            }
        for constraint in self.model_ir.constraints:
            violation = float(violations[constraint.name][0].item())
            satisfied = violation <= constraint.tolerance
            feasible &= satisfied
            rows[constraint.name] = {
                "lhs": float(constraint.expression.evaluate(values)[0].item()),
                "sense": constraint.sense,
                "rhs": constraint.rhs,
                "violation": violation,
                "scaled_violation": violation / constraint.scale,
                "tolerance": constraint.tolerance,
                "search_weight": constraint.weight,
                "priority": constraint.priority,
                "feasible": satisfied,
            }
        return {
            "label": "objective",
            "value": float(objective.item()),
            "unit": "",
            "feasible": feasible,
            "extra": {
                "constraints": rows,
                "domain_violation": domain_violation,
                "sense": ObjectiveSense(self.model_ir.sense).value,
            },
        }

    def repair_solution(
        self, values: torch.Tensor, *, time_limit: float | None = None
    ) -> torch.Tensor:
        from qqa.repair import repair_model_ir

        return repair_model_ir(self.model_ir, values, time_limit=time_limit)


__all__ = ["ModelIRProblem"]
