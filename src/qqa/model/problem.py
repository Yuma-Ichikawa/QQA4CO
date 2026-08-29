"""Executable QQA adapter for canonical ModelIR models."""

from __future__ import annotations

import math
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


def _scalar_bound(value: torch.Tensor | float | None, *, block: str, side: str) -> float:
    if value is None:
        raise ValueError(
            f"Pure QQA requires an explicit finite {side} bound for variable block {block!r}."
        )
    flat = torch.as_tensor(value).reshape(-1)
    if flat.numel() != 1:
        raise ValueError("QQA execution currently requires uniform bounds per block.")
    if not torch.isfinite(flat).all():
        raise ValueError(
            f"Pure QQA requires a finite {side} bound for variable block {block!r}; "
            "no implicit [-10, 10] replacement is performed."
        )
    return float(flat.item())


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
                    lower = int(
                        math.ceil(_scalar_bound(block.lower, block=block.name, side="lower"))
                    )
                    upper = int(
                        math.floor(_scalar_bound(block.upper, block=block.name, side="upper"))
                    )
                    variables.append(IntegerVariable(name, lower, upper, block.size))
                elif block.domain is VariableDomain.REAL:
                    variables.append(
                        RealVariable(
                            name,
                            _scalar_bound(block.lower, block=block.name, side="lower"),
                            _scalar_bound(block.upper, block=block.name, side="upper"),
                            block.size,
                        )
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
        rows = {}
        feasible = True
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
                "sense": ObjectiveSense(self.model_ir.sense).value,
            },
        }

    def repair_solution(self, values: torch.Tensor) -> torch.Tensor:
        from qqa.repair import repair_model_ir

        return repair_model_ir(self.model_ir, values)


__all__ = ["ModelIRProblem"]
