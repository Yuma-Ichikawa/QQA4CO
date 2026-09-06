"""Device-resident typed factor hypergraph with portable Torch kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from qqa.model.capabilities import inspect_capabilities
from qqa.model.ir import (
    CardinalityFactor,
    ClauseFactor,
    LinearFactor,
    ModelIR,
    ObjectiveIR,
    QuadraticEdgeFactor,
)

_TYPE_IDS = {"linear": 1, "quadratic": 2, "cardinality": 3, "clause": 4}


def segmented_sum(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Sum a ragged final dimension without host-side loops."""
    if values.shape[-1] != segment_ids.numel():
        raise ValueError("values and segment_ids do not align.")
    result = values.new_zeros((*values.shape[:-1], num_segments))
    shape = (1,) * (values.ndim - 1) + (segment_ids.numel(),)
    indices = segment_ids.to(values.device).reshape(shape).expand_as(values)
    return result.scatter_add_(-1, indices, values)


@dataclass(frozen=True, slots=True)
class CompiledFactorGraph:
    num_variables: int
    constant: float
    linear: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    cardinality_indices: torch.Tensor
    cardinality_segments: torch.Tensor
    cardinality_targets: torch.Tensor
    cardinality_weights: torch.Tensor
    clause_indices: torch.Tensor
    clause_signs: torch.Tensor
    clause_segments: torch.Tensor
    clause_weights: torch.Tensor
    factor_offsets: torch.Tensor
    factor_variables: torch.Tensor
    factor_type_ids: torch.Tensor
    constraints: tuple[CompiledFactorGraph, ...] = ()
    constraint_names: tuple[str, ...] = ()
    constraint_senses: tuple[str, ...] = ()
    constraint_rhs: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    constraint_scales: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    constraint_tolerances: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    constraint_weights: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    constraint_priorities: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> CompiledFactorGraph:
        integer_names = {
            "edge_index",
            "cardinality_indices",
            "cardinality_segments",
            "clause_indices",
            "clause_signs",
            "clause_segments",
            "factor_offsets",
            "factor_variables",
            "factor_type_ids",
        }
        values: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if torch.is_tensor(value):
                values[name] = value.to(
                    device=device, dtype=value.dtype if name in integer_names else dtype
                )
            elif name == "constraints":
                values[name] = tuple(item.to(device, dtype) for item in value)
            else:
                values[name] = value
        return CompiledFactorGraph(**values)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.num_variables:
            raise ValueError("values do not match the compiled variable dimension.")
        energy = (values * self.linear.to(values)).sum(dim=-1) + self.constant
        if self.edge_weight.numel():
            source, target = self.edge_index.to(values.device)
            energy = energy + (
                values[..., source] * values[..., target] * self.edge_weight.to(values)
            ).sum(dim=-1)
        if self.cardinality_targets.numel():
            selected = values[..., self.cardinality_indices.to(values.device)]
            totals = segmented_sum(
                selected,
                self.cardinality_segments,
                len(self.cardinality_targets),
            )
            energy = energy + (
                (totals - self.cardinality_targets.to(values)).square()
                * self.cardinality_weights.to(values)
            ).sum(dim=-1)
        if self.clause_weights.numel():
            selected = values[..., self.clause_indices.to(values.device)]
            literals = torch.where(self.clause_signs.to(values.device) > 0, selected, 1 - selected)
            clauses = values.new_ones((*values.shape[:-1], len(self.clause_weights)))
            shape = (1,) * (values.ndim - 1) + (self.clause_segments.numel(),)
            segments = self.clause_segments.to(values.device).reshape(shape).expand_as(literals)
            clauses.scatter_reduce_(-1, segments, 1 - literals, reduce="prod", include_self=True)
            energy = energy + (clauses * self.clause_weights.to(values)).sum(dim=-1)
        return energy

    def evaluate_constraints(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            name: factor.evaluate(values)
            for name, factor in zip(self.constraint_names, self.constraints, strict=True)
        }

    def constraint_violations(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate canonical non-negative, scaled row violations on-device."""
        lhs = self.evaluate_constraints(values)
        result = {}
        for index, (name, sense) in enumerate(
            zip(self.constraint_names, self.constraint_senses, strict=True)
        ):
            residual = lhs[name] - self.constraint_rhs[index].to(values)
            violation = (
                residual.clamp_min(0)
                if sense == "<="
                else (-residual).clamp_min(0)
                if sense == ">="
                else residual.abs()
            )
            result[name] = violation / self.constraint_scales[index].to(values)
        return result

    def constraint_penalty(self, values: torch.Tensor) -> torch.Tensor:
        """Return the declared weighted search penalty, separate from tolerance.

        ``constraint_tolerances`` are used only for mathematical feasibility;
        row scaling, search weights, and user priorities remain independent.
        """
        violations = self.constraint_violations(values)
        if not violations:
            return values.new_zeros(values.shape[:-1])
        matrix = torch.stack(
            [violations[name] for name in self.constraint_names],
            dim=-1,
        )
        return (matrix.square() * self.constraint_weights.to(values)).sum(dim=-1)


def _scope(factor: Any) -> torch.Tensor:
    for name in ("indices", "edge_index"):
        value = getattr(factor, name, None)
        if torch.is_tensor(value):
            return torch.unique(value.reshape(-1).to(torch.long), sorted=True)
    return torch.zeros(0, dtype=torch.long)


def _compile_expression(expression: ObjectiveIR, num_variables: int) -> CompiledFactorGraph:
    linear = torch.zeros(num_variables, dtype=torch.float64)
    edges: list[torch.Tensor] = []
    edge_weights: list[torch.Tensor] = []
    cardinality_indices: list[torch.Tensor] = []
    cardinality_segments: list[torch.Tensor] = []
    cardinality_targets = []
    cardinality_weights = []
    clause_indices: list[torch.Tensor] = []
    clause_signs: list[torch.Tensor] = []
    clause_segments: list[torch.Tensor] = []
    clause_weights: list[torch.Tensor] = []
    factor_variables: list[torch.Tensor] = []
    factor_offsets = [0]
    factor_type_ids = []
    cardinality_count = 0
    clause_count = 0
    for factor in expression.factors:
        scope = _scope(factor)
        factor_variables.append(scope)
        factor_offsets.append(factor_offsets[-1] + len(scope))
        if isinstance(factor, LinearFactor):
            linear.scatter_add_(0, factor.indices, factor.weights.to(torch.float64))
            factor_type_ids.append(_TYPE_IDS["linear"])
        elif isinstance(factor, QuadraticEdgeFactor):
            edges.append(factor.edge_index)
            edge_weights.append(factor.weights.to(torch.float64))
            factor_type_ids.append(_TYPE_IDS["quadratic"])
        elif isinstance(factor, CardinalityFactor):
            cardinality_indices.append(factor.indices)
            cardinality_segments.append(
                torch.full((len(factor.indices),), cardinality_count, dtype=torch.long)
            )
            cardinality_targets.append(factor.target)
            cardinality_weights.append(factor.weight)
            cardinality_count += 1
            factor_type_ids.append(_TYPE_IDS["cardinality"])
        elif isinstance(factor, ClauseFactor):
            flat_indices = factor.indices.reshape(-1)
            flat_signs = factor.signs.reshape(-1)
            width = factor.indices.shape[1]
            count = factor.indices.shape[0]
            clause_indices.append(flat_indices)
            clause_signs.append(flat_signs)
            clause_segments.append(
                torch.arange(
                    clause_count, clause_count + count, dtype=torch.long
                ).repeat_interleave(width)
            )
            if factor.weights is None:
                raise RuntimeError("Clause factor weights were not initialized.")
            clause_weights.append(factor.weights.to(torch.float64))
            clause_count += count
            factor_type_ids.append(_TYPE_IDS["clause"])
        else:
            raise NotImplementedError(
                f"No compiled GPU kernel is registered for {type(factor).__name__}."
            )

    def concatenate(
        items: list[torch.Tensor], *, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.cat(items, dim=-1) if items else torch.empty(shape, dtype=dtype)

    return CompiledFactorGraph(
        num_variables,
        expression.constant,
        linear,
        concatenate(edges, shape=(2, 0), dtype=torch.long),
        concatenate(edge_weights, shape=(0,), dtype=torch.float64),
        concatenate(cardinality_indices, shape=(0,), dtype=torch.long),
        concatenate(cardinality_segments, shape=(0,), dtype=torch.long),
        torch.as_tensor(cardinality_targets, dtype=torch.float64),
        torch.as_tensor(cardinality_weights, dtype=torch.float64),
        concatenate(clause_indices, shape=(0,), dtype=torch.long),
        concatenate(clause_signs, shape=(0,), dtype=torch.int8),
        concatenate(clause_segments, shape=(0,), dtype=torch.long),
        concatenate(clause_weights, shape=(0,), dtype=torch.float64),
        torch.as_tensor(factor_offsets, dtype=torch.long),
        concatenate(factor_variables, shape=(0,), dtype=torch.long),
        torch.as_tensor(factor_type_ids, dtype=torch.int16),
    )


def compile_factor_graph(model: ModelIR) -> CompiledFactorGraph:
    """Lower supported typed factors and preserve constraint separation."""
    report = inspect_capabilities(model)
    unsupported = [
        record.factor_type for record in report.factors if "gpu_kernel" not in record.capabilities
    ]
    if unsupported:
        raise NotImplementedError(
            "GPU lowering is unavailable for: " + ", ".join(sorted(set(unsupported)))
        )
    objective = _compile_expression(model.objective, model.num_variables)
    constraints = tuple(
        _compile_expression(row.expression, model.num_variables) for row in model.constraints
    )
    values = {name: getattr(objective, name) for name in objective.__dataclass_fields__}
    values["constraints"] = constraints
    values["constraint_names"] = tuple(row.name for row in model.constraints)
    values["constraint_senses"] = tuple(row.sense for row in model.constraints)
    values["constraint_rhs"] = torch.tensor(
        [row.rhs for row in model.constraints], dtype=torch.float64
    )
    values["constraint_scales"] = torch.tensor(
        [row.scale for row in model.constraints], dtype=torch.float64
    )
    values["constraint_tolerances"] = torch.tensor(
        [row.tolerance for row in model.constraints], dtype=torch.float64
    )
    values["constraint_weights"] = torch.tensor(
        [row.weight for row in model.constraints], dtype=torch.float64
    )
    values["constraint_priorities"] = torch.tensor(
        [row.priority for row in model.constraints], dtype=torch.float64
    )
    return CompiledFactorGraph(**values)


__all__ = ["CompiledFactorGraph", "compile_factor_graph", "segmented_sum"]
