"""Conservative, reversible presolve passes over canonical ModelIR."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from qqa.model.bounds import (
    BoundTighteningInfeasibleError,
    tighten_singleton_bounds,
)
from qqa.model.ir import (
    CardinalityFactor,
    ClauseFactor,
    ConstraintIR,
    Factor,
    HigherOrderFactor,
    LinearFactor,
    ModelIR,
    ObjectiveIR,
    QuadraticEdgeFactor,
    VariableBlock,
    VariableDomain,
)


class PresolveInfeasibleError(ValueError):
    """Raised when a constant constraint proves the model infeasible."""


@dataclass(frozen=True, slots=True)
class EmbeddedFactor:
    """Evaluate an original factor after injecting eliminated fixed values."""

    factor: Factor
    original_size: int
    active_indices: torch.Tensor
    fixed_indices: torch.Tensor
    fixed_values: torch.Tensor

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(
            (*values.shape[:-1], self.original_size),
            dtype=values.dtype,
            device=values.device,
        )
        full[..., self.active_indices.to(values.device)] = values
        full[..., self.fixed_indices.to(values.device)] = self.fixed_values.to(values)
        return self.factor.evaluate(full)


@dataclass(frozen=True, slots=True)
class PresolveReport:
    original_variables: int
    reduced_variables: int
    fixed_variables: int
    removed_empty_constraints: int
    removed_duplicate_constraints: int
    rescaled_constraints: int
    tightened_bounds: int


@dataclass(frozen=True, slots=True)
class PresolveResult:
    original: ModelIR
    model: ModelIR
    active_indices: torch.Tensor
    fixed_indices: torch.Tensor
    fixed_values: torch.Tensor
    report: PresolveReport

    def reduce(self, values: torch.Tensor) -> torch.Tensor:
        """Project an original-space point into reduced variable order."""
        if values.shape[-1] != self.original.num_variables:
            raise ValueError("Original solution does not match the model variable count.")
        if len(self.fixed_indices):
            expected = self.fixed_values.to(values)
            actual = values[..., self.fixed_indices.to(values.device)]
            if not torch.allclose(actual, expected.expand_as(actual), atol=1e-6, rtol=0.0):
                raise ValueError("Initial solution disagrees with a presolved fixed variable.")
        return values[..., self.active_indices.to(values.device)]

    def restore(self, values: torch.Tensor) -> torch.Tensor:
        unbatched = values.ndim == 1
        reduced = values.unsqueeze(0) if unbatched else values
        if reduced.shape[-1] != len(self.active_indices):
            raise ValueError("Reduced solution does not match presolved variable count.")
        full = torch.zeros(
            (*reduced.shape[:-1], self.original.num_variables),
            dtype=reduced.dtype,
            device=reduced.device,
        )
        full[..., self.active_indices.to(reduced.device)] = reduced
        full[..., self.fixed_indices.to(reduced.device)] = self.fixed_values.to(reduced)
        return full[0] if unbatched else full


def _factor_scale(factor: Factor) -> float:
    values = []
    for name in ("weights", "table", "outputs", "demands", "supplies"):
        tensor = getattr(factor, name, None)
        if torch.is_tensor(tensor) and tensor.numel():
            values.append(float(tensor.detach().abs().max().item()))
    for name in ("weight", "capacity", "target"):
        value = getattr(factor, name, None)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(abs(float(value)))
    return max(values, default=1.0)


def _constant_constraint_satisfied(row: ConstraintIR) -> bool:
    residual = row.expression.constant - row.rhs
    if row.sense == "<=":
        return residual <= row.tolerance
    if row.sense == ">=":
        return residual >= -row.tolerance
    return abs(residual) <= row.tolerance


def _fixed_coordinates(model: ModelIR) -> tuple[torch.Tensor, torch.Tensor]:
    fixed_indices: list[int] = []
    fixed_values: list[float] = []
    offset = 0
    supported = {
        VariableDomain.BINARY,
        VariableDomain.SPIN,
        VariableDomain.INTEGER,
        VariableDomain.REAL,
    }
    for block in model.variables:
        if block.domain not in supported or block.lower is None or block.upper is None:
            offset += block.size
            continue
        lower = torch.as_tensor(block.lower).reshape(-1).expand(block.size)
        upper = torch.as_tensor(block.upper).reshape(-1).expand(block.size)
        local = torch.nonzero(
            torch.isfinite(lower) & torch.isfinite(upper) & (lower == upper), as_tuple=False
        ).reshape(-1)
        fixed_indices.extend((local + offset).tolist())
        fixed_values.extend(lower[local].tolist())
        offset += block.size
    return (
        torch.as_tensor(fixed_indices, dtype=torch.long),
        torch.as_tensor(fixed_values, dtype=torch.float64),
    )


def _structurally_reduce_factor(
    factor: Factor,
    *,
    remap: torch.Tensor,
    fixed_by_index: torch.Tensor,
) -> tuple[list[Factor], float] | None:
    """Reduce built-in sparse factors without rebuilding full vectors.

    ``None`` delegates uncommon/native factors to :class:`EmbeddedFactor`,
    preserving correctness while common MIP/QUBO factors retain sparse GPU
    execution after fixed-variable elimination.
    """
    if isinstance(factor, LinearFactor):
        linear_factor_indices = factor.indices.cpu()
        linear_factor_weights = factor.weights.cpu()
        linear_mapped = remap[linear_factor_indices]
        linear_active = linear_mapped >= 0
        constant = float(
            (
                linear_factor_weights[~linear_active]
                * fixed_by_index[linear_factor_indices[~linear_active]]
            )
            .sum()
            .item()
        )
        factors: list[Factor] = []
        if linear_active.any():
            factors.append(
                LinearFactor(linear_mapped[linear_active], linear_factor_weights[linear_active])
            )
        return factors, constant

    if isinstance(factor, QuadraticEdgeFactor):
        source, target = factor.edge_index.cpu()
        quadratic_factor_weights = factor.weights.cpu()
        mapped_source, mapped_target = remap[source], remap[target]
        source_active, target_active = mapped_source >= 0, mapped_target >= 0
        both_active = source_active & target_active
        factors = []
        if both_active.any():
            factors.append(
                QuadraticEdgeFactor(
                    torch.stack((mapped_source[both_active], mapped_target[both_active])),
                    quadratic_factor_weights[both_active],
                )
            )
        one_active = source_active ^ target_active
        if one_active.any():
            quadratic_active_indices = torch.where(
                source_active[one_active], mapped_source[one_active], mapped_target[one_active]
            )
            quadratic_fixed_indices = torch.where(
                source_active[one_active], target[one_active], source[one_active]
            )
            factors.append(
                LinearFactor(
                    quadratic_active_indices,
                    quadratic_factor_weights[one_active] * fixed_by_index[quadratic_fixed_indices],
                )
            )
        neither_active = ~source_active & ~target_active
        constant = float(
            (
                quadratic_factor_weights[neither_active]
                * fixed_by_index[source[neither_active]]
                * fixed_by_index[target[neither_active]]
            )
            .sum()
            .item()
        )
        return factors, constant

    if isinstance(factor, HigherOrderFactor):
        linear_indices: list[int] = []
        linear_weights: list[float] = []
        quadratic_indices: list[list[int]] = []
        quadratic_weights: list[float] = []
        higher: dict[int, tuple[list[list[int]], list[float]]] = {}
        constant = 0.0
        for higher_indices, higher_weight in zip(
            factor.indices.cpu().tolist(), factor.weights.cpu().tolist(), strict=True
        ):
            remaining_indices = [int(remap[index]) for index in higher_indices if remap[index] >= 0]
            coefficient = float(higher_weight)
            for index in higher_indices:
                if remap[index] < 0:
                    coefficient *= float(fixed_by_index[index])
            if coefficient == 0.0:
                continue
            if not remaining_indices:
                constant += coefficient
            elif len(remaining_indices) == 1:
                linear_indices.append(remaining_indices[0])
                linear_weights.append(coefficient)
            elif len(remaining_indices) == 2:
                quadratic_indices.append(remaining_indices)
                quadratic_weights.append(coefficient)
            else:
                grouped_rows, grouped_weights = higher.setdefault(len(remaining_indices), ([], []))
                grouped_rows.append(remaining_indices)
                grouped_weights.append(coefficient)
        factors = []
        if linear_indices:
            factors.append(LinearFactor(torch.tensor(linear_indices), torch.tensor(linear_weights)))
        if quadratic_indices:
            factors.append(
                QuadraticEdgeFactor(
                    torch.tensor(quadratic_indices).T.contiguous(),
                    torch.tensor(quadratic_weights),
                )
            )
        for grouped_rows, grouped_weights in higher.values():
            factors.append(
                HigherOrderFactor(torch.tensor(grouped_rows), torch.tensor(grouped_weights))
            )
        return factors, constant

    if isinstance(factor, CardinalityFactor):
        cardinality_indices = factor.indices.cpu()
        cardinality_mapped = remap[cardinality_indices]
        cardinality_active = cardinality_mapped >= 0
        fixed_sum = float(fixed_by_index[cardinality_indices[~cardinality_active]].sum().item())
        if not cardinality_active.any():
            return [], float(factor.weight * (fixed_sum - factor.target) ** 2)
        return [
            CardinalityFactor(
                cardinality_mapped[cardinality_active],
                factor.target - fixed_sum,
                factor.weight,
            )
        ], 0.0

    if isinstance(factor, ClauseFactor):
        grouped: dict[int, tuple[list[list[int]], list[list[int]], list[float]]] = {}
        constant = 0.0
        assert factor.weights is not None
        for clause_indices, clause_signs, clause_weight in zip(
            factor.indices.cpu().tolist(),
            factor.signs.cpu().tolist(),
            factor.weights.cpu().tolist(),
            strict=True,
        ):
            clause_active_indices: list[int] = []
            clause_active_signs: list[int] = []
            satisfied = False
            for index, sign in zip(clause_indices, clause_signs, strict=True):
                mapped_index = int(remap[index])
                if mapped_index >= 0:
                    clause_active_indices.append(mapped_index)
                    clause_active_signs.append(sign)
                    continue
                value = float(fixed_by_index[index])
                satisfied |= (sign > 0 and value == 1.0) or (sign < 0 and value == 0.0)
            if satisfied:
                continue
            if not clause_active_indices:
                constant += float(clause_weight)
                continue
            clause_rows, grouped_signs, grouped_weights = grouped.setdefault(
                len(clause_active_indices), ([], [], [])
            )
            clause_rows.append(clause_active_indices)
            grouped_signs.append(clause_active_signs)
            grouped_weights.append(float(clause_weight))
        factors = [
            ClauseFactor(
                torch.tensor(clause_rows),
                torch.tensor(grouped_signs),
                torch.tensor(grouped_weights),
            )
            for clause_rows, grouped_signs, grouped_weights in grouped.values()
        ]
        return factors, constant

    return None


def presolve_model(model: ModelIR, *, auto_scale: bool = True) -> PresolveResult:
    """Apply safe reductions and return an explicit original-space decoder."""
    if not isinstance(model, ModelIR):
        raise TypeError("model must be a ModelIR.")
    original = model
    try:
        tightening = tighten_singleton_bounds(model)
    except BoundTighteningInfeasibleError as exc:
        raise PresolveInfeasibleError(str(exc)) from exc
    model = tightening.model
    tightened_bounds = tightening.tightened_bounds
    fixed_indices, fixed_values = _fixed_coordinates(model)
    active_mask = torch.ones(model.num_variables, dtype=torch.bool)
    active_mask[fixed_indices] = False
    active_indices = torch.nonzero(active_mask, as_tuple=False).reshape(-1)
    # Preserve the original model if every variable is fixed because ModelIR
    # deliberately requires at least one block; constant evaluation remains exact.
    eliminate = bool(len(fixed_indices) and len(active_indices))

    variables: list[VariableBlock] = []
    offset = 0
    if eliminate:
        for block in model.variables:
            local_active = active_mask[offset : offset + block.size]
            count = int(local_active.sum().item())
            if count:
                lower = None
                upper = None
                if block.lower is not None:
                    lower = (
                        torch.as_tensor(block.lower).reshape(-1).expand(block.size)[local_active]
                    )
                if block.upper is not None:
                    upper = (
                        torch.as_tensor(block.upper).reshape(-1).expand(block.size)[local_active]
                    )
                variables.append(
                    VariableBlock(
                        block.name,
                        block.domain,
                        (count,),
                        lower,
                        upper,
                        block.categories,
                    )
                )
            offset += block.size
    else:
        variables = list(model.variables)

    remap = torch.full((model.num_variables,), -1, dtype=torch.long)
    remap[active_indices] = torch.arange(len(active_indices))
    fixed_by_index = torch.zeros(model.num_variables, dtype=torch.float64)
    fixed_by_index[fixed_indices] = fixed_values

    def expression(source: ObjectiveIR) -> ObjectiveIR:
        if not eliminate:
            return source
        factors: list[Factor] = []
        constant = float(source.constant)
        for factor in source.factors:
            reduced_factor = _structurally_reduce_factor(
                factor,
                remap=remap,
                fixed_by_index=fixed_by_index,
            )
            if reduced_factor is None:
                factors.append(
                    EmbeddedFactor(
                        factor,
                        model.num_variables,
                        active_indices,
                        fixed_indices,
                        fixed_values,
                    )
                )
                continue
            reduced_factors, fixed_constant = reduced_factor
            factors.extend(reduced_factors)
            constant += fixed_constant
        return ObjectiveIR(tuple(factors), constant)

    constraints = []
    empty = 0
    duplicate = 0
    rescaled = 0
    seen = set()
    for row in model.constraints:
        reduced_expression = expression(row.expression)
        if not reduced_expression.factors:
            reduced_row = replace(row, expression=reduced_expression)
            if not _constant_constraint_satisfied(reduced_row):
                raise PresolveInfeasibleError(
                    f"Constant constraint {row.name!r} proves the model infeasible."
                )
            empty += 1
            continue
        # Only remove rows that share the exact immutable expression object.
        # Tensor reprs truncate large values and can create false matches,
        # which would make a conservative presolve unsound.
        signature = (row.sense, row.rhs, id(row.expression))
        if signature in seen:
            duplicate += 1
            continue
        seen.add(signature)
        scale = row.scale
        if auto_scale:
            inferred = max(
                1.0,
                abs(row.rhs),
                abs(row.expression.constant),
                *(_factor_scale(factor) for factor in row.expression.factors),
            )
            if inferred > scale:
                scale = inferred
                rescaled += 1
        constraints.append(replace(row, expression=reduced_expression, scale=scale))

    reduced = ModelIR(
        tuple(variables),
        expression(model.objective),
        tuple(constraints),
        model.sense,
        model.metadata,
        model.transformations,
    )
    operations = []
    if eliminate:
        operations.append(("fixed-variable-elimination", {"count": len(fixed_indices)}))
    if empty:
        operations.append(("empty-constraint-removal", {"count": empty}))
    if duplicate:
        operations.append(("duplicate-constraint-removal", {"count": duplicate}))
    if rescaled:
        operations.append(("constraint-scaling", {"count": rescaled}))
    for operation, details in operations:
        reduced = reduced.transformed(operation, **details)
    report = PresolveReport(
        original.num_variables,
        reduced.num_variables,
        len(fixed_indices) if eliminate else 0,
        empty,
        duplicate,
        rescaled,
        tightened_bounds,
    )
    return PresolveResult(
        original,
        reduced,
        active_indices if eliminate else torch.arange(model.num_variables),
        fixed_indices if eliminate else torch.empty(0, dtype=torch.long),
        fixed_values if eliminate else torch.empty(0, dtype=torch.float64),
        report,
    )


__all__ = [
    "EmbeddedFactor",
    "PresolveInfeasibleError",
    "PresolveReport",
    "PresolveResult",
    "presolve_model",
]
