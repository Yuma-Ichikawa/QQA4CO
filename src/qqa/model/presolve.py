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
    ConstraintIR,
    Factor,
    ModelIR,
    ObjectiveIR,
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

    def expression(source: ObjectiveIR) -> ObjectiveIR:
        if not eliminate:
            return source
        return ObjectiveIR(
            tuple(
                EmbeddedFactor(
                    factor,
                    model.num_variables,
                    active_indices,
                    fixed_indices,
                    fixed_values,
                )
                for factor in source.factors
            ),
            source.constant,
        )

    constraints = []
    empty = 0
    duplicate = 0
    rescaled = 0
    seen = set()
    for row in model.constraints:
        if not row.expression.factors:
            if not _constant_constraint_satisfied(row):
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
        constraints.append(replace(row, expression=expression(row.expression), scale=scale))

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
