"""Safe singleton-row bound propagation for canonical ModelIR models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qqa.model.ir import LinearFactor, ModelIR, VariableBlock, VariableDomain


class BoundTighteningInfeasibleError(ValueError):
    """Raised when propagated lower and upper bounds no longer intersect."""


@dataclass(frozen=True, slots=True)
class BoundTighteningResult:
    model: ModelIR
    tightened_bounds: int


def _domain_bounds(block: VariableBlock) -> tuple[torch.Tensor, torch.Tensor]:
    if block.domain is VariableDomain.BINARY:
        default_lower, default_upper = 0.0, 1.0
    elif block.domain is VariableDomain.SPIN:
        default_lower, default_upper = -1.0, 1.0
    else:
        default_lower, default_upper = -torch.inf, torch.inf
    lower = (
        torch.full((block.size,), default_lower, dtype=torch.float64)
        if block.lower is None
        else torch.as_tensor(block.lower, dtype=torch.float64)
        .reshape(-1)
        .expand(block.size)
        .clone()
    )
    upper = (
        torch.full((block.size,), default_upper, dtype=torch.float64)
        if block.upper is None
        else torch.as_tensor(block.upper, dtype=torch.float64)
        .reshape(-1)
        .expand(block.size)
        .clone()
    )
    return lower, upper


def _singleton(row) -> tuple[int, float, float] | None:
    coefficients: dict[int, float] = {}
    for factor in row.expression.factors:
        if not isinstance(factor, LinearFactor):
            return None
        for index, weight in zip(
            factor.indices.detach().cpu().tolist(),
            factor.weights.detach().cpu().tolist(),
            strict=True,
        ):
            coefficients[index] = coefficients.get(index, 0.0) + float(weight)
    coefficients = {index: value for index, value in coefficients.items() if value != 0.0}
    if len(coefficients) != 1:
        return None
    index, coefficient = next(iter(coefficients.items()))
    return index, coefficient, float(row.expression.constant)


def tighten_singleton_bounds(model: ModelIR) -> BoundTighteningResult:
    """Propagate one-variable linear rows without changing the feasible set."""
    if not isinstance(model, ModelIR):
        raise TypeError("model must be a ModelIR.")
    lower_parts: list[torch.Tensor] = []
    upper_parts: list[torch.Tensor] = []
    domains: list[VariableDomain] = []
    for block in model.variables:
        lower, upper = _domain_bounds(block)
        lower_parts.append(lower)
        upper_parts.append(upper)
        domains.extend([block.domain] * block.size)
    lower = torch.cat(lower_parts)
    upper = torch.cat(upper_parts)
    original_lower = lower.clone()
    original_upper = upper.clone()

    for row in model.constraints:
        singleton = _singleton(row)
        if singleton is None:
            continue
        index, coefficient, constant = singleton
        if not 0 <= index < model.num_variables:
            raise ValueError(f"Constraint {row.name!r} references an out-of-range variable.")
        if coefficient == 0.0:
            continue
        limits: list[tuple[str, float]] = []
        if row.sense in {"<=", "=="}:
            limits.append(("<=", row.rhs + row.tolerance))
        if row.sense in {">=", "=="}:
            limits.append((">=", row.rhs - row.tolerance))
        for sense, rhs in limits:
            bound = (rhs - constant) / coefficient
            lower_bound = sense == ">="
            if coefficient < 0:
                lower_bound = not lower_bound
            if lower_bound:
                lower[index] = torch.maximum(lower[index], lower.new_tensor(bound))
            else:
                upper[index] = torch.minimum(upper[index], upper.new_tensor(bound))

    for index, domain in enumerate(domains):
        if domain in {VariableDomain.INTEGER, VariableDomain.BINARY}:
            lower[index] = torch.ceil(lower[index])
            upper[index] = torch.floor(upper[index])
        elif domain is VariableDomain.SPIN:
            lower[index] = -1.0 if lower[index] <= -1.0 else 1.0
            upper[index] = 1.0 if upper[index] >= 1.0 else -1.0
    infeasible = torch.nonzero(lower > upper, as_tuple=False).reshape(-1)
    if len(infeasible):
        raise BoundTighteningInfeasibleError(
            f"Singleton bound propagation proves variable {int(infeasible[0])} infeasible."
        )

    changed = int(((lower != original_lower) | (upper != original_upper)).sum().item())
    if not changed:
        return BoundTighteningResult(model, 0)

    variables: list[VariableBlock] = []
    offset = 0
    for block in model.variables:
        stop = offset + block.size
        variables.append(
            VariableBlock(
                block.name,
                block.domain,
                block.shape,
                lower[offset:stop].reshape(block.shape),
                upper[offset:stop].reshape(block.shape),
                block.categories,
            )
        )
        offset = stop
    tightened = ModelIR(
        tuple(variables),
        model.objective,
        model.constraints,
        model.sense,
        model.metadata,
        model.transformations,
    ).transformed("singleton-bound-tightening", count=changed)
    return BoundTighteningResult(tightened, changed)


__all__ = [
    "BoundTighteningInfeasibleError",
    "BoundTighteningResult",
    "tighten_singleton_bounds",
]
