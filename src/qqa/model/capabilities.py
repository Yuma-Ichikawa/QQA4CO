"""Machine-readable factor capability contracts.

The canonical IR deliberately represents more models than the differentiable
QQA engine can solve by itself.  This module keeps representation support,
primal-search support, and proof support separate so planners never infer one
from another or silently route a non-differentiable factor through QQA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch

from qqa.model.ir import ModelIR, VariableDomain


class FactorCapability(str, Enum):
    EVALUATE = "evaluate"
    DIFFERENTIABLE = "differentiable"
    SUBGRADIENT = "subgradient"
    PROX = "prox"
    GPU_KERNEL = "gpu_kernel"
    PROPAGATE = "propagate"
    SEPARATE = "separate"
    REPAIR = "repair"
    LOWER_BOUND = "lower_bound"
    EXACT_ENCODE = "exact_encode"
    PROOF_SAFE = "proof_safe"


_C = FactorCapability

# Capabilities are conservative.  In particular, factors that cast a relaxed
# value to an integer or compare values for equality are evaluable but are not
# advertised as differentiable even though PyTorch may return a zero gradient.
_BUILTIN_CAPABILITIES: dict[str, frozenset[FactorCapability]] = {
    "LinearFactor": frozenset(
        {
            _C.EVALUATE,
            _C.DIFFERENTIABLE,
            _C.PROX,
            _C.GPU_KERNEL,
            _C.LOWER_BOUND,
            _C.EXACT_ENCODE,
            _C.PROOF_SAFE,
        }
    ),
    "QuadraticEdgeFactor": frozenset(
        {
            _C.EVALUATE,
            _C.DIFFERENTIABLE,
            _C.GPU_KERNEL,
            _C.LOWER_BOUND,
            _C.EXACT_ENCODE,
            _C.PROOF_SAFE,
        }
    ),
    "HigherOrderFactor": frozenset(
        {_C.EVALUATE, _C.DIFFERENTIABLE, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "ClauseFactor": frozenset(
        {
            _C.EVALUATE,
            _C.DIFFERENTIABLE,
            _C.GPU_KERNEL,
            _C.PROPAGATE,
            _C.SEPARATE,
            _C.REPAIR,
            _C.LOWER_BOUND,
            _C.EXACT_ENCODE,
            _C.PROOF_SAFE,
        }
    ),
    "CardinalityFactor": frozenset(
        {
            _C.EVALUATE,
            _C.DIFFERENTIABLE,
            _C.PROX,
            _C.GPU_KERNEL,
            _C.PROPAGATE,
            _C.SEPARATE,
            _C.REPAIR,
            _C.LOWER_BOUND,
            _C.EXACT_ENCODE,
            _C.PROOF_SAFE,
        }
    ),
    "AssignmentFactor": frozenset(
        {
            _C.EVALUATE,
            _C.DIFFERENTIABLE,
            _C.PROX,
            _C.GPU_KERNEL,
            _C.PROPAGATE,
            _C.REPAIR,
            _C.LOWER_BOUND,
            _C.EXACT_ENCODE,
            _C.PROOF_SAFE,
        }
    ),
    "PairwisePottsFactor": frozenset(
        {_C.EVALUATE, _C.DIFFERENTIABLE, _C.GPU_KERNEL, _C.REPAIR, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "AllDifferentFactor": frozenset(
        {_C.EVALUATE, _C.PROPAGATE, _C.REPAIR, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "TableFactor": frozenset(
        {_C.EVALUATE, _C.PROPAGATE, _C.REPAIR, _C.LOWER_BOUND, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "BlackBoxFactor": frozenset({_C.EVALUATE}),
    "IndicatorFactor": frozenset(
        {_C.EVALUATE, _C.DIFFERENTIABLE, _C.PROPAGATE, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "SOS1Factor": frozenset(
        {_C.EVALUATE, _C.DIFFERENTIABLE, _C.PROPAGATE, _C.REPAIR, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "SOS2Factor": frozenset(
        {_C.EVALUATE, _C.DIFFERENTIABLE, _C.PROPAGATE, _C.REPAIR, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "PiecewiseLinearFactor": frozenset(
        {_C.EVALUATE, _C.SUBGRADIENT, _C.PROX, _C.LOWER_BOUND, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "LogicalFactor": frozenset(
        {_C.EVALUATE, _C.PROPAGATE, _C.REPAIR, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "PrecedenceFactor": frozenset(
        {
            _C.EVALUATE,
            _C.DIFFERENTIABLE,
            _C.PROPAGATE,
            _C.REPAIR,
            _C.LOWER_BOUND,
            _C.EXACT_ENCODE,
            _C.PROOF_SAFE,
        }
    ),
    "NoOverlapFactor": frozenset(
        {_C.EVALUATE, _C.SUBGRADIENT, _C.PROPAGATE, _C.REPAIR, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "CumulativeResourceFactor": frozenset(
        {_C.EVALUATE, _C.DIFFERENTIABLE, _C.PROPAGATE, _C.REPAIR, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "FlowConservationFactor": frozenset(
        {
            _C.EVALUATE,
            _C.DIFFERENTIABLE,
            _C.PROX,
            _C.PROPAGATE,
            _C.LOWER_BOUND,
            _C.EXACT_ENCODE,
            _C.PROOF_SAFE,
        }
    ),
    "MatchingFactor": frozenset(
        {
            _C.EVALUATE,
            _C.DIFFERENTIABLE,
            _C.PROX,
            _C.PROPAGATE,
            _C.REPAIR,
            _C.LOWER_BOUND,
            _C.EXACT_ENCODE,
            _C.PROOF_SAFE,
        }
    ),
    "SubtourEliminationFactor": frozenset(
        {_C.EVALUATE, _C.SEPARATE, _C.REPAIR, _C.EXACT_ENCODE, _C.PROOF_SAFE}
    ),
    "ScenarioFactor": frozenset({_C.EVALUATE, _C.DIFFERENTIABLE}),
    "ChanceConstraintFactor": frozenset({_C.EVALUATE, _C.DIFFERENTIABLE}),
    "DistributionallyRobustChanceFactor": frozenset({_C.EVALUATE, _C.DIFFERENTIABLE}),
    "MomentAmbiguityDROFactor": frozenset({_C.EVALUATE, _C.DIFFERENTIABLE}),
    "PhiDivergenceDROFactor": frozenset({_C.EVALUATE, _C.DIFFERENTIABLE}),
    "WassersteinDROFactor": frozenset({_C.EVALUATE, _C.DIFFERENTIABLE}),
}


def factor_capabilities(factor: Any) -> frozenset[FactorCapability]:
    """Return conservative capabilities for one factor instance.

    Third-party factors can expose a ``capabilities`` iterable containing enum
    values or their string forms.  Unknown factors remain representable when
    they implement ``evaluate`` but are not assumed differentiable or safe for
    an exact proof.
    """
    wrapped = getattr(factor, "factor", None)
    if wrapped is not None and callable(getattr(wrapped, "evaluate", None)):
        return factor_capabilities(wrapped)
    declared = getattr(factor, "capabilities", None)
    if declared is not None:
        return frozenset(FactorCapability(item) for item in declared)
    known = _BUILTIN_CAPABILITIES.get(type(factor).__name__)
    if known is not None:
        return known
    return frozenset({_C.EVALUATE}) if callable(getattr(factor, "evaluate", None)) else frozenset()


@dataclass(frozen=True, slots=True)
class FactorCapabilityRecord:
    location: str
    factor_type: str
    capabilities: tuple[str, ...]
    qqa_compatible: bool
    exact_compatible: bool


@dataclass(frozen=True, slots=True)
class ModelCapabilityReport:
    factors: tuple[FactorCapabilityRecord, ...]
    qqa_compatible: bool
    exact_compatible: bool
    finite_bounds_for_qqa: bool
    unsupported_qqa: tuple[str, ...]
    unsupported_exact: tuple[str, ...]
    missing_bounds: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _qqa_capable(capabilities: frozenset[FactorCapability]) -> bool:
    return _C.EVALUATE in capabilities and bool(
        capabilities & {_C.DIFFERENTIABLE, _C.SUBGRADIENT, _C.PROX}
    )


def inspect_capabilities(model: ModelIR) -> ModelCapabilityReport:
    """Inspect every objective/constraint factor and every QQA bound."""
    if not isinstance(model, ModelIR):
        raise TypeError("model must be a ModelIR.")
    records: list[FactorCapabilityRecord] = []
    unsupported_qqa: list[str] = []
    unsupported_exact: list[str] = []
    expressions = [("objective", model.objective)] + [
        (f"constraint:{row.name}", row.expression) for row in model.constraints
    ]
    for location, expression in expressions:
        for index, factor in enumerate(expression.factors):
            capabilities = factor_capabilities(factor)
            qqa = _qqa_capable(capabilities)
            exact = {_C.EXACT_ENCODE, _C.PROOF_SAFE} <= capabilities
            label = f"{location}[{index}]={type(factor).__name__}"
            if not qqa:
                unsupported_qqa.append(label)
            if not exact:
                unsupported_exact.append(label)
            records.append(
                FactorCapabilityRecord(
                    location,
                    type(factor).__name__,
                    tuple(sorted(item.value for item in capabilities)),
                    qqa,
                    exact,
                )
            )

    missing_bounds = []
    for block in model.variables:
        if block.domain in {
            VariableDomain.BINARY,
            VariableDomain.SPIN,
            VariableDomain.CATEGORICAL,
            VariableDomain.PERMUTATION,
        }:
            continue
        lower = None if block.lower is None else torch.as_tensor(block.lower)
        upper = None if block.upper is None else torch.as_tensor(block.upper)
        if (
            lower is None
            or upper is None
            or not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all()
        ):
            missing_bounds.append(block.name)
    return ModelCapabilityReport(
        tuple(records),
        not unsupported_qqa and not missing_bounds,
        not unsupported_exact,
        not missing_bounds,
        tuple(unsupported_qqa),
        tuple(unsupported_exact),
        tuple(missing_bounds),
    )


def require_qqa_capabilities(model: ModelIR) -> ModelCapabilityReport:
    """Validate the pure-QQA route without silently changing semantics."""
    report = inspect_capabilities(model)
    messages = []
    if report.unsupported_qqa:
        messages.append("non-differentiable factors: " + ", ".join(report.unsupported_qqa))
    if report.missing_bounds:
        messages.append("missing/non-finite variable bounds: " + ", ".join(report.missing_bounds))
    if messages:
        raise NotImplementedError(
            "Pure QQA cannot execute this ModelIR without changing its semantics; "
            + "; ".join(messages)
            + ". Use a compatible CP/SAT/exact backend or supply explicit finite bounds."
        )
    return report


__all__ = [
    "FactorCapability",
    "FactorCapabilityRecord",
    "ModelCapabilityReport",
    "factor_capabilities",
    "inspect_capabilities",
    "require_qqa_capabilities",
]
