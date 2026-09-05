"""Immutable factor-based intermediate representation used by all planners."""

from __future__ import annotations

import ipaddress
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from operator import mul
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable
from urllib.parse import urlsplit

import torch


class VariableDomain(str, Enum):
    BINARY = "binary"
    SPIN = "spin"
    INTEGER = "integer"
    REAL = "real"
    CATEGORICAL = "categorical"
    PERMUTATION = "permutation"


class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    @property
    def canonical_sign(self) -> float:
        return 1.0 if self is ObjectiveSense.MINIMIZE else -1.0


def _shape_size(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


def _tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=dtype).detach().clone()
    if result.is_floating_point() and not torch.isfinite(result).all():
        raise ValueError("Factor data must contain only finite values.")
    return result


def _bound_tensor(value: Any) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float64).detach().clone()
    if torch.isnan(result).any():
        raise ValueError("Variable bounds must not contain NaN.")
    return result


@dataclass(frozen=True, slots=True)
class VariableBlock:
    """One named, contiguous variable block in original model order."""

    name: str
    domain: VariableDomain | str
    shape: tuple[int, ...] = (1,)
    lower: torch.Tensor | float | None = None
    upper: torch.Tensor | float | None = None
    categories: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("VariableBlock.name must be non-empty.")
        domain = VariableDomain(self.domain)
        shape = tuple(self.shape)
        if not shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape
        ):
            raise ValueError("VariableBlock.shape must contain positive integers.")
        if self.categories is not None and (
            isinstance(self.categories, bool) or self.categories < 2
        ):
            raise ValueError("categories must be an integer >= 2 or None.")
        if (
            domain in {VariableDomain.CATEGORICAL, VariableDomain.PERMUTATION}
            and self.categories is None
        ):
            raise ValueError(f"{domain.value} blocks require categories.")
        lower = None if self.lower is None else _bound_tensor(self.lower)
        upper = None if self.upper is None else _bound_tensor(self.upper)
        if lower is not None and lower.numel() not in {1, _shape_size(shape)}:
            raise ValueError("lower must be scalar or align with the block shape.")
        if upper is not None and upper.numel() not in {1, _shape_size(shape)}:
            raise ValueError("upper must be scalar or align with the block shape.")
        if (
            lower is not None
            and upper is not None
            and torch.any(lower.reshape(-1) > upper.reshape(-1))
        ):
            raise ValueError("lower must not exceed upper.")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def size(self) -> int:
        return _shape_size(self.shape)

    @property
    def domain_value(self) -> VariableDomain:
        """Canonical domain established during validation."""
        return cast(VariableDomain, self.domain)


@runtime_checkable
class Factor(Protocol):
    """A vectorised sparse factor over flattened model variables."""

    def evaluate(self, values: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class LinearFactor:
    indices: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        indices = _tensor(self.indices, dtype=torch.long).reshape(-1)
        weights = _tensor(self.weights).reshape(-1)
        if indices.shape != weights.shape or torch.any(indices < 0):
            raise ValueError("LinearFactor indices/weights must align and indices be non-negative.")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "weights", weights)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        return (values[..., self.indices] * self.weights.to(values)).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class QuadraticEdgeFactor:
    edge_index: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        edges = _tensor(self.edge_index, dtype=torch.long)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, edges).")
        weights = _tensor(self.weights).reshape(-1)
        if edges.shape[1] != weights.numel() or torch.any(edges < 0):
            raise ValueError("QuadraticEdgeFactor edges and weights must align.")
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "weights", weights)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        source, target = self.edge_index
        return (
            values[..., source]
            * values[..., target]
            * self.weights.to(device=values.device, dtype=values.dtype)
        ).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class HigherOrderFactor:
    indices: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        indices = _tensor(self.indices, dtype=torch.long)
        if indices.ndim != 2 or indices.shape[1] < 3 or torch.any(indices < 0):
            raise ValueError("HigherOrderFactor indices must have shape (factors, order>=3).")
        weights = _tensor(self.weights).reshape(-1)
        if indices.shape[0] != weights.numel():
            raise ValueError("HigherOrderFactor indices and weights must align.")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "weights", weights)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        selected = values[..., self.indices]
        return (selected.prod(dim=-1) * self.weights.to(values)).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class ClauseFactor:
    """Weighted CNF clauses; positive signs mean ``x``, negative mean ``not x``."""

    indices: torch.Tensor
    signs: torch.Tensor
    weights: torch.Tensor | None = None

    def __post_init__(self) -> None:
        indices = _tensor(self.indices, dtype=torch.long)
        signs = _tensor(self.signs, dtype=torch.int8)
        if indices.ndim != 2 or indices.shape != signs.shape or torch.any(indices < 0):
            raise ValueError("ClauseFactor indices/signs must be aligned rank-two tensors.")
        if torch.any((signs != 1) & (signs != -1)):
            raise ValueError("Clause signs must be +1 or -1.")
        weights = (
            torch.ones(indices.shape[0], dtype=torch.float32)
            if self.weights is None
            else _tensor(self.weights).reshape(-1)
        )
        if weights.numel() != indices.shape[0] or torch.any(weights < 0):
            raise ValueError("Clause weights must be non-negative and align with clauses.")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "signs", signs)
        object.__setattr__(self, "weights", weights)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        selected = values[..., self.indices]
        literals = torch.where(self.signs.to(values.device) > 0, selected, 1.0 - selected)
        unsatisfied = (1.0 - literals).prod(dim=-1)
        weights = self.weights
        assert weights is not None
        return (unsatisfied * weights.to(values)).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class CardinalityFactor:
    indices: torch.Tensor
    target: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        indices = _tensor(self.indices, dtype=torch.long).reshape(-1)
        if indices.numel() == 0 or torch.any(indices < 0):
            raise ValueError("CardinalityFactor requires non-negative indices.")
        if not math.isfinite(self.target) or not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Cardinality target/weight must be finite and weight non-negative.")
        object.__setattr__(self, "indices", indices)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        residual = values[..., self.indices].sum(dim=-1) - self.target
        return self.weight * residual.square()


@dataclass(frozen=True, slots=True)
class AllDifferentFactor:
    indices: torch.Tensor
    weight: float = 1.0

    def __post_init__(self) -> None:
        indices = _tensor(self.indices, dtype=torch.long).reshape(-1)
        if indices.numel() < 2 or torch.any(indices < 0):
            raise ValueError("AllDifferentFactor requires at least two indices.")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("AllDifferentFactor weight must be finite and positive.")
        object.__setattr__(self, "indices", indices)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        selected = values[..., self.indices]
        duplicate = selected.unsqueeze(-1) == selected.unsqueeze(-2)
        return self.weight * torch.triu(duplicate, diagonal=1).sum(dim=(-2, -1)).to(values.dtype)


@dataclass(frozen=True, slots=True)
class AssignmentFactor:
    """Squared row/column residuals for a flattened assignment matrix."""

    indices: torch.Tensor
    weight: float = 1.0

    def __post_init__(self) -> None:
        indices = _tensor(self.indices, dtype=torch.long)
        if indices.ndim != 2 or torch.any(indices < 0):
            raise ValueError("AssignmentFactor.indices must be a rank-two index matrix.")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("AssignmentFactor weight must be finite and positive.")
        object.__setattr__(self, "indices", indices)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        matrix = values[..., self.indices]
        row = (matrix.sum(dim=-1) - 1.0).square().sum(dim=-1)
        column = (matrix.sum(dim=-2) - 1.0).square().sum(dim=-1)
        return self.weight * (row + column)


@dataclass(frozen=True, slots=True)
class PairwisePottsFactor:
    edge_index: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        edges = _tensor(self.edge_index, dtype=torch.long)
        weights = _tensor(self.weights).reshape(-1)
        if (
            edges.ndim != 2
            or edges.shape[0] != 2
            or edges.shape[1] != weights.numel()
            or torch.any(edges < 0)
        ):
            raise ValueError("PairwisePottsFactor edges/weights must align.")
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "weights", weights)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        source, target = self.edge_index
        if values.ndim >= 3:
            same = (values[..., source, :] * values[..., target, :]).sum(dim=-1)
        else:
            same = (values[..., source] == values[..., target]).to(values.dtype)
        return ((1.0 - same) * self.weights.to(values)).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class TableFactor:
    indices: torch.Tensor
    table: torch.Tensor

    def __post_init__(self) -> None:
        indices = _tensor(self.indices, dtype=torch.long).reshape(-1)
        table = _tensor(self.table)
        if table.ndim != indices.numel() or torch.any(indices < 0):
            raise ValueError("Table rank must equal the number of scoped variables.")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "table", table)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        states = values[..., self.indices].long()
        shape = torch.as_tensor(self.table.shape, device=values.device, dtype=torch.long)
        if torch.any(states < 0) or torch.any(states >= shape):
            raise ValueError("TableFactor received a state outside its table domain.")
        strides = torch.ones_like(shape)
        if len(shape) > 1:
            strides[:-1] = torch.flip(
                torch.cumprod(torch.flip(shape[1:], dims=(0,)), dim=0), dims=(0,)
            )
        flat_index = (states * strides).sum(dim=-1)
        return self.table.to(values).reshape(-1)[flat_index]


@dataclass(frozen=True, slots=True)
class BlackBoxFactor:
    function: Callable[[torch.Tensor], torch.Tensor]
    name: str = "black-box"
    differentiable: bool = True

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("BlackBoxFactor.function must be callable.")
        if not isinstance(self.differentiable, bool):
            raise TypeError("BlackBoxFactor.differentiable must be boolean.")

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Explicit execution declaration; black boxes are never proof-safe."""
        return ("evaluate", "differentiable") if self.differentiable else ("evaluate",)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        result = torch.as_tensor(self.function(values), device=values.device, dtype=values.dtype)
        expected = values.shape[:-2] if values.ndim >= 3 else values.shape[:-1]
        if result.shape != expected:
            raise ValueError("BlackBoxFactor must preserve every leading batch dimension.")
        return result


@dataclass(frozen=True, slots=True)
class ObjectiveIR:
    factors: tuple[Factor, ...]
    constant: float = 0.0

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        if any(not isinstance(item, Factor) for item in factors):
            raise TypeError("Every objective item must implement Factor.")
        if not math.isfinite(self.constant):
            raise ValueError("Objective constant must be finite.")
        object.__setattr__(self, "factors", factors)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        if not self.factors:
            # A structured categorical state is (batch, nodes, categories),
            # while ordinary models are (batch, variables).
            batch_shape = values.shape[:-2] if values.ndim >= 3 else values.shape[:-1]
            return torch.full(batch_shape, self.constant, device=values.device, dtype=values.dtype)
        first, *remaining = self.factors
        result = first.evaluate(values) + self.constant
        for factor in remaining:
            contribution = factor.evaluate(values)
            if contribution.shape != result.shape:
                raise ValueError("Every factor must return the same leading batch shape.")
            result = result + contribution
        return result


@dataclass(frozen=True, slots=True)
class ConstraintIR:
    name: str
    expression: ObjectiveIR
    sense: str = "<="
    rhs: float = 0.0
    scale: float = 1.0
    tolerance: float = 1e-6
    weight: float = 1.0
    priority: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ConstraintIR.name must be non-empty.")
        if self.sense not in {"<=", ">=", "=="}:
            raise ValueError("ConstraintIR.sense must be <=, >=, or ==.")
        if not all(
            math.isfinite(value)
            for value in (self.rhs, self.scale, self.tolerance, self.weight, self.priority)
        ):
            raise ValueError(
                "Constraint rhs, scale, tolerance, weight, and priority must be finite."
            )
        if self.scale <= 0 or self.tolerance < 0 or self.weight <= 0 or self.priority <= 0:
            raise ValueError(
                "Constraint scale/weight/priority must be > 0 and tolerance must be >= 0."
            )

    def canonical_residual(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.expression.evaluate(values) - self.rhs
        return -residual if self.sense == ">=" else residual

    def violation(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.canonical_residual(values)
        return residual.abs() if self.sense == "==" else residual.clamp_min(0.0)


def _private_string(value: str) -> bool:
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return True
    host = urlsplit(value).hostname
    if host is None:
        return False
    host = host.lower().rstrip(".")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(address.is_private or address.is_loopback or address.is_link_local)


def _safe_metadata(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(mapping)
    forbidden = {"absolute_path", "hostname", "server", "internal_url", "worker"}
    for key, value in result.items():
        if not isinstance(key, str):
            raise TypeError("Metadata keys must be strings.")
        if key.lower() in forbidden:
            raise ValueError(f"Private environment metadata key {key!r} is not allowed.")
        if isinstance(value, str) and _private_string(value):
            raise ValueError(f"Private environment metadata value for {key!r} is not allowed.")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    name: str = "model"
    problem_class: str | None = None
    source_format: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelMetadata.name must be non-empty.")
        object.__setattr__(self, "attributes", _safe_metadata(self.attributes))


@dataclass(frozen=True, slots=True)
class TransformationRecord:
    operation: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation:
            raise ValueError("Transformation operation must be non-empty.")
        object.__setattr__(self, "details", _safe_metadata(self.details))


@dataclass(frozen=True, slots=True)
class ModelIR:
    """Canonical sparse model with one objective-sense conversion boundary."""

    variables: tuple[VariableBlock, ...]
    objective: ObjectiveIR
    constraints: tuple[ConstraintIR, ...] = ()
    sense: ObjectiveSense | str = ObjectiveSense.MINIMIZE
    metadata: ModelMetadata = field(default_factory=ModelMetadata)
    transformations: tuple[TransformationRecord, ...] = ()

    def __post_init__(self) -> None:
        variables = tuple(self.variables)
        constraints = tuple(self.constraints)
        if not variables:
            raise ValueError("ModelIR requires at least one variable block.")
        if len({item.name for item in variables}) != len(variables):
            raise ValueError("Variable block names must be unique.")
        if len({item.name for item in constraints}) != len(constraints):
            raise ValueError("Constraint names must be unique.")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "sense", ObjectiveSense(self.sense))
        object.__setattr__(self, "transformations", tuple(self.transformations))

    @property
    def num_variables(self) -> int:
        return sum(block.size for block in self.variables)

    @property
    def structured_block(self) -> VariableBlock | None:
        """Return the one categorical/permutation block, when present."""
        if len(self.variables) != 1:
            return None
        block = self.variables[0]
        return (
            block
            if block.domain in {VariableDomain.CATEGORICAL, VariableDomain.PERMUTATION}
            else None
        )

    def _validate_values(self, values: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(values):
            raise TypeError("ModelIR values must be a torch.Tensor.")
        structured = self.structured_block
        if structured is not None:
            categories = int(structured.categories or 0)
            if values.ndim == 1 and values.shape[0] == structured.size:
                return values.unsqueeze(0)
            if values.ndim == 2:
                if values.shape == (structured.size, categories):
                    return values.unsqueeze(0)
                if values.shape[-1] == structured.size:
                    return values
            if values.ndim >= 3 and values.shape[-2:] == (structured.size, categories):
                return values
            raise ValueError(
                "Structured values must contain state labels (..., nodes) or "
                "one-hot/simplex values (..., nodes, categories)."
            )
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape[-1] != self.num_variables:
            raise ValueError(
                f"Expected a final dimension of {self.num_variables}, got {values.shape[-1]}."
            )
        return values

    def objective_values(self, values: torch.Tensor) -> torch.Tensor:
        return self.objective.evaluate(self._validate_values(values))

    def internal_energy(self, values: torch.Tensor) -> torch.Tensor:
        return ObjectiveSense(self.sense).canonical_sign * self.objective_values(values)

    def constraint_violations(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self._validate_values(values)
        return {row.name: row.violation(values) for row in self.constraints}

    def domain_violations(self, values: torch.Tensor) -> torch.Tensor:
        """Return the maximum variable-domain violation for every candidate.

        Objective and factor evaluation intentionally accepts relaxed points.
        Feasibility does not: it independently checks finiteness, declared
        bounds, integrality, binary/spin membership, and structured one-hot or
        permutation semantics.  Keeping these contracts separate prevents a
        relaxed objective probe from being mistaken for a verified incumbent.
        """
        values = self._validate_values(values)
        if values.is_complex():
            raise TypeError("ModelIR values must use a real numeric tensor dtype.")
        work = values if values.is_floating_point() else values.to(torch.float64)
        structured = self.structured_block
        batch_shape = (
            work.shape[:-2] if structured is not None and work.ndim >= 3 else work.shape[:-1]
        )
        maximum = torch.zeros(batch_shape, device=work.device, dtype=work.dtype)
        finite = torch.isfinite(work)
        finite_by_candidate = finite.reshape(*batch_shape, -1).all(dim=-1)
        finite_penalty = torch.where(
            finite_by_candidate,
            torch.zeros_like(maximum),
            torch.full_like(maximum, torch.finfo(work.dtype).max),
        )
        maximum = torch.maximum(maximum, finite_penalty)
        safe = torch.where(finite, work, torch.zeros_like(work))

        if structured is not None:
            categories = int(structured.categories or 0)
            if safe.ndim >= 3 and safe.shape[-2:] == (structured.size, categories):
                maximum = torch.maximum(
                    maximum,
                    (safe - safe.round()).abs().reshape(*batch_shape, -1).amax(dim=-1),
                )
                maximum = torch.maximum(maximum, (-safe).clamp_min(0).amax(dim=(-2, -1)))
                maximum = torch.maximum(maximum, (safe - 1.0).clamp_min(0).amax(dim=(-2, -1)))
                maximum = torch.maximum(maximum, (safe.sum(dim=-1) - 1.0).abs().amax(dim=-1))
                if structured.domain is VariableDomain.PERMUTATION:
                    maximum = torch.maximum(maximum, (safe.sum(dim=-2) - 1.0).abs().amax(dim=-1))
                return maximum
            maximum = torch.maximum(maximum, (safe - safe.round()).abs().amax(dim=-1))
            maximum = torch.maximum(maximum, (-safe).clamp_min(0).amax(dim=-1))
            maximum = torch.maximum(
                maximum, (safe - float(categories - 1)).clamp_min(0).amax(dim=-1)
            )
            if structured.domain is VariableDomain.PERMUTATION:
                sorted_states = safe.sort(dim=-1).values
                target = torch.arange(categories, device=safe.device, dtype=safe.dtype)
                if structured.size != categories:
                    maximum = torch.maximum(maximum, torch.ones_like(maximum))
                else:
                    maximum = torch.maximum(maximum, (sorted_states - target).abs().amax(dim=-1))
            return maximum

        offset = 0
        for block in self.variables:
            part = safe[..., offset : offset + block.size]
            offset += block.size
            lower = block.lower
            upper = block.upper
            implicit_lower: float | None
            implicit_upper: float | None
            if block.domain is VariableDomain.BINARY:
                implicit_lower, implicit_upper = 0.0, 1.0
                membership = torch.minimum(part.abs(), (part - 1.0).abs()).amax(dim=-1)
                maximum = torch.maximum(maximum, membership)
            elif block.domain is VariableDomain.SPIN:
                implicit_lower, implicit_upper = -1.0, 1.0
                membership = torch.minimum((part + 1.0).abs(), (part - 1.0).abs()).amax(dim=-1)
                maximum = torch.maximum(maximum, membership)
            else:
                implicit_lower = implicit_upper = None
                if block.domain is VariableDomain.INTEGER:
                    maximum = torch.maximum(maximum, (part - part.round()).abs().amax(dim=-1))
            if lower is not None or implicit_lower is not None:
                bound = (
                    (
                        torch.as_tensor(lower, device=part.device, dtype=part.dtype)
                        if lower is not None
                        else part.new_tensor(implicit_lower)
                    )
                    .reshape(-1)
                    .expand(block.size)
                )
                maximum = torch.maximum(maximum, (bound - part).clamp_min(0).amax(dim=-1))
            if upper is not None or implicit_upper is not None:
                bound = (
                    (
                        torch.as_tensor(upper, device=part.device, dtype=part.dtype)
                        if upper is not None
                        else part.new_tensor(implicit_upper)
                    )
                    .reshape(-1)
                    .expand(block.size)
                )
                maximum = torch.maximum(maximum, (part - bound).clamp_min(0).amax(dim=-1))
        return maximum

    def feasible(self, values: torch.Tensor) -> torch.Tensor:
        values = self._validate_values(values)
        mask = self.domain_violations(values) <= 1e-6
        for row in self.constraints:
            mask &= row.violation(values) <= row.tolerance
        return mask

    def transformed(self, operation: str, **details: Any) -> ModelIR:
        """Return a copy with one reversible transformation ledger entry."""
        from dataclasses import replace

        return replace(
            self,
            transformations=(*self.transformations, TransformationRecord(operation, details)),
        )


__all__ = [
    "AllDifferentFactor",
    "AssignmentFactor",
    "BlackBoxFactor",
    "CardinalityFactor",
    "ClauseFactor",
    "ConstraintIR",
    "Factor",
    "HigherOrderFactor",
    "LinearFactor",
    "ModelIR",
    "ModelMetadata",
    "ObjectiveIR",
    "ObjectiveSense",
    "PairwisePottsFactor",
    "QuadraticEdgeFactor",
    "TableFactor",
    "TransformationRecord",
    "VariableBlock",
    "VariableDomain",
]
