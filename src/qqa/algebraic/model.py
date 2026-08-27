"""A sparse, solver-independent algebraic intermediate representation.

The callable-based :mod:`qqa.mixed` API is intentionally convenient for
small user models.  MIPLIB and QPLIB require a different representation:
large sparse matrices, infinite bounds, stable original names, and exact
feasibility evaluation.  This module provides that representation without
embedding a machine path, hostname, or solver-specific object in serialised
metadata.
"""

from __future__ import annotations

import ipaddress
import math
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import numpy as np
from scipy import sparse

_ZERO_QUADRATICS: weakref.WeakValueDictionary[int, sparse.csr_matrix] = (
    weakref.WeakValueDictionary()
)


def _zero_quadratic(size: int) -> sparse.csr_matrix:
    """Share the structural zero Hessian across every linear model row."""
    cached = _ZERO_QUADRATICS.get(size)
    if cached is not None:
        return cached
    matrix = sparse.csr_matrix((size, size), dtype=np.float64)
    for values in (matrix.data, matrix.indices, matrix.indptr):
        values.setflags(write=False)
    _ZERO_QUADRATICS[size] = matrix
    return matrix


class VariableType(str, Enum):
    """Domain type of one algebraic variable."""

    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"
    IMPLICIT_INTEGER = "implicit_integer"

    @property
    def integral(self) -> bool:
        return self is not VariableType.CONTINUOUS


def _private_metadata_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return True
    parsed = urlsplit(value)
    host = parsed.hostname
    if host is None:
        return False
    lowered = host.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
    )


def _finite_vector(values: Sequence[float] | np.ndarray, size: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values.")
    array = array.copy()
    array.setflags(write=False)
    return array


def _bound_vector(values: Sequence[float] | np.ndarray, size: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},), got {array.shape}.")
    if np.isnan(array).any():
        raise ValueError(f"{label} must not contain NaN.")
    array = array.copy()
    array.setflags(write=False)
    return array


def _symmetric_csr(matrix: sparse.spmatrix | np.ndarray, size: int) -> sparse.csr_matrix:
    sparse_matrix = cast(sparse.spmatrix, matrix) if sparse.issparse(matrix) else None
    if sparse_matrix is not None and sparse_matrix.shape == (size, size) and sparse_matrix.nnz == 0:
        return _zero_quadratic(size)
    result = sparse.csr_matrix(matrix, dtype=np.float64, shape=(size, size))
    result.sum_duplicates()
    result.eliminate_zeros()
    if result.data.size and not np.isfinite(result.data).all():
        raise ValueError("Quadratic coefficients must be finite.")
    difference = result - result.T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
        raise ValueError("Quadratic matrix must be symmetric.")
    # Average tiny parser round-off asymmetries and make the arrays immutable.
    result = ((result + result.T) * 0.5).tocsr()
    result.sort_indices()
    for values in (result.data, result.indices, result.indptr):
        values.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SparseQuadratic:
    """Sparse expression ``0.5 * x.T @ quadratic @ x + linear @ x + constant``."""

    quadratic: sparse.spmatrix
    linear: sparse.spmatrix | Sequence[float] | np.ndarray
    constant: float = 0.0

    def __post_init__(self) -> None:
        if sparse.issparse(self.linear):
            linear = sparse.csr_matrix(self.linear, dtype=np.float64)
            if linear.shape[0] != 1:
                if linear.shape[1] == 1:
                    linear = linear.T.tocsr()
                else:
                    raise ValueError("Sparse linear coefficients must have shape (1, n).")
        else:
            dense = np.asarray(self.linear, dtype=np.float64)
            if dense.ndim != 1:
                raise ValueError("linear must be a one-dimensional vector.")
            linear = sparse.csr_matrix(dense.reshape(1, -1), dtype=np.float64)
        linear.sum_duplicates()
        linear.eliminate_zeros()
        linear.sort_indices()
        if linear.data.size and not np.isfinite(linear.data).all():
            raise ValueError("Linear coefficients must be finite.")
        constant = float(self.constant)
        if not math.isfinite(constant):
            raise ValueError("constant must be finite.")
        for values in (linear.data, linear.indices, linear.indptr):
            values.setflags(write=False)
        dimension = linear.shape[1]
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadratic", _symmetric_csr(self.quadratic, dimension))
        object.__setattr__(self, "constant", constant)

    @classmethod
    def linear_expression(
        cls,
        coefficients: sparse.spmatrix | Sequence[float] | np.ndarray,
        *,
        constant: float = 0.0,
    ) -> SparseQuadratic:
        if sparse.issparse(coefficients):
            vector = sparse.csr_matrix(coefficients, dtype=np.float64)
            if vector.shape[0] != 1:
                if vector.shape[1] == 1:
                    vector = vector.T.tocsr()
                else:
                    raise ValueError("Sparse coefficients must have shape (1, n).")
            dimension = vector.shape[1]
        else:
            dense = np.asarray(coefficients, dtype=np.float64)
            if dense.ndim != 1:
                raise ValueError("coefficients must be one-dimensional.")
            vector = dense
            dimension = len(dense)
        return cls(_zero_quadratic(dimension), vector, constant)

    @property
    def dimension(self) -> int:
        return self.linear_csr.shape[1]

    @property
    def linear_csr(self) -> sparse.csr_matrix:
        """Canonical CSR linear row established during validation."""
        return cast(sparse.csr_matrix, self.linear)

    @property
    def linear_nonzeros(self) -> int:
        return int(self.linear_csr.nnz)

    def linear_dense(self) -> np.ndarray:
        """Return a dense copy for algorithms that explicitly require one."""
        return np.asarray(self.linear_csr.toarray(), dtype=np.float64).reshape(-1)

    @property
    def is_linear(self) -> bool:
        return self.quadratic.nnz == 0

    def value(self, point: Sequence[float] | np.ndarray) -> float:
        x = _finite_vector(point, self.dimension, "point")
        linear_value = float(self.linear_csr.dot(x)[0])
        return float(0.5 * x @ self.quadratic.dot(x) + linear_value + self.constant)

    def gradient(self, point: Sequence[float] | np.ndarray) -> np.ndarray:
        x = _finite_vector(point, self.dimension, "point")
        return np.asarray(self.quadratic.dot(x) + self.linear_dense(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AlgebraicConstraint:
    """A ranged sparse algebraic constraint ``lower <= expression <= upper``."""

    name: str
    expression: SparseQuadratic
    lower: float = -math.inf
    upper: float = math.inf

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Constraint name must be a non-empty string.")
        lower = float(self.lower)
        upper = float(self.upper)
        if math.isnan(lower) or math.isnan(upper):
            raise ValueError("Constraint bounds must not be NaN.")
        if lower > upper:
            raise ValueError(f"Constraint {self.name!r} has lower > upper.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def violation(self, value: float) -> float:
        return max(0.0, self.lower - value, value - self.upper)


@dataclass(frozen=True, slots=True)
class AlgebraicEvaluation:
    """Objective and official-style feasibility diagnostics at one point."""

    objective: float
    constraint_values: np.ndarray
    constraint_violation: float
    bound_violation: float
    integrality_violation: float
    maximum_infeasibility: float

    @property
    def feasible(self) -> bool:
        return self.maximum_infeasibility <= 1e-6


@dataclass(frozen=True, slots=True)
class AlgebraicModel:
    """Sparse linear/quadratic model with original-space variable metadata."""

    name: str
    variable_names: Sequence[str]
    variable_types: Sequence[VariableType | str]
    lower_bounds: Sequence[float] | np.ndarray
    upper_bounds: Sequence[float] | np.ndarray
    objective: SparseQuadratic
    constraints: Sequence[AlgebraicConstraint] = ()
    objective_sense: Literal["minimize", "maximize"] = "minimize"
    problem_type: str | None = None
    source_format: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Model name must be a non-empty string.")
        names = tuple(str(name) for name in self.variable_names)
        if not names or any(not name for name in names):
            raise ValueError("variable_names must contain non-empty names.")
        if len(set(names)) != len(names):
            raise ValueError("variable_names must be unique.")
        size = len(names)
        types = tuple(
            value if isinstance(value, VariableType) else VariableType(value)
            for value in self.variable_types
        )
        if len(types) != size:
            raise ValueError("variable_types must align with variable_names.")
        lower = _bound_vector(self.lower_bounds, size, "lower_bounds")
        upper = _bound_vector(self.upper_bounds, size, "upper_bounds")
        if np.any(lower > upper):
            raise ValueError("Every lower bound must be <= its upper bound.")
        if self.objective.dimension != size:
            raise ValueError("Objective dimension does not match variables.")
        constraints = tuple(self.constraints)
        if any(row.expression.dimension != size for row in constraints):
            raise ValueError("Constraint dimension does not match variables.")
        constraint_names = [row.name for row in constraints]
        if len(set(constraint_names)) != len(constraint_names):
            raise ValueError("Constraint names must be unique.")
        if self.objective_sense not in {"minimize", "maximize"}:
            raise ValueError("objective_sense must be 'minimize' or 'maximize'.")

        # Metadata is deliberately shallow and path-free by contract. Loaders
        # store a source basename/URL and snapshot hash, never an absolute path.
        safe_metadata = dict(self.metadata)
        for key, value in safe_metadata.items():
            if not isinstance(key, str):
                raise TypeError("metadata keys must be strings.")
            if key.lower() in {"absolute_path", "hostname", "server", "internal_url"}:
                raise ValueError(f"Private environment metadata key {key!r} is not allowed.")
            if _private_metadata_value(value):
                raise ValueError(f"Private environment metadata value for {key!r} is not allowed.")

        object.__setattr__(self, "variable_names", names)
        object.__setattr__(self, "variable_types", types)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "metadata", MappingProxyType(safe_metadata))

    @property
    def num_variables(self) -> int:
        return len(self.variable_names)

    @property
    def num_constraints(self) -> int:
        return len(self.constraints)

    @property
    def variable_type_values(self) -> tuple[VariableType, ...]:
        """Canonical variable domains established during validation."""
        return cast(tuple[VariableType, ...], self.variable_types)

    @property
    def lower_array(self) -> np.ndarray:
        """Canonical immutable lower-bound vector."""
        return cast(np.ndarray, self.lower_bounds)

    @property
    def upper_array(self) -> np.ndarray:
        """Canonical immutable upper-bound vector."""
        return cast(np.ndarray, self.upper_bounds)

    @property
    def integer_indices(self) -> np.ndarray:
        return np.fromiter(
            (index for index, kind in enumerate(self.variable_type_values) if kind.integral),
            dtype=np.int64,
        )

    @property
    def continuous_indices(self) -> np.ndarray:
        return np.fromiter(
            (
                index
                for index, kind in enumerate(self.variable_type_values)
                if kind is VariableType.CONTINUOUS
            ),
            dtype=np.int64,
        )

    def validate_point(self, point: Sequence[float] | np.ndarray) -> np.ndarray:
        return _finite_vector(point, self.num_variables, "point")

    def evaluate(self, point: Sequence[float] | np.ndarray) -> AlgebraicEvaluation:
        """Evaluate objective and QPLIB-compatible maximum infeasibility."""
        x = self.validate_point(point)
        values = np.asarray(
            [constraint.expression.value(x) for constraint in self.constraints],
            dtype=np.float64,
        )
        constraint_violation = max(
            (
                row.violation(float(value))
                for row, value in zip(self.constraints, values, strict=True)
            ),
            default=0.0,
        )
        bound_violation = float(
            max(
                0.0,
                np.max(self.lower_bounds - x, initial=0.0),
                np.max(x - self.upper_bounds, initial=0.0),
            )
        )
        integer = self.integer_indices
        integrality_violation = (
            float(np.max(np.abs(x[integer] - np.rint(x[integer])), initial=0.0))
            if integer.size
            else 0.0
        )
        maximum = max(constraint_violation, bound_violation, integrality_violation)
        return AlgebraicEvaluation(
            objective=self.objective.value(x),
            constraint_values=values,
            constraint_violation=float(constraint_violation),
            bound_violation=bound_violation,
            integrality_violation=integrality_violation,
            maximum_infeasibility=float(maximum),
        )

    def summary(self) -> dict[str, Any]:
        counts = {kind.value: self.variable_types.count(kind) for kind in VariableType}
        return {
            "name": self.name,
            "problem_type": self.problem_type,
            "source_format": self.source_format,
            "objective_sense": self.objective_sense,
            "num_variables": self.num_variables,
            "num_constraints": self.num_constraints,
            "variable_counts": counts,
            "objective_linear_nonzeros": self.objective.linear_nonzeros,
            "objective_quadratic_nonzeros": int(self.objective.quadratic.nnz),
            "constraint_linear_nonzeros": int(
                sum(row.expression.linear_nonzeros for row in self.constraints)
            ),
            "constraint_quadratic_nonzeros": int(
                sum(row.expression.quadratic.nnz for row in self.constraints)
            ),
            "metadata": dict(self.metadata),
        }


__all__ = [
    "AlgebraicConstraint",
    "AlgebraicEvaluation",
    "AlgebraicModel",
    "SparseQuadratic",
    "VariableType",
]
