"""Deterministic row and column scaling for sparse algebraic models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from qqa.algebraic import AlgebraicConstraint, AlgebraicModel, SparseQuadratic


@dataclass(frozen=True, slots=True)
class ScalingFactors:
    """Positive scaling factors with mappings back to original units."""

    columns: np.ndarray
    rows: np.ndarray

    def __post_init__(self) -> None:
        columns = np.asarray(self.columns, dtype=np.float64)
        rows = np.asarray(self.rows, dtype=np.float64)
        if columns.ndim != 1 or rows.ndim != 1:
            raise ValueError("Scaling factors must be one-dimensional.")
        if not np.isfinite(columns).all() or not np.isfinite(rows).all():
            raise ValueError("Scaling factors must be finite.")
        if np.any(columns <= 0) or np.any(rows <= 0):
            raise ValueError("Scaling factors must be positive.")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rows", rows)

    def to_original(self, scaled_point: np.ndarray) -> np.ndarray:
        return np.asarray(scaled_point, dtype=np.float64) * self.columns

    def to_scaled(self, original_point: np.ndarray) -> np.ndarray:
        return np.asarray(original_point, dtype=np.float64) / self.columns


def _maximum_abs_by_column(model: AlgebraicModel) -> np.ndarray:
    maximum = np.zeros(model.num_variables, dtype=np.float64)

    def accumulate(expression: SparseQuadratic) -> None:
        linear = expression.linear_csr.tocoo()
        np.maximum.at(maximum, linear.col, np.abs(linear.data))
        quadratic = expression.quadratic.tocoo()
        np.maximum.at(maximum, quadratic.col, np.abs(quadratic.data))
        np.maximum.at(maximum, quadratic.row, np.abs(quadratic.data))

    accumulate(model.objective)
    for constraint in model.constraints:
        accumulate(constraint.expression)
    return maximum


def compute_scaling(
    model: AlgebraicModel,
    *,
    minimum: float = 1e-6,
    maximum: float = 1e6,
) -> ScalingFactors:
    """Compute bounded geometric column/row scales from sparse coefficients."""
    if not 0 < minimum <= 1 <= maximum:
        raise ValueError("Require 0 < minimum <= 1 <= maximum.")
    magnitude = _maximum_abs_by_column(model)
    columns = np.ones(model.num_variables, dtype=np.float64)
    nonzero = magnitude > 0
    columns[nonzero] = np.clip(1.0 / np.sqrt(magnitude[nonzero]), minimum, maximum)

    rows = np.ones(model.num_constraints, dtype=np.float64)
    diagonal = sparse.diags(columns)
    for index, constraint in enumerate(model.constraints):
        linear = constraint.expression.linear @ diagonal
        quadratic = diagonal @ constraint.expression.quadratic @ diagonal
        scale = max(
            np.max(np.abs(linear.data), initial=0.0),
            np.max(np.abs(quadratic.data), initial=0.0),
            1.0,
        )
        rows[index] = np.clip(1.0 / scale, minimum, maximum)
    return ScalingFactors(columns=columns, rows=rows)


def scaled_model(
    model: AlgebraicModel,
    factors: ScalingFactors | None = None,
) -> tuple[AlgebraicModel, ScalingFactors]:
    """Return ``x = column_scale * x_scaled`` model and reversible factors."""
    factors = compute_scaling(model) if factors is None else factors
    if factors.columns.shape != (model.num_variables,):
        raise ValueError("Column scales do not match model dimension.")
    if factors.rows.shape != (model.num_constraints,):
        raise ValueError("Row scales do not match constraint count.")
    diagonal = sparse.diags(factors.columns)

    def transform(expression: SparseQuadratic, row_scale: float = 1.0) -> SparseQuadratic:
        return SparseQuadratic(
            row_scale * (diagonal @ expression.quadratic @ diagonal),
            row_scale * (expression.linear @ diagonal),
            row_scale * expression.constant,
        )

    constraints = []
    for row, row_scale in zip(model.constraints, factors.rows, strict=True):
        constraints.append(
            AlgebraicConstraint(
                row.name,
                transform(row.expression, float(row_scale)),
                lower=float(row.lower * row_scale),
                upper=float(row.upper * row_scale),
            )
        )
    scaled = AlgebraicModel(
        name=model.name,
        variable_names=model.variable_names,
        variable_types=model.variable_types,
        lower_bounds=model.lower_bounds / factors.columns,
        upper_bounds=model.upper_bounds / factors.columns,
        objective=transform(model.objective),
        constraints=constraints,
        objective_sense=model.objective_sense,
        problem_type=model.problem_type,
        source_format=model.source_format,
        metadata={**model.metadata, "scaled": True},
    )
    return scaled, factors


__all__ = ["ScalingFactors", "compute_scaling", "scaled_model"]
