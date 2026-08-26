"""Objective- and LP-row-aware local surrogate models for conditional QQA."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qqa.algebraic import AlgebraicModel
from qqa.hybrid.core_selector import CoreSelection
from qqa.presolve.scip_bridge import SCIPState


@dataclass(frozen=True, slots=True)
class CoreSurrogate:
    """Small dense restriction used only for a selected integer core.

    The full benchmark model remains sparse.  At most ``core_size`` columns
    and ``max_lp_rows`` currently active SCIP rows are materialised here.
    """

    quadratic: np.ndarray
    linear: np.ndarray
    objective_scale: float
    row_matrix: np.ndarray
    row_offset: np.ndarray
    row_lower: np.ndarray
    row_upper: np.ndarray
    row_scale: np.ndarray
    objective_source: str

    @property
    def num_variables(self) -> int:
        return int(self.linear.size)

    @property
    def num_rows(self) -> int:
        return int(self.row_matrix.shape[0])

    def objective_values(self, values: np.ndarray) -> np.ndarray:
        points = np.asarray(values, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != self.num_variables:
            raise ValueError("values do not match the surrogate core dimension.")
        quadratic = 0.5 * np.einsum("bi,ij,bj->b", points, self.quadratic, points, optimize=True)
        return (quadratic + points @ self.linear) / self.objective_scale

    def row_violations(self, values: np.ndarray) -> np.ndarray:
        points = np.asarray(values, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if self.num_rows == 0:
            return np.empty((len(points), 0), dtype=np.float64)
        activity = points @ self.row_matrix.T + self.row_offset
        lower = np.maximum(self.row_lower - activity, 0.0)
        upper = np.maximum(activity - self.row_upper, 0.0)
        return (lower + upper) / self.row_scale

    def merit_values(
        self,
        values: np.ndarray,
        *,
        target: np.ndarray,
        span: np.ndarray,
        row_penalty: float,
        proximity_weight: float,
    ) -> np.ndarray:
        """Evaluate the same feasibility/objective merit used by QQA."""
        points = np.asarray(values, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        violations = self.row_violations(points)
        row_loss = (
            row_penalty * np.mean(np.square(violations), axis=1)
            if self.num_rows
            else np.zeros(len(points), dtype=np.float64)
        )
        proximity = proximity_weight * np.mean(
            np.square((points - np.asarray(target, dtype=np.float64)) / span), axis=1
        )
        return self.objective_values(points) + row_loss + proximity


def generate_surrogate_candidates(
    surrogate: CoreSurrogate,
    *,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_candidates: int = 4,
    row_penalty: float = 20.0,
    proximity_weight: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Generate cheap row-aware floor/ceil candidates by coordinate descent."""
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates < 0
    ):
        raise ValueError("max_candidates must be a non-negative integer.")
    if max_candidates == 0:
        return np.empty((0, surrogate.num_variables), dtype=np.float64)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    target = np.asarray(target, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if (
        target.shape != (surrogate.num_variables,)
        or lower.shape != target.shape
        or upper.shape != target.shape
    ):
        raise ValueError("target and bounds must match the surrogate core dimension.")
    span = np.maximum(1.0, upper - lower)
    rng = np.random.default_rng(seed)
    starts = [np.minimum(np.maximum(np.rint(target), lower), upper)]
    floor = np.minimum(np.maximum(np.floor(target), lower), upper)
    ceil = np.minimum(np.maximum(np.ceil(target), lower), upper)
    for _ in range(max_candidates * 2):
        starts.append(np.where(rng.random(len(target)) < 0.5, floor, ceil))

    candidates: dict[bytes, tuple[float, np.ndarray]] = {}
    for start in starts:
        current = start.astype(np.float64, copy=True)
        current_merit = float(
            surrogate.merit_values(
                current,
                target=target,
                span=span,
                row_penalty=row_penalty,
                proximity_weight=proximity_weight,
            )[0]
        )
        for _ in range(2 * len(current)):
            alternatives = np.repeat(current.reshape(1, -1), len(current), axis=0)
            diagonal = np.arange(len(current))
            alternatives[diagonal, diagonal] = np.where(
                current <= lower + 1e-9,
                upper,
                lower,
            )
            merits = surrogate.merit_values(
                alternatives,
                target=target,
                span=span,
                row_penalty=row_penalty,
                proximity_weight=proximity_weight,
            )
            best = int(np.argmin(merits))
            if float(merits[best]) >= current_merit - 1e-12:
                break
            current = alternatives[best]
            current_merit = float(merits[best])
        signature = np.asarray(current, dtype=np.int64).tobytes()
        previous = candidates.get(signature)
        if previous is None or current_merit < previous[0]:
            candidates[signature] = (current_merit, current)
    ranked = sorted(candidates.values(), key=lambda item: item[0])[:max_candidates]
    return np.stack([item[1] for item in ranked]) if ranked else np.empty((0, len(target)))


def _fallback_point(algebraic: AlgebraicModel) -> np.ndarray:
    point = np.zeros(algebraic.num_variables, dtype=np.float64)
    finite_lower = np.isfinite(algebraic.lower_bounds)
    finite_upper = np.isfinite(algebraic.upper_bounds)
    point[finite_lower] = np.maximum(point[finite_lower], algebraic.lower_bounds[finite_lower])
    point[finite_upper] = np.minimum(point[finite_upper], algebraic.upper_bounds[finite_upper])
    return point


def _restricted_objective(
    state: SCIPState,
    selection: CoreSelection,
    selected_indices: np.ndarray,
    algebraic: AlgebraicModel | None,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    size = len(selected_indices)
    if algebraic is not None:
        by_name = {name: index for index, name in enumerate(algebraic.variable_names)}
        original_indices = [by_name.get(state.names[index]) for index in selected_indices]
        if (
            all(index is not None for index in original_indices)
            and len(set(original_indices)) == size
        ):
            original = np.asarray(original_indices, dtype=np.int64)
            base = _fallback_point(algebraic)
            state_to_original = np.asarray(
                [by_name.get(name, -1) for name in state.names], dtype=np.int64
            )
            mapped = state_to_original >= 0
            reference = (
                state.incumbent_values if state.incumbent_values is not None else state.lp_values
            )
            base[state_to_original[mapped]] = reference[mapped]
            fixed_map = state_to_original[selection.fixed_indices]
            fixed_mapped = fixed_map >= 0
            base[fixed_map[fixed_mapped]] = selection.fixed_values[fixed_mapped]

            expression = algebraic.objective
            quadratic = expression.quadratic[original][:, original].toarray()
            full_gradient = expression.quadratic.dot(base) + expression.linear_dense()
            base_core = base[original]
            linear = full_gradient[original] - quadratic @ base_core
            sense = -1.0 if algebraic.objective_sense == "maximize" else 1.0
            return sense * quadratic, sense * linear, sense, "algebraic"

    quadratic = np.zeros((size, size), dtype=np.float64)
    linear = np.asarray(
        [float(state.variables[index].getObj()) for index in selected_indices],
        dtype=np.float64,
    )
    return quadratic, linear, 1.0, "scip-linear"


def _restricted_lp_rows(
    model,
    state: SCIPState,
    selection: CoreSelection,
    selected_indices: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_lp_rows: int,
    ignored_variables: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = len(selected_indices)
    if max_lp_rows == 0:
        return (
            np.empty((0, size), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    by_exact_name = {variable.name: index for index, variable in enumerate(state.variables)}
    by_public_name = {name: index for index, name in enumerate(state.names)}
    core_position = {int(index): position for position, index in enumerate(selected_indices)}
    reference = state.lp_values.copy()
    reference[selection.fixed_indices] = selection.fixed_values
    infinity = abs(float(model.infinity()))
    rounded = np.rint(state.lp_values[selected_indices])
    rounded = np.minimum(np.maximum(rounded, lower), upper)
    candidates: list[tuple[float, np.ndarray, float, float, float, float]] = []

    for row in model.getLPRowsData():
        coefficients = np.zeros(size, dtype=np.float64)
        offset = float(row.getConstant())
        ignored_row = False
        for column, coefficient in zip(row.getCols(), row.getVals(), strict=True):
            variable = column.getVar()
            index = by_exact_name.get(variable.name)
            if index is None:
                index = by_public_name.get(variable.name.removeprefix("t_"))
            if index is None:
                continue
            if index in ignored_variables:
                ignored_row = True
                break
            position = core_position.get(index)
            if position is None:
                offset += float(coefficient) * reference[index]
            else:
                coefficients[position] += float(coefficient)
        if ignored_row or not np.any(coefficients):
            continue

        raw_lhs = float(row.getLhs())
        raw_rhs = float(row.getRhs())
        lhs = raw_lhs if math.isfinite(raw_lhs) and abs(raw_lhs) < 0.99 * infinity else -math.inf
        rhs = raw_rhs if math.isfinite(raw_rhs) and abs(raw_rhs) < 0.99 * infinity else math.inf
        if not math.isfinite(lhs) and not math.isfinite(rhs):
            continue
        minimum = offset + float(
            np.sum(np.where(coefficients >= 0.0, coefficients * lower, coefficients * upper))
        )
        maximum = offset + float(
            np.sum(np.where(coefficients >= 0.0, coefficients * upper, coefficients * lower))
        )
        if minimum >= lhs - 1e-9 and maximum <= rhs + 1e-9:
            continue
        scale = max(
            1.0,
            abs(offset),
            abs(lhs) if math.isfinite(lhs) else 0.0,
            abs(rhs) if math.isfinite(rhs) else 0.0,
            float(np.linalg.norm(coefficients, ord=1)),
        )
        rounded_activity = offset + float(coefficients @ rounded)
        rounded_violation = max(
            lhs - rounded_activity if math.isfinite(lhs) else 0.0,
            rounded_activity - rhs if math.isfinite(rhs) else 0.0,
            0.0,
        )
        lp_activity = offset + float(coefficients @ state.lp_values[selected_indices])
        slack = min(
            lp_activity - lhs if math.isfinite(lhs) else math.inf,
            rhs - lp_activity if math.isfinite(rhs) else math.inf,
        )
        domain_effect = float(np.abs(coefficients) @ (upper - lower))
        priority = rounded_violation / scale + domain_effect / (scale + max(0.0, slack))
        candidates.append((priority, coefficients, offset, lhs, rhs, scale))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_lp_rows]
    if not selected:
        return (
            np.empty((0, size), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.stack([item[1] for item in selected]),
        np.asarray([item[2] for item in selected], dtype=np.float64),
        np.asarray([item[3] for item in selected], dtype=np.float64),
        np.asarray([item[4] for item in selected], dtype=np.float64),
        np.asarray([item[5] for item in selected], dtype=np.float64),
    )


def _restricted_algebraic_rows(
    algebraic: AlgebraicModel,
    state: SCIPState,
    selection: CoreSelection,
    selected_indices: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if any(not constraint.expression.is_linear for constraint in algebraic.constraints):
        return None
    by_name = {name: index for index, name in enumerate(algebraic.variable_names)}
    original_indices = [by_name.get(state.names[index]) for index in selected_indices]
    if any(index is None for index in original_indices):
        return None
    original = np.asarray(original_indices, dtype=np.int64)
    base = _fallback_point(algebraic)
    state_to_original = np.asarray([by_name.get(name, -1) for name in state.names], dtype=np.int64)
    mapped = state_to_original >= 0
    reference = state.incumbent_values if state.incumbent_values is not None else state.lp_values
    base[state_to_original[mapped]] = reference[mapped]
    fixed_map = state_to_original[selection.fixed_indices]
    fixed_mapped = fixed_map >= 0
    base[fixed_map[fixed_mapped]] = selection.fixed_values[fixed_mapped]
    rounded = np.minimum(
        np.maximum(np.rint(state.lp_values[selected_indices]), lower),
        upper,
    )
    candidates: list[tuple[float, np.ndarray, float, float, float, float]] = []
    for constraint in algebraic.constraints:
        expression = constraint.expression
        coefficients = expression.linear[:, original].toarray().reshape(-1)
        if not np.any(coefficients):
            continue
        base_activity = expression.value(base)
        offset = base_activity - float(coefficients @ base[original])
        lhs = float(constraint.lower)
        rhs = float(constraint.upper)
        minimum = offset + float(
            np.sum(np.where(coefficients >= 0.0, coefficients * lower, coefficients * upper))
        )
        maximum = offset + float(
            np.sum(np.where(coefficients >= 0.0, coefficients * upper, coefficients * lower))
        )
        if minimum >= lhs - 1e-9 and maximum <= rhs + 1e-9:
            continue
        scale = max(
            1.0,
            abs(offset),
            abs(lhs) if math.isfinite(lhs) else 0.0,
            abs(rhs) if math.isfinite(rhs) else 0.0,
            float(np.linalg.norm(coefficients, ord=1)),
        )
        rounded_activity = offset + float(coefficients @ rounded)
        rounded_violation = max(
            lhs - rounded_activity if math.isfinite(lhs) else 0.0,
            rounded_activity - rhs if math.isfinite(rhs) else 0.0,
            0.0,
        )
        domain_effect = float(np.abs(coefficients) @ (upper - lower))
        priority = rounded_violation / scale + domain_effect / scale
        candidates.append((priority, coefficients, offset, lhs, rhs, scale))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_rows]
    size = len(selected_indices)
    if not selected:
        return (
            np.empty((0, size), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.stack([item[1] for item in selected]),
        np.asarray([item[2] for item in selected], dtype=np.float64),
        np.asarray([item[3] for item in selected], dtype=np.float64),
        np.asarray([item[4] for item in selected], dtype=np.float64),
        np.asarray([item[5] for item in selected], dtype=np.float64),
    )


def build_core_surrogate(
    model,
    state: SCIPState,
    selection: CoreSelection,
    positions: list[int],
    *,
    algebraic: AlgebraicModel | None = None,
    max_lp_rows: int = 128,
) -> CoreSurrogate:
    """Restrict the original objective and active LP rows to one QQA core."""
    if isinstance(max_lp_rows, bool) or not isinstance(max_lp_rows, int) or max_lp_rows < 0:
        raise ValueError("max_lp_rows must be a non-negative integer.")
    selected_indices = selection.core_indices[positions]
    lower = selection.local_lower[positions]
    upper = selection.local_upper[positions]
    quadratic, linear, _, source = _restricted_objective(
        state, selection, selected_indices, algebraic
    )
    span = np.maximum(1.0, upper - lower)
    objective_scale = max(
        1e-9,
        float(np.abs(linear) @ span),
        0.5 * float(np.sum(np.abs(quadratic) * np.outer(span, span))),
    )
    algebraic_names = set(algebraic.variable_names) if algebraic is not None else set()
    ignored_variables = (
        {
            index
            for index, (name, variable) in enumerate(zip(state.names, state.variables, strict=True))
            if name not in algebraic_names and abs(float(variable.getObj())) > 0
        }
        if algebraic is not None
        else set()
    )
    algebraic_rows = (
        _restricted_algebraic_rows(
            algebraic,
            state,
            selection,
            selected_indices,
            lower,
            upper,
            max_rows=max_lp_rows,
        )
        if algebraic is not None and algebraic.source_format == "qplib"
        else None
    )
    if algebraic_rows is None:
        algebraic_rows = _restricted_lp_rows(
            model,
            state,
            selection,
            selected_indices,
            lower,
            upper,
            max_lp_rows=max_lp_rows,
            ignored_variables=ignored_variables,
        )
    row_matrix, row_offset, row_lower, row_upper, row_scale = algebraic_rows
    return CoreSurrogate(
        quadratic=quadratic,
        linear=linear,
        objective_scale=objective_scale,
        row_matrix=row_matrix,
        row_offset=row_offset,
        row_lower=row_lower,
        row_upper=row_upper,
        row_scale=row_scale,
        objective_source=source,
    )


__all__ = [
    "CoreSurrogate",
    "build_core_surrogate",
    "generate_surrogate_candidates",
]
