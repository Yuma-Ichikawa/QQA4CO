"""Extract transformed SCIP state without losing original variable identity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from qqa.algebraic import AlgebraicModel, SparseQuadratic, VariableType


@dataclass(frozen=True, slots=True)
class SCIPVariableMap:
    """Original/transformed variable pairs retained across SCIP presolve."""

    original: tuple[Any, ...]
    transformed: tuple[Any | None, ...]
    names: tuple[str, ...]

    @classmethod
    def from_model(cls, model) -> SCIPVariableMap:
        original = tuple(model.getVars(transformed=False))
        transformed: list[Any | None] = []
        for variable in original:
            try:
                transformed.append(model.getTransformedVar(variable))
            except Exception:
                transformed.append(None)
        return cls(original, tuple(transformed), tuple(variable.name for variable in original))


@dataclass(frozen=True, slots=True)
class SCIPState:
    """Numerical node state consumed by core selection and QQA-LNS."""

    variables: tuple[Any, ...]
    names: tuple[str, ...]
    variable_types: tuple[str, ...]
    lp_values: np.ndarray
    incumbent_values: np.ndarray | None
    local_lower: np.ndarray
    local_upper: np.ndarray
    reduced_costs: np.ndarray
    pseudocosts: np.ndarray
    node_number: int
    depth: int
    interaction_edges: np.ndarray | None = None
    conflict_scores: np.ndarray | None = None
    gradient_scores: np.ndarray | None = None
    historical_scores: np.ndarray | None = None
    reference_history: tuple[np.ndarray, ...] = ()

    @property
    def integer_indices(self) -> np.ndarray:
        return np.fromiter(
            (
                index
                for index, kind in enumerate(self.variable_types)
                if kind in {"BINARY", "INTEGER", "IMPLINT"}
            ),
            dtype=np.int64,
        )


def _safe_float(function, default: float) -> float:
    try:
        value = float(function())
    except Exception:
        return default
    return value if math.isfinite(value) else default


def extract_scip_state(model) -> SCIPState:
    """Read active transformed variables, LP/incumbent, local bounds and costs."""
    variables = tuple(model.getVars(transformed=True))
    best = model.getBestSol()
    lp_values: list[float] = []
    incumbent_values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    reduced: list[float] = []
    pseudo: list[float] = []
    conflict: list[float] = []

    for variable in variables:
        lb = _safe_float(variable.getLbLocal, -model.infinity())
        ub = _safe_float(variable.getUbLocal, model.infinity())
        fallback = min(max(0.0, lb), ub)
        lp = _safe_float(variable.getLPSol, fallback) if variable.isInLP() else fallback
        lp_values.append(min(max(lp, lb), ub))
        lower.append(lb)
        upper.append(ub)
        if best is not None:
            incumbent_values.append(_safe_float(lambda v=variable: model.getSolVal(best, v), lp))
        reduced.append(_safe_float(lambda v=variable: model.getVarRedcost(v), 0.0))
        pseudo.append(_safe_float(lambda v=variable, x=lp: model.getVarPseudocostScore(v, x), 0.0))
        conflict.append(_safe_float(lambda v=variable: model.getVarConflictScore(v), 0.0))

    # Build a compact variable-interaction graph from active constraints.  A
    # large row is represented as a star to keep extraction linear in row
    # width while preserving connectivity for graph-induced neighbourhoods.
    by_name = {variable.name.removeprefix("t_"): index for index, variable in enumerate(variables)}
    edges: set[tuple[int, int]] = set()
    try:
        constraints = tuple(model.getConss())
    except Exception:
        constraints = ()
    for constraint in constraints:
        try:
            row_variables = tuple(model.getConsVars(constraint))
        except Exception:
            try:
                row_variables = tuple(model.getValsLinear(constraint))
            except Exception:
                continue
        indices = sorted(
            {
                by_name[name]
                for variable in row_variables
                if (name := variable.name.removeprefix("t_")) in by_name
            }
        )
        if len(indices) <= 64:
            edges.update(
                (indices[left], indices[right])
                for left in range(len(indices))
                for right in range(left + 1, len(indices))
            )
        elif indices:
            edges.update((indices[0], index) for index in indices[1:])
    interaction_edges = (
        np.asarray(sorted(edges), dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    )

    node = model.getCurrentNode()
    return SCIPState(
        variables=variables,
        names=tuple(variable.name.removeprefix("t_") for variable in variables),
        variable_types=tuple(variable.vtype() for variable in variables),
        lp_values=np.asarray(lp_values, dtype=np.float64),
        incumbent_values=(
            np.asarray(incumbent_values, dtype=np.float64) if best is not None else None
        ),
        local_lower=np.asarray(lower, dtype=np.float64),
        local_upper=np.asarray(upper, dtype=np.float64),
        reduced_costs=np.asarray(reduced, dtype=np.float64),
        pseudocosts=np.asarray(pseudo, dtype=np.float64),
        node_number=int(model.getNNodes()),
        depth=int(node.getDepth()) if node is not None else 0,
        interaction_edges=interaction_edges,
        conflict_scores=np.asarray(conflict, dtype=np.float64),
        gradient_scores=np.abs(np.asarray(reduced, dtype=np.float64)),
    )


def _scip_expression(expression: SparseQuadratic, variables, quicksum):
    # ``SparseQuadratic.__post_init__`` normalises every accepted dense or
    # sequence input to a SciPy sparse matrix.  Keep the public constructor
    # type broad while making that post-init invariant explicit here.
    linear = cast(Any, expression.linear).tocoo()
    result = float(expression.constant) + quicksum(
        float(coefficient) * variables[column]
        for column, coefficient in zip(linear.col, linear.data, strict=True)
    )
    quadratic = expression.quadratic.tocoo()
    upper = np.asarray(quadratic.row)
    columns = np.asarray(quadratic.col)
    values = np.asarray(quadratic.data)
    terms = []
    for row, column, value in zip(upper, columns, values, strict=True):
        if row > column:
            continue
        coefficient = 0.5 * value if row == column else value
        if coefficient != 0:
            terms.append(float(coefficient) * variables[row] * variables[column])
    if terms:
        result = result + quicksum(terms)
    return result


def build_scip_model(
    algebraic: AlgebraicModel,
    *,
    name: str | None = None,
    verbose: bool = False,
):
    """Build a SCIP model from the sparse IR and return it with variables."""
    try:
        from pyscipopt import Model, quicksum
        from pyscipopt.recipes.nonlinear import set_nonlinear_objective
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        raise ImportError("SCIP model building requires `qqa[scip]`.") from exc
    if not isinstance(algebraic, AlgebraicModel):
        raise TypeError("algebraic must be an AlgebraicModel.")
    model = Model(name or algebraic.name)
    if not verbose:
        model.hideOutput()
    infinity = float(model.infinity())

    def finite_bound(value: float) -> float:
        if np.isneginf(value):
            return -infinity
        if np.isposinf(value):
            return infinity
        return float(value)

    type_map = {
        VariableType.CONTINUOUS: "C",
        VariableType.BINARY: "B",
        VariableType.INTEGER: "I",
        VariableType.IMPLICIT_INTEGER: "I",
    }
    variables = tuple(
        model.addVar(
            name=variable_name,
            vtype=type_map[VariableType(variable_type)],
            lb=finite_bound(lower),
            ub=finite_bound(upper),
        )
        for variable_name, variable_type, lower, upper in zip(
            algebraic.variable_names,
            algebraic.variable_types,
            algebraic.lower_bounds,
            algebraic.upper_bounds,
            strict=True,
        )
    )
    objective = _scip_expression(algebraic.objective, variables, quicksum)
    if algebraic.objective.is_linear:
        model.setObjective(objective, algebraic.objective_sense)
    else:
        set_nonlinear_objective(model, objective, algebraic.objective_sense)

    for row in algebraic.constraints:
        expression = _scip_expression(row.expression, variables, quicksum)
        finite_lower = np.isfinite(row.lower)
        finite_upper = np.isfinite(row.upper)
        if finite_lower and finite_upper and abs(row.upper - row.lower) <= 1e-12:
            model.addCons(expression == float(row.lower), name=row.name)
        else:
            if finite_lower:
                name_lower = row.name if not finite_upper else f"{row.name}__lower"
                model.addCons(expression >= float(row.lower), name=name_lower)
            if finite_upper:
                name_upper = row.name if not finite_lower else f"{row.name}__upper"
                model.addCons(expression <= float(row.upper), name=name_upper)
    return model, variables


__all__ = [
    "SCIPState",
    "SCIPVariableMap",
    "build_scip_model",
    "extract_scip_state",
]
