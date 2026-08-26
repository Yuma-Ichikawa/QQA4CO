"""Adapters from legacy/public model types into :class:`qqa.ModelIR`."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import torch

from qqa.algebraic import AlgebraicModel, SparseQuadratic, VariableType
from qqa.compile import SparseQUBO, compile_sparse_qubo
from qqa.model.ir import (
    BlackBoxFactor,
    ConstraintIR,
    Factor,
    LinearFactor,
    ModelIR,
    ModelMetadata,
    ObjectiveIR,
    ObjectiveSense,
    QuadraticEdgeFactor,
    VariableBlock,
    VariableDomain,
)


def _sparse_expression(expression: SparseQuadratic) -> ObjectiveIR:
    factors: list[Factor] = []
    linear = cast(Any, expression.linear).tocoo()
    if linear.nnz:
        factors.append(
            LinearFactor(
                torch.tensor(linear.col, dtype=torch.long),
                torch.tensor(linear.data, dtype=torch.float64),
            )
        )
    quadratic = expression.quadratic.tocoo()
    if quadratic.nnz:
        # SparseQuadratic uses 0.5*x.T@H@x.  Retaining every COO entry and
        # halving its coefficient preserves diagonal and non-symmetric parser
        # round-off without a dense intermediate.
        factors.append(
            QuadraticEdgeFactor(
                torch.tensor(np.vstack((quadratic.row, quadratic.col)), dtype=torch.long),
                0.5 * torch.tensor(quadratic.data, dtype=torch.float64),
            )
        )
    return ObjectiveIR(tuple(factors), constant=expression.constant)


def _constraint_scale(expression: SparseQuadratic, rhs: float) -> float:
    """Estimate a stable row scale without constructing a dense matrix."""
    candidates = [1.0, abs(float(rhs)), abs(float(expression.constant))]
    linear = cast(Any, expression.linear)
    quadratic = expression.quadratic
    if linear.nnz:
        candidates.append(float(np.max(np.abs(linear.data))))
        candidates.append(float(np.sum(np.abs(linear.data))))
    if quadratic.nnz:
        # The algebraic convention is 0.5*x'Hx.
        candidates.append(0.5 * float(np.max(np.abs(quadratic.data))))
        candidates.append(0.5 * float(np.sum(np.abs(quadratic.data))))
    return max(candidates)


def algebraic_to_model_ir(model: AlgebraicModel) -> ModelIR:
    """Convert MPS/QPLIB algebraic models while retaining original ordering."""
    domain_map = {
        VariableType.BINARY: VariableDomain.BINARY,
        VariableType.INTEGER: VariableDomain.INTEGER,
        VariableType.IMPLICIT_INTEGER: VariableDomain.INTEGER,
        VariableType.CONTINUOUS: VariableDomain.REAL,
    }
    variables = tuple(
        VariableBlock(
            name=name,
            domain=domain_map[VariableType(kind)],
            lower=float(model.lower_bounds[index]),
            upper=float(model.upper_bounds[index]),
        )
        for index, (name, kind) in enumerate(
            zip(model.variable_names, model.variable_types, strict=True)
        )
    )
    constraints = []
    for row in model.constraints:
        expression = _sparse_expression(row.expression)
        if math.isfinite(row.lower) and math.isfinite(row.upper) and row.lower == row.upper:
            constraints.append(
                ConstraintIR(
                    row.name,
                    expression,
                    "==",
                    row.lower,
                    scale=_constraint_scale(row.expression, row.lower),
                )
            )
            continue
        if math.isfinite(row.upper):
            constraints.append(
                ConstraintIR(
                    row.name if not math.isfinite(row.lower) else f"{row.name}__upper",
                    expression,
                    "<=",
                    row.upper,
                    scale=_constraint_scale(row.expression, row.upper),
                )
            )
        if math.isfinite(row.lower):
            constraints.append(
                ConstraintIR(
                    row.name if not math.isfinite(row.upper) else f"{row.name}__lower",
                    expression,
                    ">=",
                    row.lower,
                    scale=_constraint_scale(row.expression, row.lower),
                )
            )
    return ModelIR(
        variables,
        _sparse_expression(model.objective),
        tuple(constraints),
        ObjectiveSense(model.objective_sense),
        ModelMetadata(
            name=model.name,
            problem_class=model.problem_type,
            source_format=model.source_format,
            attributes=dict(model.metadata),
        ),
    )


def problem_to_model_ir(problem: Any) -> ModelIR:
    """Adapt sparse QUBO and callable COProblem objects to the common IR."""
    if isinstance(problem, ModelIR):
        return problem
    if isinstance(problem, AlgebraicModel):
        return algebraic_to_model_ir(problem)
    try:
        qubo = compile_sparse_qubo(problem)
    except (TypeError, ValueError, AttributeError):
        qubo = None
    if isinstance(qubo, SparseQUBO):
        return ModelIR(
            (
                VariableBlock(
                    "x",
                    VariableDomain.BINARY,
                    shape=(qubo.num_variables,),
                    lower=0.0,
                    upper=1.0,
                ),
            ),
            qubo.objective_ir(),
            sense=ObjectiveSense.MINIMIZE,
            metadata=ModelMetadata(
                name=getattr(problem, "name", type(problem).__name__),
                problem_class=type(problem).__name__,
            ),
        )
    relaxation = getattr(problem, "relaxation", None)
    from qqa.relaxation import (  # noqa: PLC0415
        BinaryRelaxation,
        CategoricalRelaxation,
        SinkhornRelaxation,
        SpinRelaxation,
    )

    if isinstance(relaxation, CategoricalRelaxation):
        num_nodes = getattr(problem, "num_node", getattr(problem, "num_nodes", None))
        categories = getattr(problem, "num_category", None)
        if (
            not isinstance(num_nodes, int)
            or num_nodes < 1
            or not isinstance(categories, int)
            or categories < 2
        ):
            raise TypeError("Categorical problems must expose positive num_node/num_category.")

        def structured_objective(values: torch.Tensor) -> torch.Tensor:
            result = problem.loss_fn(values)
            if result.shape != values.shape[:-2]:
                raise ValueError(
                    "Categorical loss_fn must preserve dimensions before nodes/categories."
                )
            return result

        domain = (
            VariableDomain.PERMUTATION
            if isinstance(relaxation, SinkhornRelaxation)
            else VariableDomain.CATEGORICAL
        )
        return ModelIR(
            (
                VariableBlock(
                    "x",
                    domain,
                    shape=(num_nodes,),
                    categories=categories,
                ),
            ),
            ObjectiveIR((BlackBoxFactor(structured_objective, type(problem).__name__),)),
            metadata=ModelMetadata(
                name=getattr(problem, "name", type(problem).__name__),
                problem_class=type(problem).__name__,
            ),
        )

    space = getattr(problem, "space", None)
    declared_variables = getattr(space, "variables", ())
    if declared_variables:
        domain_map = {
            "binary": VariableDomain.BINARY,
            "integer": VariableDomain.INTEGER,
            "real": VariableDomain.REAL,
        }
        blocks = tuple(
            VariableBlock(
                variable.name,
                domain_map[variable.kind],
                shape=(variable.size,),
                lower=variable.lower,
                upper=variable.upper,
            )
            for variable in declared_variables
        )
        return ModelIR(
            blocks,
            ObjectiveIR((BlackBoxFactor(problem.loss_fn, type(problem).__name__),)),
            metadata=ModelMetadata(
                name=getattr(problem, "name", type(problem).__name__),
                problem_class=type(problem).__name__,
            ),
        )

    num_variables = getattr(problem, "num_vars", getattr(problem, "num_nodes", None))
    if not isinstance(num_variables, int) or num_variables < 1:
        raise TypeError(
            "Cannot adapt model: provide ModelIR, AlgebraicModel, SparseQUBO, "
            "or a problem exposing num_vars/num_nodes and loss_fn."
        )

    def objective(values: torch.Tensor) -> torch.Tensor:
        result = problem.loss_fn(values)
        if result.shape != values.shape[:-1]:
            raise ValueError("Legacy loss_fn must preserve leading batch dimensions.")
        return result

    if isinstance(relaxation, SpinRelaxation):
        domain = VariableDomain.SPIN
        lower, upper = -1.0, 1.0
    elif isinstance(relaxation, BinaryRelaxation):
        domain = VariableDomain.BINARY
        lower, upper = 0.0, 1.0
    else:
        domain = VariableDomain.REAL
        lower, upper = 0.0, 1.0
    return ModelIR(
        (
            VariableBlock(
                "x",
                domain,
                shape=(num_variables,),
                lower=lower,
                upper=upper,
            ),
        ),
        ObjectiveIR((BlackBoxFactor(objective, type(problem).__name__),)),
        metadata=ModelMetadata(
            name=getattr(problem, "name", type(problem).__name__),
            problem_class=type(problem).__name__,
        ),
    )


__all__ = ["algebraic_to_model_ir", "problem_to_model_ir"]
