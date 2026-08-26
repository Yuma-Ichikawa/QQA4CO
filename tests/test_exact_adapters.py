"""Contract tests against the real optional exact-solver Python APIs."""

from __future__ import annotations

import numpy as np
import pytest

from qqa.algebraic import AlgebraicConstraint, AlgebraicModel, SparseQuadratic, VariableType
from qqa.hybrid.exact import solve_exact_algebraic


def _binary_max_model() -> AlgebraicModel:
    objective = SparseQuadratic.linear_expression([3.0, 2.0], constant=5.0)
    capacity = AlgebraicConstraint(
        "capacity",
        SparseQuadratic.linear_expression([1.0, 1.0]),
        upper=1.0,
    )
    return AlgebraicModel(
        "binary-max",
        ("x", "y"),
        (VariableType.BINARY, VariableType.BINARY),
        (0.0, 0.0),
        (1.0, 1.0),
        objective,
        (capacity,),
        objective_sense="maximize",
    )


def test_highs_adapter_preserves_maximize_constant_bound_and_gap():
    pytest.importorskip("highspy")
    model = _binary_max_model()
    result = solve_exact_algebraic(model, "highs", time_limit=5.0, threads=1)
    evaluation = model.evaluate(result.best_sol.numpy())
    assert evaluation.feasible
    assert evaluation.objective == pytest.approx(8.0)
    assert result.best_obj == pytest.approx(-8.0)
    assert result.dual_bound == pytest.approx(8.0)
    assert result.gap == pytest.approx(0.0)


def test_cpsat_adapter_preserves_maximize_objective_and_bound():
    pytest.importorskip("ortools")
    model = _binary_max_model()
    result = solve_exact_algebraic(
        model,
        "cpsat",
        time_limit=5.0,
        threads=1,
        warm_start=np_to_tensor([0.0, 1.0]),
    )
    evaluation = model.evaluate(result.best_sol.numpy())
    assert evaluation.feasible
    assert evaluation.objective == pytest.approx(8.0)
    assert result.best_obj == pytest.approx(-8.0)
    assert result.dual_bound == pytest.approx(8.0)
    assert result.gap == pytest.approx(0.0)


def np_to_tensor(values: list[float]):
    import torch

    return torch.as_tensor(np.asarray(values), dtype=torch.float64)


def test_cpsat_rejects_fractional_coefficients_instead_of_rounding():
    pytest.importorskip("ortools")
    model = AlgebraicModel(
        "fractional",
        ("x",),
        (VariableType.INTEGER,),
        (0.0,),
        (2.0,),
        SparseQuadratic.linear_expression([0.5]),
    )
    with pytest.raises(ValueError, match="integral objective coefficients"):
        solve_exact_algebraic(model, "cpsat", time_limit=1.0)
