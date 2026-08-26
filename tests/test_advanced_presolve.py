from __future__ import annotations

import pytest
import torch

from qqa.compile import SparseQUBO
from qqa.model import (
    ConstraintIR,
    LinearFactor,
    ModelIR,
    ObjectiveIR,
    VariableBlock,
    presolve_model,
)
from qqa.presolve import (
    detect_qubo_symmetries,
    dominance_fixings,
    exact_probe_persistency,
    submodular_roof_duality,
)


def test_singleton_bounds_are_tightened_and_fixed_variables_restore() -> None:
    model = ModelIR(
        (VariableBlock("x", "binary", (3,)),),
        ObjectiveIR((LinearFactor(torch.arange(3), torch.tensor([1.0, 2.0, 3.0])),)),
        (
            ConstraintIR(
                "fix-zero",
                ObjectiveIR((LinearFactor(torch.tensor([0]), torch.tensor([1.0])),)),
                "<=",
                0.0,
            ),
            ConstraintIR(
                "fix-one",
                ObjectiveIR((LinearFactor(torch.tensor([1]), torch.tensor([1.0])),)),
                ">=",
                1.0,
            ),
        ),
    )
    reduced = presolve_model(model)
    assert reduced.report.tightened_bounds == 2
    assert reduced.report.fixed_variables == 2
    restored = reduced.restore(torch.tensor([1.0]))
    assert restored.tolist() == [0.0, 1.0, 1.0]


def test_dominance_and_exact_probing_return_only_strong_fixings() -> None:
    qubo = SparseQUBO(
        torch.tensor([-2.0, 1.0, 0.0]),
        torch.empty((2, 0), dtype=torch.long),
        torch.empty(0),
    )
    assert dominance_fixings(qubo) == {0: 1, 1: 0}
    result = exact_probe_persistency(qubo)
    assert result.fixings == {0: 1, 1: 0}
    assert result.optimum_or_lower_bound == pytest.approx(-2.0)


def test_submodular_roof_duality_and_symmetry_are_exact() -> None:
    qubo = SparseQUBO(
        torch.tensor([1.0, 1.0]),
        torch.tensor([[0], [1]]),
        torch.tensor([-3.0]),
    )
    result = submodular_roof_duality(qubo)
    assert result.fixings == {0: 1, 1: 1}
    assert result.optimum_or_lower_bound == pytest.approx(-1.0)
    assert detect_qubo_symmetries(qubo) == ((0, 1),)

    nonsubmodular = SparseQUBO(
        torch.zeros(2),
        torch.tensor([[0], [1]]),
        torch.ones(1),
    )
    with pytest.raises(ValueError, match="non-positive"):
        submodular_roof_duality(nonsubmodular)
