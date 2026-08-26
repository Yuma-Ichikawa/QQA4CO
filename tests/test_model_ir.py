"""Native factors, presolve reversibility, and portable parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import qqa
from qqa.io.formats import load_dimacs, load_ising_text, load_opb, load_qubo_text
from qqa.model import (
    BlackBoxFactor,
    ConstraintIR,
    FlowConservationFactor,
    LinearFactor,
    LogicalFactor,
    ModelIR,
    NoOverlapFactor,
    ObjectiveIR,
    PairwisePottsFactor,
    PiecewiseLinearFactor,
    TableFactor,
    VariableBlock,
    presolve_model,
)


def test_presolve_fixed_variables_restore_original_space():
    model = ModelIR(
        (
            VariableBlock("fixed", "binary", (1,), 1.0, 1.0),
            VariableBlock("free", "binary", (2,), 0.0, 1.0),
        ),
        ObjectiveIR((LinearFactor(torch.arange(3), torch.tensor([5.0, 2.0, 3.0])),)),
        (
            ConstraintIR("always", ObjectiveIR((), constant=0.0), "<=", 1.0),
            ConstraintIR(
                "scaled",
                ObjectiveIR((LinearFactor(torch.tensor([1]), torch.tensor([100.0])),)),
                "<=",
                100.0,
            ),
        ),
    )
    presolved = presolve_model(model)
    assert presolved.report.fixed_variables == 1
    assert presolved.report.removed_empty_constraints == 1
    assert presolved.report.rescaled_constraints == 1
    restored = presolved.restore(torch.tensor([0.0, 1.0]))
    assert restored.tolist() == [1.0, 0.0, 1.0]
    assert torch.allclose(
        model.objective_values(restored),
        presolved.model.objective_values(torch.tensor([0.0, 1.0])),
    )


def test_native_factors_have_expected_discrete_semantics():
    values = torch.tensor([[1.0, 1.0, 0.0, 0.0, 2.0, 1.0]])
    logical = LogicalFactor(torch.tensor([0, 1, 2]), "and")
    assert logical.evaluate(values).item() == 1.0
    table = TableFactor(torch.tensor([0, 1]), torch.tensor([[0.0, 2.0], [3.0, 4.0]]))
    assert table.evaluate(values).item() == 4.0
    piecewise = PiecewiseLinearFactor(
        4, torch.tensor([0.0, 1.0, 3.0]), torch.tensor([0.0, 2.0, 4.0])
    )
    assert piecewise.evaluate(values).item() == pytest.approx(3.0)
    no_overlap = NoOverlapFactor(torch.tensor([4, 5]), torch.tensor([2.0, 2.0]))
    assert no_overlap.evaluate(values).item() == pytest.approx(1.0)
    flow = FlowConservationFactor(
        torch.tensor([0, 1]),
        torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
        torch.tensor([0.0, 0.0]),
    )
    assert flow.evaluate(values).item() == pytest.approx(0.0)


def test_categorical_model_ir_executes_with_automatic_relaxation():
    model = ModelIR(
        (VariableBlock("colour", "categorical", (3,), categories=3),),
        ObjectiveIR(
            (
                PairwisePottsFactor(
                    torch.tensor([[0, 1, 0], [1, 2, 2]]),
                    -torch.ones(3),
                ),
            )
        ),
    )
    result = qqa.solve(model, profile="fast", replicas=8, epochs=10, polish=False)
    assert result.solution.shape == (3, 3)
    assert torch.equal(result.solution.sum(dim=1), torch.ones(3))


def test_permutation_model_ir_repairs_to_a_bijection():
    def diagonal_reward(values: torch.Tensor) -> torch.Tensor:
        return -torch.diagonal(values, dim1=-2, dim2=-1).sum(dim=-1)

    model = ModelIR(
        (VariableBlock("assignment", "permutation", (4,), categories=4),),
        ObjectiveIR((BlackBoxFactor(diagonal_reward, "diagonal-reward"),)),
    )
    initial = torch.zeros(4, 4)
    initial[:, 0] = 1.0
    result = qqa.solve(
        model,
        profile="fast",
        replicas=8,
        epochs=5,
        initial_solution=initial,
        polish=False,
    )
    assert result.solution.shape == (4, 4)
    assert torch.equal(result.solution.sum(dim=0), torch.ones(4))
    assert torch.equal(result.solution.sum(dim=1), torch.ones(4))


def test_constrained_structured_model_uses_shape_preserving_archive():
    def category_zero_count(values: torch.Tensor) -> torch.Tensor:
        return values[..., 0].sum(dim=-1)

    model = ModelIR(
        (VariableBlock("choice", "categorical", (3,), categories=2),),
        ObjectiveIR((BlackBoxFactor(lambda values: -values[..., 1].sum(dim=-1), "reward"),)),
        (
            ConstraintIR(
                "one-category-zero",
                ObjectiveIR((BlackBoxFactor(category_zero_count, "category-zero-count"),)),
                "==",
                1.0,
            ),
        ),
    )
    result = qqa.solve(
        model,
        profile="fast",
        replicas=8,
        epochs=5,
        polish=False,
        return_population=True,
    )
    assert result.solution.shape == (3, 2)
    assert result.population is not None
    assert result.population.shape[-2:] == (3, 2)


def test_legacy_problem_inspection_preserves_spin_and_permutation_domains():
    spin = qqa.inspect(qqa.SherringtonKirkpatrick(N=6, seed=0))
    assert spin.domains == {"spin": 6}
    assert "mixed-domain" not in spin.structure

    permutation = qqa.inspect(qqa.TSP(N=5, seed=0))
    assert permutation.domains == {"permutation": 5}
    assert "assignment" in permutation.structure


def test_portable_formats_parse_small_instances(tmp_path: Path):
    qubo_path = tmp_path / "small.qubo"
    qubo_path.write_text("0 0 -1\n0 1 2\n", encoding="utf-8")
    qubo = load_qubo_text(qubo_path)
    assert qubo.num_nodes == 2

    cnf_path = tmp_path / "small.cnf"
    cnf_path.write_text("c demo\np cnf 2 2\n1 -2 0\n2 0\n", encoding="utf-8")
    cnf = load_dimacs(cnf_path)
    assert cnf.num_variables == 2

    opb_path = tmp_path / "small.opb"
    opb_path.write_text("min: +1 x1 +2 x2;\n+1 x1 +1 x2 >= 1;\n", encoding="utf-8")
    opb = load_opb(opb_path)
    assert opb.num_variables == 2
    assert len(opb.constraints) == 1


@pytest.mark.parametrize(
    "content",
    [
        "p cnf 2 1\n1 2\n",
        "p unsupported 1 1\n1 0\n",
        "1 2 3 4\n",
        "p cnf 2 2\n1 0\n",
        "p cnf 2 1\n3 0\n",
        "p cnf 2 1\np cnf 2 1\n1 0\n",
        "p cnf 2 1\nnot-a-literal 0\n",
    ],
)
def test_dimacs_parser_rejects_malformed_input(tmp_path: Path, content: str):
    path = tmp_path / "bad.cnf"
    path.write_text(content, encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_dimacs(path)


@pytest.mark.parametrize(
    ("loader", "suffix", "content"),
    [
        (load_qubo_text, ".qubo", "-1 -1 2\n"),
        (load_qubo_text, ".qubo", "0 0 nan\n"),
        (load_qubo_text, ".qubo", "0 1\n"),
        (load_ising_text, ".ising", "-1 2\n"),
        (load_ising_text, ".ising", "0 1 inf\n"),
    ],
)
def test_qubo_parser_rejects_unsafe_numeric_input(tmp_path, loader, suffix, content):
    path = tmp_path / f"bad{suffix}"
    path.write_text(content, encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        loader(path)
