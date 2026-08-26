"""Mathematical invariants shared across model, sparse, and result layers."""

from __future__ import annotations

import itertools

import pytest
import torch

import qqa
from qqa.compile import SparseQUBO
from qqa.model import (
    ConstraintIR,
    LinearFactor,
    ModelIR,
    ModelMetadata,
    ObjectiveIR,
    QuadraticEdgeFactor,
    VariableBlock,
)


def _bits(size: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product((0.0, 1.0), repeat=size)))


def test_sparse_qubo_dense_energy_and_gradient_parity():
    qubo = SparseQUBO(
        torch.tensor([1.0, -2.0, 0.5, 3.0]),
        torch.tensor([[0, 0, 1, 2], [1, 3, 2, 3]]),
        torch.tensor([0.25, -1.5, 2.0, 0.75]),
        constant=4.0,
    )
    values = torch.rand(7, 4, requires_grad=True)
    sparse_energy = qubo.energy(values)
    dense = qubo.to_dense()
    dense.fill_diagonal_(0.0)
    dense_energy = (
        (values * qubo.linear).sum(dim=-1)
        + torch.einsum("bi,ij,bj->b", values, dense, values)
        + qubo.constant
    )
    assert torch.allclose(sparse_energy, dense_energy, atol=1e-6)
    sparse_gradient = torch.autograd.grad(sparse_energy.sum(), values, retain_graph=True)[0]
    dense_gradient = torch.autograd.grad(dense_energy.sum(), values)[0]
    assert torch.allclose(sparse_gradient, dense_gradient, atol=1e-6)


@pytest.mark.parametrize("sense,expected", [("minimize", 0.0), ("maximize", 6.0)])
def test_objective_sense_is_canonicalized_once(sense, expected):
    model = ModelIR(
        (VariableBlock("x", "binary", (3,)),),
        ObjectiveIR((LinearFactor(torch.arange(3), torch.tensor([1.0, 2.0, 3.0])),)),
        sense=sense,
    )
    candidates = _bits(3)
    selected = torch.argmin(model.internal_energy(candidates))
    assert float(model.objective_values(candidates)[selected]) == expected


def test_variable_permutation_is_metamorphic():
    original = ObjectiveIR(
        (
            LinearFactor(torch.tensor([0, 1, 2]), torch.tensor([1.0, -2.0, 0.5])),
            QuadraticEdgeFactor(torch.tensor([[0, 1], [2, 2]]), torch.tensor([3.0, -1.0])),
        ),
        constant=7.0,
    )
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    permuted = ObjectiveIR(
        (
            LinearFactor(inverse[torch.tensor([0, 1, 2])], torch.tensor([1.0, -2.0, 0.5])),
            QuadraticEdgeFactor(inverse[torch.tensor([[0, 1], [2, 2]])], torch.tensor([3.0, -1.0])),
        ),
        constant=7.0,
    )
    values = torch.rand(12, 3)
    assert torch.allclose(original.evaluate(values), permuted.evaluate(values[:, permutation]))


def test_model_metadata_rejects_private_environment_values():
    with pytest.raises(ValueError, match="Private environment"):
        ModelMetadata(name="unsafe", attributes={"server": "example"})
    with pytest.raises(ValueError, match="Private environment"):
        ModelMetadata(name="unsafe", attributes={"source": "/private/model.mps"})
    with pytest.raises(ValueError, match="Private environment"):
        ModelMetadata(name="unsafe", attributes={"source": "http://localhost/model"})


def test_score_and_repair_never_mutate_tsp_candidate():
    problem = qqa.TSP(N=5, seed=3)
    candidate = torch.zeros(5, 5)
    candidate[:, 0] = 1
    original = candidate.clone()
    problem.score_summary(candidate)
    repaired = problem.repair_solution(candidate)
    assert torch.equal(candidate, original)
    assert repaired.data_ptr() != candidate.data_ptr()
    assert torch.equal(repaired.sum(dim=0), torch.ones(5))
    assert torch.equal(repaired.sum(dim=1), torch.ones(5))


def test_stable_solve_separates_raw_repaired_and_original_objective():
    model = ModelIR(
        (VariableBlock("x", "binary", (4,)),),
        ObjectiveIR((LinearFactor(torch.arange(4), torch.tensor([1.0, 2.0, 3.0, 4.0])),)),
        (
            ConstraintIR(
                "choose-two",
                ObjectiveIR((LinearFactor(torch.arange(4), torch.ones(4)),)),
                "==",
                2.0,
                scale=2.0,
            ),
        ),
    )
    result = qqa.solve(
        model,
        profile="fast",
        replicas=8,
        epochs=5,
        polish=False,
        exact_backend="none",
    )
    assert result.repaired_solution is not None
    assert result.feasible
    assert result.violations.feasible
    assert result.repaired_objective_value == pytest.approx(
        float(model.objective_values(result.solution)[0])
    )
    payload = result.to_dict(include_solutions=True)
    assert payload["raw_solution"] is not None
    assert payload["repaired_solution"] is not None


def test_strict_config_rejects_unknown_options():
    with pytest.raises(TypeError, match="Unknown SolverConfig"):
        qqa.SolverConfig.from_mapping({"epoch": 10})


def test_strict_config_can_explicitly_disable_profile_restarts():
    config = qqa.SolverConfig.for_profile("balanced", restart_patience=0)
    assert config.resolved().restart_patience == 0
    assert config.anneal_kwargs()["restart_patience"] is None


def test_stable_api_never_silently_skips_requested_certification():
    problem = qqa.SherringtonKirkpatrick(N=4, seed=0)
    with pytest.raises(NotImplementedError, match="no requested certificate was silently skipped"):
        qqa.solve(problem, profile="certify", exact_backend="scip", budget=1)
