"""Opt-in binary, sparse-simplex, and mirror-descent relaxation tests."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import qqa


class _CategoricalShape:
    num_node = 3
    num_category = 4


def test_straight_through_binary_is_hard_with_finite_surrogate_gradient() -> None:
    relaxation = qqa.StraightThroughBinaryRelaxation(temperature=0.7)
    logits = torch.tensor([[-2.0, 0.0, 2.0]], requires_grad=True)
    values = relaxation.forward(logits)
    assert set(values.detach().flatten().tolist()) <= {0.0, 1.0}
    values.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad > 0)


def test_stochastic_binary_uses_bernoulli_forward_and_deterministic_projection() -> None:
    torch.manual_seed(5)
    relaxation = qqa.StochasticBinaryRelaxation(temperature=1.0)
    logits = torch.zeros((64, 4), requires_grad=True)
    samples = relaxation.forward(logits)
    assert set(samples.detach().flatten().tolist()) == {0.0, 1.0}
    assert torch.equal(relaxation.project(logits), torch.ones_like(logits))


@pytest.mark.parametrize("mapping", ["sparsemax", "entmax15"])
def test_sparse_categorical_maps_to_simplex_with_finite_gradients(mapping: str) -> None:
    relaxation = qqa.SparseCategoricalRelaxation(mapping=mapping)
    logits = torch.tensor([[[4.0, 1.0, -2.0, -3.0]]], requires_grad=True)
    probabilities = relaxation.forward(logits)
    assert probabilities.sum(dim=-1).item() == pytest.approx(1.0, abs=1e-6)
    assert torch.count_nonzero(probabilities).item() < probabilities.numel()
    probabilities.square().sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_categorical_temperature_annealing_is_endpoint_inclusive() -> None:
    relaxation = qqa.EntropicCategoricalRelaxation(
        temperature=2.0,
        final_temperature=0.125,
    )
    relaxation.set_progress(0.0)
    assert relaxation.temperature == pytest.approx(2.0)
    relaxation.set_progress(0.5)
    assert relaxation.temperature == pytest.approx(0.5)
    relaxation.set_progress(1.0)
    assert relaxation.temperature == pytest.approx(0.125)


def test_mirror_descent_preserves_simplex_and_runs_through_annealer() -> None:
    problem = qqa.Coloring(nx.cycle_graph(6), num_category=3)
    problem.relaxation = qqa.MirrorDescentCategoricalRelaxation()
    result = qqa.anneal(
        problem,
        sol_size=8,
        num_epochs=4,
        learning_rate=0.05,
        optimizer="mirror-descent",
        polish=False,
        verbose=False,
        return_population=True,
    )
    assert result.final_population is not None
    assert torch.all(result.final_population.sum(dim=-1) == 1)
    assert result.diagnostics["optimizer"] == "mirror-descent"


def test_mirror_descent_rejects_incompatible_relaxation() -> None:
    problem = qqa.MaximumIndependentSet(nx.path_graph(4))
    with pytest.raises(TypeError, match="mirror_step"):
        qqa.anneal(
            problem,
            num_epochs=0,
            optimizer="mirror-descent",
            verbose=False,
        )
