from __future__ import annotations

import torch

from qqa.compile import SparseQUBO
from qqa.learned import (
    DiscreteDiffusionGenerator,
    FactorGraphWarmStart,
    OnlineSolverSelector,
    factor_graph_warm_start,
    model_features,
    model_to_factor_graph,
)
from qqa.model import ConstraintIR, LinearFactor, ModelIR, ObjectiveIR, VariableBlock


def _model() -> ModelIR:
    return ModelIR(
        (VariableBlock("x", "binary", (3,)),),
        ObjectiveIR((LinearFactor(torch.arange(3), torch.tensor([-2.0, 1.0, 0.5])),)),
        (
            ConstraintIR(
                "at-most-two",
                ObjectiveIR((LinearFactor(torch.arange(3), torch.ones(3)),)),
                "<=",
                2.0,
            ),
        ),
    )


def test_factor_graph_and_message_passing_shapes() -> None:
    graph = model_to_factor_graph(_model())
    assert graph.variable_features.shape == (3, 8)
    assert graph.factor_features.shape == (2, 4)
    assert graph.incidence_edge_index.shape == (2, 6)
    network = FactorGraphWarmStart()
    assert network(graph).shape == (3,)


def test_factor_graph_warm_start_is_binary_and_deterministic() -> None:
    first = factor_graph_warm_start(_model(), steps=5, hidden_size=8, seed=7)
    second = factor_graph_warm_start(_model(), steps=5, hidden_size=8, seed=7)
    assert torch.equal(first, second)
    assert bool(((first == 0) | (first == 1)).all())


def test_online_selector_learns_backend_reward() -> None:
    features = model_features(_model())
    selector = OnlineSolverSelector(("qqa", "exact"))
    assert selector.select(features) == "qqa"
    selector.update("exact", features, 10.0)
    assert selector.select(features) == "exact"
    assert selector.observations == {"qqa": 0, "exact": 1}


def test_discrete_diffusion_returns_sorted_polished_population() -> None:
    qubo = SparseQUBO(
        torch.tensor([-1.0, -0.5, 0.25]),
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([1.5, -0.75]),
    )
    result = DiscreteDiffusionGenerator(qubo).generate(12, steps=8, seed=3)
    assert result.population.shape == (12, 3)
    assert bool(((result.population == 0) | (result.population == 1)).all())
    assert bool(torch.all(result.objectives[:-1] <= result.objectives[1:]))
