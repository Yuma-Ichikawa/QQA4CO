"""SCIP hybrid, Pareto, black-box, and population result tests."""

from __future__ import annotations

import itertools

import networkx as nx
import pytest
import torch

import qqa


def test_anneal_optionally_returns_projected_population():
    problem = qqa.MaxCut(nx.cycle_graph(5))
    compact = qqa.anneal(problem, sol_size=8, num_epochs=0, verbose=False)
    rich = qqa.anneal(
        problem,
        sol_size=8,
        num_epochs=0,
        return_population=True,
        verbose=False,
    )
    assert compact.final_population is None
    assert rich.final_population.shape == (8, 5)
    assert set(rich.final_population.unique().tolist()) <= {0.0, 1.0}


def test_one_shot_pareto_finds_integer_tradeoff_curve():
    problem = qqa.MultiObjectiveProblem(
        [qqa.Integer("x", 0, 10)],
        [
            qqa.Objective(lambda v: v["x"].square(), "cost"),
            qqa.Objective(lambda v: (v["x"] - 10).square(), "delay"),
        ],
    )
    result = problem.solve_pareto(sol_size=256, num_epochs=30, seed=2)
    points = result.named_solutions(problem)["x"].round().to(torch.int64)
    assert {0, 10} <= set(points.tolist())
    assert len(set(points.tolist())) >= 9
    assert result.objectives.shape[1] == 2

    figure = qqa.plot_pareto(result, backend="matplotlib", show=False)
    assert figure[0] is not None


def test_nondominated_filter_respects_maximisation_direction():
    problem = qqa.MultiObjectiveProblem(
        [qqa.Integer("x", 0, 4)],
        [
            qqa.Objective(lambda v: v["x"], "reward", direction="max"),
            qqa.Objective(lambda v: v["x"].square(), "risk", direction="min"),
        ],
    )
    result = problem.solve_pareto(sol_size=128, num_epochs=10, seed=4)
    x = result.named_solutions(problem)["x"]
    assert {0, 4} <= set(x.round().to(torch.int64).tolist())

    with pytest.raises(TypeError, match="solve_pareto"):
        qqa.anneal(problem, num_epochs=0, verbose=False)


def test_blackbox_mixed_optimizer_and_parallel_batches():
    calls = []

    def objective(point):
        calls.append((point["enabled"], point["units"], point["ratio"]))
        return (
            5 * (point["enabled"] - 1) ** 2
            + (point["units"] - 3) ** 2
            + (point["ratio"] - 0.2) ** 2
        )

    problem = qqa.BlackBoxProblem(
        [
            qqa.Binary("enabled"),
            qqa.Integer("units", 0, 6),
            qqa.Real("ratio", 0.0, 1.0),
        ],
        objective,
    )
    result = problem.solve(
        budget=60,
        batch_size=4,
        workers=2,
        candidate_pool=512,
        seed=3,
    )
    assert result.best_point["enabled"] == 1
    assert result.best_point["units"] == 3
    assert result.best_point["ratio"] == pytest.approx(0.2, abs=0.08)
    assert result.best_value < 0.01
    assert result.evaluations == len(calls) <= 60
    figure = qqa.plot_blackbox(result, backend="matplotlib", show=False)
    assert len(figure[1]) == 3


def test_blackbox_constraints_use_feasibility_first():
    problem = qqa.BlackBoxProblem(
        [qqa.Integer("x", 0, 8)],
        lambda point: point["x"],
        direction="max",
        constraints=[
            qqa.BlackBoxConstraint(
                lambda point: point["x"],
                sense="<=",
                rhs=5,
                name="capacity",
            )
        ],
    )
    result = problem.solve(budget=20, batch_size=2, candidate_pool=128, seed=1)
    assert result.feasible
    assert result.best_point["x"] == 5
    assert result.best_value == 5


def test_qqa_scip_certifies_qubo_optimum():
    pytest.importorskip("pyscipopt")
    graph = nx.cycle_graph(7)
    problem = qqa.MaxCut(graph)
    brute = min(
        float(problem.loss_fn(torch.tensor(bits, dtype=torch.float32).unsqueeze(0))[0])
        for bits in itertools.product((0.0, 1.0), repeat=7)
    )
    result = qqa.solve_qqa_scip(
        problem,
        qqa_kwargs={"sol_size": 32, "num_epochs": 10, "verbose": False},
        time_limit=10,
    )
    assert result.best_obj == pytest.approx(brute)
    assert result.proven_optimal
    assert result.gap == pytest.approx(0.0)
    assert result.n_warm_starts >= 1
    assert result.history is result.qqa_result.history
