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


def test_realistic_application_builders_and_feasible_dispatch():
    model = qqa.build_microgrid_dispatch()
    point = model.pack(
        {
            "commit": [1, 1, 1, 1],
            "power": [80, 60, 20, 5],
            "storage_units": 1,
            "storage_mw": 5,
            "demand_response": 0,
        }
    )
    score = model.score_summary(point)
    assert score["feasible"]
    assert score["value"] > 0
    assert qqa.build_microgrid_pareto().num_objectives == 3
    assert isinstance(qqa.build_process_blackbox(), qqa.BlackBoxProblem)


def test_chunked_dominance_and_pareto_decision_support():
    from qqa.multiobjective.solver import ParetoResult, nondominated_mask

    values = torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [3.0, 3.0]])
    assert nondominated_mask(values, chunk_size=2).tolist() == [True, True, True, False]
    result = ParetoResult(
        solutions=torch.arange(3.0).unsqueeze(1),
        objectives=values[:3],
        weights=torch.ones(3, 2) / 2,
        runtime=0.0,
        objective_names=("cost", "risk"),
        directions=("min", "min"),
    )
    assert result.select() == 1
    assert result.hypervolume([4.0, 4.0]) == pytest.approx(6.0)
    assert result.select([1, 0]) == 0


def test_blackbox_can_resume_without_repeating_evaluations():
    calls = []

    def objective(point):
        calls.append(point["x"])
        return (point["x"] - 0.35) ** 2

    problem = qqa.BlackBoxProblem([qqa.Real("x", 0.0, 1.0)], objective)
    first = problem.solve(budget=12, batch_size=3, candidate_pool=64, seed=5)
    count = len(calls)
    resumed = problem.solve(
        budget=18,
        batch_size=3,
        candidate_pool=64,
        seed=5,
        resume_from=first,
    )
    assert len(calls) - count == 6
    assert resumed.evaluations == 18
    assert resumed.metadata["resumed"] is True
    assert resumed.metadata["cumulative_runtime"] >= resumed.runtime
    assert resumed.best_value <= first.best_value + 1e-12


def test_blackbox_resume_rejects_a_different_problem():
    first_problem = qqa.BlackBoxProblem(
        [qqa.Real("x", 0.0, 1.0)],
        lambda point: point["x"],
        name="first",
    )
    campaign = first_problem.solve(budget=8, batch_size=2, candidate_pool=64)
    different_problem = qqa.BlackBoxProblem(
        [qqa.Real("x", 0.0, 1.0)],
        lambda point: -point["x"],
        direction="max",
        name="different",
    )
    with pytest.raises(ValueError, match="different black-box problem"):
        different_problem.solve(
            budget=10,
            batch_size=2,
            candidate_pool=64,
            resume_from=campaign,
        )


def test_scip_solves_safe_mixed_nonlinear_model_exactly():
    pytest.importorskip("pyscipopt")
    spec = qqa.ModelSpec.from_dict(
        {
            "name": "mixed-scip-test",
            "variables": [
                {"name": "open", "kind": "binary", "lower": 0, "upper": 1, "size": 1},
                {"name": "lots", "kind": "integer", "lower": 0, "upper": 8, "size": 1},
                {"name": "slack", "kind": "real", "lower": 0, "upper": 5, "size": 1},
            ],
            "objectives": [
                {
                    "name": "cost",
                    "direction": "min",
                    "expression": "20*open + 3*lots + square(slack)",
                    "unit": "USD",
                }
            ],
            "constraints": [
                {
                    "name": "demand",
                    "expression": "2*lots + slack",
                    "sense": ">=",
                    "rhs": 11,
                    "weight": 1000,
                    "scale": 11,
                    "tolerance": 1e-5,
                },
                {
                    "name": "link",
                    "expression": "lots - 8*open",
                    "sense": "<=",
                    "rhs": 0,
                    "weight": 1000,
                    "scale": 8,
                    "tolerance": 1e-5,
                },
            ],
            "notes": "",
        }
    )
    result = qqa.solve_spec_scip(
        spec,
        qqa_kwargs={"sol_size": 32, "num_epochs": 20, "verbose": False},
        time_limit=10,
    )
    assert result.proven_optimal
    assert result.score["feasible"]
    assert result.gap == pytest.approx(0.0)
    assert result.objective_value == pytest.approx(36.0)
