"""Tests for opt-in portfolio, black-box, uncertainty, and schedule extensions."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

import qqa
from qqa.blackbox import BlackBoxProblem, EvaluationDatabase
from qqa.hybrid.neighborhood_portfolio import NeighborhoodBudget, NeighborhoodPortfolio
from qqa.model import LinearFactor
from qqa.presolve.scip_bridge import SCIPState
from qqa.schedule import make_schedule
from qqa.uncertainty import ScenarioFactor


@pytest.mark.parametrize(
    "name", ["linear", "cosine", "exponential", "sigmoid", "polynomial", "adaptive"]
)
def test_schedule_endpoints_are_inclusive(name: str):
    schedule = make_schedule(name, minimum=-2.0, maximum=0.5)
    assert schedule(0, 11) == pytest.approx(-2.0, abs=1e-6)
    assert schedule(10, 11) == pytest.approx(0.5, abs=1e-6)


def test_adaptive_schedule_reheats_only_after_observation():
    schedule = make_schedule("adaptive", minimum=-2.0, maximum=0.5)
    before = schedule(5, 11)
    schedule.observe(improved=False, diversity_ratio=0.0)
    assert schedule(5, 11) < before
    assert schedule(10, 11) == pytest.approx(0.5)


def _state() -> SCIPState:
    return SCIPState(
        variables=(object(), object(), object()),
        names=("x", "y", "z"),
        variable_types=("BINARY", "INTEGER", "BINARY"),
        lp_values=np.asarray([0.5, 10.4, 0.1]),
        incumbent_values=np.asarray([1.0, 8.0, 0.0]),
        local_lower=np.asarray([0.0, -100.0, 0.0]),
        local_upper=np.asarray([1.0, 100.0, 1.0]),
        reduced_costs=np.asarray([0.1, 3.0, 0.2]),
        pseudocosts=np.asarray([0.2, 4.0, 0.1]),
        node_number=1,
        depth=2,
    )


def test_neighborhood_portfolio_records_bandit_outcomes():
    portfolio = NeighborhoodPortfolio()
    names = []
    for _ in range(4):
        name, neighborhood = portfolio.propose(_state(), NeighborhoodBudget(max_variables=2))
        names.append(name)
        assert len(neighborhood.core_indices) == 2
        portfolio.update(name, runtime=0.1, feasible=True, accepted=True, objective_gain=1.0)
    assert set(names) == {"rens", "rins", "pseudocost", "reduced-cost"}
    assert all(row["calls"] == 1 for row in portfolio.diagnostics().values())


def test_blackbox_cache_and_multi_trust_region_resume(tmp_path: Path):
    calls = {"count": 0}

    def objective(point):
        calls["count"] += 1
        return (point["x"] - 0.25) ** 2 + (point["n"] - 2) ** 2

    problem = BlackBoxProblem(
        [qqa.Real("x", -1.0, 1.0), qqa.Integer("n", 0, 3)],
        objective,
        name="cache-test",
    )
    database = EvaluationDatabase(tmp_path / "evaluations.sqlite")
    first = problem.solve(
        budget=8,
        initial_points=4,
        batch_size=2,
        candidate_pool=32,
        evaluation_database=database,
        trust_regions=2,
        surrogate="rff",
        rff_features=16,
        seed=4,
    )
    initial_calls = calls["count"]
    second = problem.solve(
        budget=10,
        batch_size=2,
        candidate_pool=32,
        evaluation_database=database,
        resume_from=first,
        trust_regions=2,
        surrogate="rff",
        rff_features=16,
        seed=4,
    )
    assert second.evaluations == 10
    assert calls["count"] - initial_calls <= 2
    assert second.metadata["evaluation_cache"] is True
    assert second.metadata["trust_regions"] == 2


def test_scenario_mean_worst_and_cvar_ordering():
    scenarios = (
        LinearFactor(torch.tensor([0]), torch.tensor([1.0])),
        LinearFactor(torch.tensor([0]), torch.tensor([3.0])),
    )
    values = torch.tensor([[2.0]])
    mean = ScenarioFactor(scenarios, "mean").evaluate(values)
    worst = ScenarioFactor(scenarios, "worst").evaluate(values)
    cvar = ScenarioFactor(scenarios, "cvar", cvar_alpha=0.5).evaluate(values)
    assert mean.item() == pytest.approx(4.0)
    assert worst.item() == pytest.approx(6.0)
    assert cvar.item() == pytest.approx(6.0)
    assert math.isfinite(cvar.item())
