from __future__ import annotations

import numpy as np
import torch

from qqa.hybrid.heuristic_runtime import (
    build_initial_population,
    rank_repair_candidates,
    solve_core_problem,
)
from qqa.hybrid.heuristic_types import QQAHeuristicConfig
from qqa.utils import resolve_device


def test_initial_population_is_bounded_deduplicated_and_deterministic():
    target = np.array([0.25, 1.75])
    lower = np.array([0.0, 1.0])
    upper = np.array([1.0, 2.0])
    seeds = [target, target.copy(), np.array([-3.0, 9.0])]

    first = build_initial_population(
        seeds,
        target=target,
        lower=lower,
        upper=upper,
        sol_size=8,
        seed=7,
    )
    second = build_initial_population(
        seeds,
        target=target,
        lower=lower,
        upper=upper,
        sol_size=8,
        seed=7,
    )

    assert first.shape == (8, 2)
    assert np.array_equal(first, second)
    assert np.array_equal(first[0], target)
    assert np.array_equal(first[1], np.array([0.0, 2.0]))
    assert np.all(first >= lower)
    assert np.all(first <= upper)


def test_repair_candidate_losses_are_evaluated_as_one_batch():
    class CountingProblem:
        def __init__(self):
            self.calls = 0

        def loss_fn(self, values):
            self.calls += 1
            tensor = torch.as_tensor(values, dtype=torch.float64)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return tensor.square().sum(dim=1)

    problem = CountingProblem()
    ranked = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    def selector(_problem, _reference, candidate, positions, **_kwargs):
        return [positions[0]] if candidate[0] else []

    reordered, plans = rank_repair_candidates(
        problem,
        ranked,
        reference=np.zeros(2),
        positions=[4, 9],
        lower=np.zeros(2),
        upper=np.ones(2),
        max_changes=1,
        beam_width=2,
        selector=selector,
    )

    assert problem.calls == 1
    assert np.array_equal(reordered[0], ranked[0])
    assert plans[np.array([1, 0], dtype=np.int64).tobytes()] == [4]


def test_core_solve_resolves_and_forwards_requested_device():
    class RecordingProblem:
        constraints = ()

        def __init__(self):
            self.options = None

        def solve(self, **options):
            self.options = options
            return "result"

    problem = RecordingProblem()
    config = QQAHeuristicConfig(
        device="auto",
        sol_size=2,
        epochs=4,
        threads=torch.get_num_threads(),
    )
    initial = np.zeros((2, 3), dtype=np.float64)

    result = solve_core_problem(problem, initial, config, seed=11, time_limit=0.5)

    assert result == "result"
    assert problem.options is not None
    assert problem.options["device"] == resolve_device("auto")
    assert problem.options["time_limit"] == 0.5
    assert problem.options["initial_state"].shape == (2, 3)
    assert problem.options["initial_state"].dtype == torch.float64
