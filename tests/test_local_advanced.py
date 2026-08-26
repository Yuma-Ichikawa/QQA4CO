from __future__ import annotations

import torch

from qqa.compile import SparseQUBO
from qqa.local import (
    iterated_local_search,
    k_flip_search,
    kempe_coloring_search,
    maxcut_fm_search,
    mis_swap_search,
    path_relink,
    tabu_search,
    three_opt_tour,
    two_opt_tour,
    walksat_search,
)


def _qubo() -> SparseQUBO:
    return SparseQUBO(
        linear=torch.tensor([-1.0, -1.0, -0.5, -0.25]),
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]]),
        edge_weight=torch.tensor([2.0, 0.75, 1.5, 1.0]),
    )


def test_advanced_qubo_searches_never_worsen_incumbent() -> None:
    qubo = _qubo()
    start = torch.ones(4)
    initial = float(qubo.energy(start).item())
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    results = (
        tabu_search(qubo, start, iterations=40),
        k_flip_search(qubo, start),
        path_relink(qubo, start, target),
        iterated_local_search(qubo, start, restarts=5),
    )
    assert all(result.objective <= initial + 1e-7 for result in results)
    assert all(result.solution.shape == start.shape for result in results)


def test_tour_searches_preserve_permutation_and_never_worsen() -> None:
    coordinates = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    distance = torch.cdist(coordinates, coordinates)
    crossing = torch.tensor([0, 2, 1, 3])
    initial = float(distance[crossing, torch.roll(crossing, -1)].sum().item())
    for result in (two_opt_tour(distance, crossing), three_opt_tour(distance, crossing)):
        assert torch.equal(torch.sort(result.solution).values, torch.arange(4))
        assert result.objective <= initial + 1e-7


def test_graph_and_sat_refinements_produce_valid_or_better_solutions() -> None:
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    weights = torch.ones(4)
    cut = maxcut_fm_search(edges, weights, torch.zeros(4))
    assert cut.objective <= -4.0

    mis = mis_swap_search(edges, torch.zeros(4))
    assert not bool((mis.solution[edges[0]] * mis.solution[edges[1]]).any())
    assert mis.solution.sum() >= 2

    colouring = kempe_coloring_search(edges, torch.zeros(4, dtype=torch.long), num_colors=2)
    assert colouring.objective == 0.0

    clauses = torch.tensor([[0, 1], [0, 2], [1, 2]])
    signs = torch.tensor([[1, 1], [-1, 1], [1, -1]])
    sat = walksat_search(clauses, signs, torch.zeros(3), max_flips=100, seed=4)
    assert sat.objective == 0.0
