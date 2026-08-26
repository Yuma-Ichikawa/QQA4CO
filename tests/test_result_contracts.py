from __future__ import annotations

import json

import networkx as nx
import pytest

import qqa
from qqa.result import SolveResult, SolveStatus


@pytest.mark.parametrize("backend", ["qqa", "sa", "pa", "isco"])
def test_stable_solve_adapts_every_heuristic_backend_to_one_contract(backend: str) -> None:
    problem = qqa.MaxCut(nx.cycle_graph(6))
    result = qqa.solve(
        problem,
        profile="fast",
        backend=backend,
        exact_backend="none",
        replicas=8,
        epochs=4,
        polish=False,
        seed=9,
    )
    assert isinstance(result, SolveResult)
    assert result.status in {SolveStatus.FEASIBLE, SolveStatus.UNKNOWN}
    assert result.solution.shape == (6,)
    assert result.runtime >= 0
    assert result.provenance.backend
    payload = result.to_dict(include_solutions=True)
    assert payload["objective_value"] == result.objective_value
    json.dumps(payload)
