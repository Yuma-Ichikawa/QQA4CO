from __future__ import annotations

import json
from types import SimpleNamespace

import networkx as nx
import pytest

import qqa
from qqa.api import _legacy_status
from qqa.result import GuaranteeLevel, SolveResult, SolveStatus


@pytest.mark.parametrize(
    ("backend_status", "has_incumbent", "expected"),
    [
        ("infeasible", False, SolveStatus.INFEASIBLE_PROVEN),
        ("inforunbd", False, SolveStatus.INFEASIBLE_OR_UNBOUNDED),
        ("time limit reached", False, SolveStatus.LIMIT_REACHED_NO_INCUMBENT),
        ("time_limit", True, SolveStatus.LIMIT_REACHED_WITH_INCUMBENT),
        ("model invalid", False, SolveStatus.MODEL_INVALID),
    ],
)
def test_backend_termination_is_normalised_without_false_claims(
    backend_status: str,
    has_incumbent: bool,
    expected: SolveStatus,
) -> None:
    result = SimpleNamespace(
        scip_status=backend_status,
        best_sol=object() if has_incumbent else None,
        diagnostics={},
    )
    assert _legacy_status(result, feasible=has_incumbent)[0] is expected


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
    assert result.guarantee_level in {
        GuaranteeLevel.VERIFIED_FEASIBLE,
        GuaranteeLevel.HEURISTIC,
    }
    assert result.solution.shape == (6,)
    assert result.runtime >= 0
    assert result.provenance.backend
    payload = result.to_dict(include_solutions=True)
    assert payload["objective_value"] == result.objective_value
    assert payload["guarantee_level"] == result.guarantee_level.value
    json.dumps(payload)


def test_solve_plan_is_a_qqa_centred_budgeted_dag() -> None:
    plan = qqa.plan(
        qqa.MaxCut(nx.cycle_graph(8)),
        profile="certify",
        exact_backend="scip",
        replicas=8,
        epochs=2,
    )
    primal = plan.stage("qqa-primal")
    certificate = plan.stage("certificate")
    assert primal is not None and primal.role == "population-primal-search"
    assert certificate is not None and "qqa-primal" in certificate.depends_on
    assert sum(stage.budget_fraction for stage in plan.stages) == pytest.approx(1.0)


def test_solve_session_exposes_plan_result_and_events() -> None:
    session = qqa.SolveSession(
        qqa.MaxCut(nx.cycle_graph(4)),
        profile="fast",
        exact_backend="none",
        replicas=4,
        epochs=1,
        polish=False,
    )
    assert session.plan().stage("qqa-primal") is not None
    result = session.run()
    assert session.result is result
    assert session.summary()["state"] == "completed"
    assert session.summary()["event_count"] == len(result.events)
