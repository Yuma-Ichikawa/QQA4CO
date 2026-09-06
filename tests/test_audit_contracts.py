"""Regression tests for solver-boundary and runtime integrity contracts."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import qqa
from qqa.algebraic import AlgebraicModel, SparseQuadratic, VariableType
from qqa.annealing import (
    AnnealResult,
    _apply_optimizer_step_scale_,
    _replica_median,
    _reset_optimizer_rows,
)
from qqa.hybrid.exact import ExactBackendResult
from qqa.model import (
    BlackBoxFactor,
    CardinalityFactor,
    ClauseFactor,
    HigherOrderFactor,
    LinearFactor,
    ModelIR,
    ObjectiveIR,
    QuadraticEdgeFactor,
    VariableBlock,
    presolve_model,
)
from qqa.model.problem import ModelIRProblem
from qqa.result import (
    ConstraintReport,
    Provenance,
    ResourceReport,
    SolveResult,
    SolveStatus,
    TimingReport,
)
from qqa.runtime.events import EventKind, EventRecorder


@pytest.mark.parametrize(
    ("domain", "candidate"),
    [
        ("binary", [0.25]),
        ("binary", [2.0]),
        ("binary", [float("nan")]),
        ("spin", [0.0]),
        ("integer", [1.5]),
    ],
)
def test_feasibility_checks_variable_domains(domain: str, candidate: list[float]) -> None:
    bounds = {} if domain in {"binary", "spin"} else {"lower": 0.0, "upper": 2.0}
    model = ModelIR((VariableBlock("x", domain, (1,), **bounds),), ObjectiveIR(()))
    assert not bool(model.feasible(torch.tensor(candidate)).item())


def test_original_model_verifier_rejects_nonfinite_objective() -> None:
    model = ModelIR(
        (VariableBlock("x", "binary", (1,)),),
        ObjectiveIR(
            (
                BlackBoxFactor(
                    lambda values: torch.full(
                        values.shape[:-1],
                        float("nan"),
                        device=values.device,
                        dtype=values.dtype,
                    )
                ),
            )
        ),
    )
    verification = model.verify_solution(torch.tensor([[0.0], [1.0]]))
    assert verification.objective_values.dtype is torch.float64
    assert not verification.objective_finite.any()
    assert not verification.feasible.any()


def test_model_ir_problem_accepts_per_coordinate_bounds() -> None:
    model = ModelIR(
        (
            VariableBlock(
                "x",
                "integer",
                (3,),
                torch.tensor([0.0, -4.0, 10.0]),
                torch.tensor([1.0, 2.0, 20.0]),
            ),
        ),
        ObjectiveIR(()),
    )
    problem = ModelIRProblem(model)
    projected = problem.relaxation.project(torch.tensor([[0.0, 0.5, 1.0]]))
    torch.testing.assert_close(projected, torch.tensor([[0.0, -1.0, 20.0]]))


def test_structural_presolve_preserves_sparse_builtin_factors() -> None:
    model = ModelIR(
        (
            VariableBlock("fixed", "binary", (1,), 1.0, 1.0),
            VariableBlock("free", "binary", (4,)),
        ),
        ObjectiveIR(
            (
                LinearFactor(torch.tensor([0, 1]), torch.tensor([2.0, 3.0])),
                QuadraticEdgeFactor(
                    torch.tensor([[0, 1, 2], [2, 2, 3]]), torch.tensor([4.0, 5.0, 6.0])
                ),
                HigherOrderFactor(torch.tensor([[0, 1, 4]]), torch.tensor([7.0])),
                CardinalityFactor(torch.tensor([0, 1, 2]), 2.0, 0.5),
                ClauseFactor(
                    torch.tensor([[0, 1], [2, 3]]),
                    torch.tensor([[1, -1], [1, 1]]),
                    torch.tensor([2.0, 3.0]),
                ),
            ),
            1.25,
        ),
    )
    reduced = presolve_model(model)
    assert all(
        type(factor).__name__ != "EmbeddedFactor" for factor in reduced.model.objective.factors
    )
    candidates = torch.randint(0, 2, (16, 4), dtype=torch.float32)
    restored = reduced.restore(candidates)
    torch.testing.assert_close(
        reduced.model.objective_values(candidates), model.objective_values(restored)
    )


def test_repair_is_retained_but_cannot_replace_a_better_verified_candidate(monkeypatch) -> None:
    model = ModelIR(
        (VariableBlock("x", "binary", (2,)),),
        ObjectiveIR((LinearFactor(torch.arange(2), torch.ones(2)),)),
    )

    def solve_stub(problem, **kwargs):
        del kwargs
        solution = torch.zeros(problem.model_ir.num_variables)
        return AnnealResult(solution, 0.0, 0.0)

    monkeypatch.setattr(importlib.import_module("qqa.model.solve"), "solve_model_ir", solve_stub)
    monkeypatch.setattr(
        importlib.import_module("qqa.repair"),
        "repair_model_ir",
        lambda _model, _candidate, **_kwargs: torch.ones(2),
    )
    result = qqa.solve(model, exact_backend="none", replicas=2, epochs=1, polish=False)
    assert result.selected_candidate_id == "raw"
    assert torch.equal(result.solution, torch.zeros(2))
    assert result.repaired_objective_value == pytest.approx(2.0)
    assert [candidate.candidate_id for candidate in result.candidates if candidate.selected] == [
        "raw"
    ]


def test_model_ir_repair_receives_remaining_api_budget(monkeypatch) -> None:
    model = ModelIR((VariableBlock("x", "binary", (1,)),), ObjectiveIR(()))
    observed: list[float | None] = []

    def solve_stub(problem, **kwargs):
        del kwargs
        return AnnealResult(torch.zeros(problem.model_ir.num_variables), 0.0, 0.0)

    def repair_stub(_model, candidate, *, time_limit=None):
        observed.append(time_limit)
        return candidate

    monkeypatch.setattr(importlib.import_module("qqa.model.solve"), "solve_model_ir", solve_stub)
    monkeypatch.setattr(importlib.import_module("qqa.repair"), "repair_model_ir", repair_stub)
    qqa.solve(
        model,
        exact_backend="none",
        replicas=2,
        epochs=1,
        polish=False,
        budget=1.0,
    )
    assert len(observed) == 1
    assert observed[0] is not None
    assert 0.0 <= observed[0] <= 1.0


def test_cuda_graph_api_override_disables_only_implicit_replica_exchange() -> None:
    api = importlib.import_module("qqa.api")
    resolved = api._resolve_config(
        profile="fast",
        budget=None,
        device="cuda",
        config=None,
        overrides={"cuda_graphs": True},
    )
    assert resolved.cuda_graphs is True
    assert resolved.heterogeneous_replicas is True
    assert resolved.replica_exchange_interval is None

    with pytest.raises(ValueError, match="cuda_graphs"):
        api._resolve_config(
            profile="fast",
            budget=None,
            device="cuda",
            config=None,
            overrides={"cuda_graphs": True, "replica_exchange_interval": 10},
        )


def test_adam_role_scale_changes_the_actual_update() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2, 1))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.0)
    origin = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    _apply_optimizer_step_scale_(parameter, origin, torch.tensor([[0.5], [2.0]]))
    delta = (parameter.detach() - origin).abs().reshape(-1)
    assert delta[1] == pytest.approx(4.0 * delta[0])


def test_restart_clears_selected_optimizer_rows_in_place() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3, 2))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.state[parameter] = {
        "exp_avg": torch.ones_like(parameter),
        "exp_avg_sq": torch.full_like(parameter, 2.0),
    }
    _reset_optimizer_rows(optimizer, parameter, torch.tensor([0, 2]))
    for state, retained in zip(optimizer.state[parameter].values(), (1.0, 2.0), strict=True):
        torch.testing.assert_close(state[0], torch.zeros(2))
        torch.testing.assert_close(state[1], state.new_full((2,), retained))
        torch.testing.assert_close(state[2], torch.zeros(2))


def test_replica_median_preserves_lower_median_semantics() -> None:
    values = torch.tensor([[4.0, 1.0], [2.0, 8.0], [3.0, 5.0], [1.0, 7.0]])
    torch.testing.assert_close(_replica_median(values), values.median(dim=0).values)


def test_convexification_uses_objective_scale_and_dimension() -> None:
    from qqa.runtime.population import estimate_convexification_beta

    problem = qqa.MaximumIndependentSet(__import__("networkx").path_graph(4), penalty=2.0)
    base = estimate_convexification_beta(problem, objective_scale=1.0, dimensions=4)
    doubled_scale = estimate_convexification_beta(problem, objective_scale=2.0, dimensions=4)
    doubled_dimension = estimate_convexification_beta(problem, objective_scale=1.0, dimensions=8)
    assert doubled_scale == pytest.approx(base / 2.0)
    assert doubled_dimension == pytest.approx(base * 2.0)


def test_event_recorder_uses_observed_times_and_labels_search_merit(monkeypatch) -> None:
    import qqa.runtime.events as event_module
    from qqa.callbacks import CallbackState

    clock = iter([100.0, 101.0, 105.0])
    monkeypatch.setattr(event_module, "perf_counter", lambda: next(clock))
    recorder = EventRecorder(stride=1)
    state = CallbackState(
        epoch=0,
        num_epochs=100,
        bg=0.0,
        x=torch.zeros(2, 1),
        losses=torch.tensor([2.0, 1.0]),
        penalties=torch.zeros(2),
        diversity=torch.tensor(0.0),
        best_obj=torch.tensor(1.0),
        hyperparams={},
        problem=object(),
        relaxation=object(),
    )
    recorder.on_train_begin(state)
    recorder.on_epoch_end(state)
    state.epoch = 1
    state.best_obj = torch.tensor(0.5)
    recorder.on_epoch_end(state)
    recorder.on_train_end(state)
    merit = [event for event in recorder.events if event.kind is EventKind.SEARCH_MERIT_IMPROVED]
    assert [event.elapsed_seconds for event in merit] == [1.0, 5.0]
    assert all("objective" not in event.payload for event in merit)


def test_cockpit_keeps_unknown_values_unknown() -> None:
    from qqa.visuals import plot_optimization_cockpit

    result = SolveResult(
        SolveStatus.NO_SOLUTION_FOUND,
        None,
        None,
        None,
        None,
        False,
        ConstraintReport.unknown(),
        TimingReport(0.0),
        ResourceReport("cpu"),
        Provenance("qqa", 0, "fast"),
    )
    figure, axes = plot_optimization_cockpit(result, backend="matplotlib")
    assert "unavailable" in axes[1, 1].texts[0].get_text()
    assert "verification unavailable" in axes[1, 0].texts[0].get_text().lower()
    figure.clear()


def _install_exact_stubs(monkeypatch, solution: torch.Tensor):
    api = importlib.import_module("qqa.api")
    hybrid = importlib.import_module("qqa.hybrid")
    ir_solver = importlib.import_module("qqa.model.solve")
    dual = importlib.import_module("qqa.dual")

    def primal(problem, **kwargs):
        del kwargs
        value = torch.zeros(problem.model_ir.num_variables, dtype=torch.float64)
        return AnnealResult(value, 0.0, 0.0, final_population=value.unsqueeze(0))

    def exact(model, backend, **kwargs):
        del backend, kwargs
        evaluation = model.evaluate(solution.numpy())
        return ExactBackendResult(
            solution.clone(),
            evaluation.objective,
            0.0,
            "optimal",
            evaluation.objective,
            0.0,
            {"backend": "test-exact"},
        )

    monkeypatch.setattr(ir_solver, "solve_model_ir", primal)
    monkeypatch.setattr(hybrid, "solve_exact_algebraic", exact)
    monkeypatch.setattr(
        dual,
        "solve_lp_relaxation",
        lambda model, **kwargs: SimpleNamespace(
            primal_solution=solution.clone(),
            dual_bound=None,
            iterations=0,
            runtime=0.0,
            primal_residual=0.0,
            dual_residual=0.0,
            kkt_residual=0.0,
            converged=True,
        ),
    )
    monkeypatch.setattr(
        importlib.import_module("qqa.repair"),
        "repair_model_ir",
        lambda _model, candidate, **_kwargs: candidate,
    )
    return api


def test_exact_incumbent_has_an_explicit_original_coordinate_contract(monkeypatch) -> None:
    solution = torch.tensor([1.0, 0.0], dtype=torch.float64)
    api = _install_exact_stubs(monkeypatch, solution)
    model = AlgebraicModel(
        "fixed-coordinate",
        ("fixed", "free"),
        (VariableType.BINARY, VariableType.BINARY),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        SparseQuadratic.linear_expression([1.0, 1.0]),
        (),
    )
    result = api.solve(model, exact_backend="scip", budget=5.0, device="cpu", replicas=2, epochs=1)
    assert torch.equal(result.raw_solution, solution)
    assert result.diagnostics["incumbent_coordinate_space"] == "original"
    assert result.population is not None and result.population.shape[-1] == 2


def test_exact_only_plan_does_not_claim_a_skipped_qqa_stage() -> None:
    model = AlgebraicModel(
        "unbounded-continuous",
        ("x",),
        (VariableType.CONTINUOUS,),
        np.array([0.0]),
        np.array([np.inf]),
        SparseQuadratic.linear_expression([1.0]),
        (),
    )
    plan = qqa.plan(model, exact_backend="scip", replicas=2, epochs=1)
    assert plan.stage("qqa-primal") is None
    with pytest.raises(RuntimeError, match="require_qqa_primal"):
        qqa.solve(
            model,
            exact_backend="scip",
            require_qqa_primal=True,
            budget=1.0,
            replicas=2,
            epochs=1,
        )
