"""Regression tests for the Phase 0--5 typed runtime contracts."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest
import torch

import qqa
from qqa.algebraic import AlgebraicConstraint, AlgebraicModel, SparseQuadratic, VariableType
from qqa.compile import SparseQUBO
from qqa.dual import BasisStatus, crossover_lp, solve_lp_relaxation
from qqa.exact import solve_cp_model_ir
from qqa.gpu import (
    assignment_repair,
    binary_flip_delta,
    compile_factor_graph,
    exact_k_repair,
    one_hot_repair,
)
from qqa.learned import OODGate
from qqa.model import (
    BlackBoxFactor,
    CardinalityFactor,
    ClauseFactor,
    ConstraintIR,
    LinearFactor,
    ModelIR,
    NoOverlapFactor,
    ObjectiveIR,
    VariableBlock,
)
from qqa.model.capabilities import inspect_capabilities
from qqa.model.problem import ModelIRProblem
from qqa.polish import greedy_spin_flip
from qqa.presolve import general_qpbo_persistency
from qqa.repair import repair_model_ir
from qqa.runtime import Checkpoint, load_checkpoint, save_checkpoint, validate_portable_payload
from qqa.service import JobManager, ServicePolicy
from qqa.visuals import decision_explorer, plot_optimization_cockpit


def test_capability_contract_rejects_declared_nondifferentiable_black_box() -> None:
    model = ModelIR(
        (VariableBlock("x", "binary", (2,)),),
        ObjectiveIR((BlackBoxFactor(lambda x: x.sum(dim=-1), differentiable=False),)),
    )
    report = inspect_capabilities(model)
    assert not report.qqa_compatible
    with pytest.raises(NotImplementedError, match="non-differentiable"):
        ModelIRProblem(model)


def test_model_doctor_and_strict_bounds_never_invent_a_box() -> None:
    model = ModelIR(
        (VariableBlock("x", "real", (1,)),),
        ObjectiveIR((LinearFactor(torch.tensor([0]), torch.tensor([1.0])),)),
    )
    report = qqa.doctor(model)
    assert not report.ready
    assert report.capabilities.missing_bounds == ("x",)
    with pytest.raises(NotImplementedError, match="missing/non-finite"):
        ModelIRProblem(model)


def test_duration_budget_and_goal_are_resolved_by_one_call_api() -> None:
    model = qqa.MaxCut(__import__("networkx").cycle_graph(4))
    planned = qqa.plan(model, goal="diverse", budget="250ms", replicas=4, epochs=1)
    assert planned.profile == "diverse"


def test_compiled_factor_graph_matches_typed_expression() -> None:
    objective = ObjectiveIR(
        (
            LinearFactor(torch.tensor([0, 1]), torch.tensor([2.0, -1.0])),
            CardinalityFactor(torch.tensor([0, 1, 2]), 2.0, 0.5),
            ClauseFactor(
                torch.tensor([[0, 2], [1, 2]]),
                torch.tensor([[1, -1], [-1, 1]]),
                torch.tensor([3.0, 4.0]),
            ),
        ),
        constant=1.25,
    )
    model = ModelIR((VariableBlock("x", "binary", (3,)),), objective)
    compiled = compile_factor_graph(model)
    values = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    assert torch.allclose(compiled.evaluate(values), objective.evaluate(values))


def test_device_resident_repairs_and_flip_delta() -> None:
    values = torch.tensor([[0.1, 0.9, 0.7, 0.2]])
    repaired = exact_k_repair(values, torch.arange(4), 2)
    assert repaired.sum().item() == 2
    one_hot = one_hot_repair(values.reshape(1, 2, 2))
    assert torch.equal(one_hot.sum(dim=-1), torch.ones((1, 2)))
    assignment = assignment_repair(torch.tensor([[[4.0, 1.0], [3.0, 2.0]]]))
    assert torch.equal(assignment.sum(dim=-1), torch.ones((1, 2)))
    assert torch.equal(assignment.sum(dim=-2), torch.ones((1, 2)))
    qubo = SparseQUBO(
        torch.tensor([1.0, -2.0]),
        torch.tensor([[0], [1]]),
        torch.tensor([0.5]),
    )
    state = torch.tensor([[0.0, 1.0]])
    delta = binary_flip_delta(qubo, state)
    for index in range(2):
        flipped = state.clone()
        flipped[:, index] = 1 - flipped[:, index]
        assert delta[:, index].item() == pytest.approx(
            (qubo.energy(flipped) - qubo.energy(state)).item()
        )


def test_model_ir_repair_resolves_assignment_linking_constraints() -> None:
    model = qqa.build_template(
        "facility-location",
        opening_costs=[4.0, 5.0, 3.0],
        assignment_costs=[[1.0, 4.0, 2.0], [3.0, 1.0, 2.0]],
    )
    repaired = repair_model_ir(model, torch.zeros(model.num_variables))
    violations = model.constraint_violations(repaired)
    assert all(float(value.item()) <= 1e-9 for value in violations.values())

    result = qqa.solve(model, profile="fast", replicas=4, epochs=1, exact_backend="none")
    figure, axes = plot_optimization_cockpit(result, backend="matplotlib")
    labels = [label.get_text() for label in axes[1, 0].get_yticklabels()]
    assert labels and max(map(len, labels)) <= 28
    assert "SolveStatus" not in axes[1, 1].texts[0].get_text()
    figure.clear()


def test_spin_polish_traverses_neutral_domain_wall_plateau() -> None:
    problem = qqa.Ising1D(N=64, J=1.0, h=0.0, periodic=True)
    local_minimum = torch.ones(64)
    local_minimum[16:32] = -1
    assert problem.loss_fn(local_minimum.unsqueeze(0)).item() == pytest.approx(-60.0)
    polished = greedy_spin_flip(problem, local_minimum)
    assert polished is not None
    assert problem.loss_fn(polished.unsqueeze(0)).item() == pytest.approx(-64.0)


def test_pdhg_returns_primal_and_valid_relaxation_bound() -> None:
    model = AlgebraicModel(
        "small-lp",
        ("x", "y"),
        (VariableType.CONTINUOUS, VariableType.CONTINUOUS),
        np.zeros(2),
        np.ones(2),
        SparseQuadratic.linear_expression([1.0, 1.0]),
        (
            AlgebraicConstraint(
                "demand",
                SparseQuadratic.linear_expression([1.0, 1.0]),
                lower=1.0,
            ),
        ),
    )
    result = solve_lp_relaxation(model, max_iterations=3000, tolerance=1e-5)
    assert result.primal_residual < 1e-3
    assert result.primal_objective == pytest.approx(1.0, abs=2e-3)
    assert result.dual_bound is not None
    # The returned ergodic primal is still slightly infeasible at this short
    # iteration budget, so it is not an upper bound. The dual remains a valid
    # lower bound on the known optimum.
    assert result.dual_bound <= 1.0 + 1e-8

    crossover = crossover_lp(model, result, time_limit=5)
    assert crossover.proven_optimal
    assert crossover.objective == pytest.approx(1.0)
    assert crossover.maximum_infeasibility <= 1e-7
    assert any(status is BasisStatus.AT_LOWER for status in crossover.variable_status)


def test_general_qpbo_exact_probe_matches_bruteforce() -> None:
    qubo = SparseQUBO(
        torch.tensor([-1.0, 0.25, -0.2]),
        torch.tensor([[0, 0, 1], [1, 2, 2]]),
        torch.tensor([2.0, -0.5, 0.75]),
    )
    result = general_qpbo_persistency(qubo, exact_component_limit=8)
    states = torch.tensor(
        [[(identifier >> index) & 1 for index in range(3)] for identifier in range(8)],
        dtype=torch.float32,
    )
    optimum = qubo.energy(states).min().item()
    assert result.exact
    assert result.optimum_or_lower_bound == pytest.approx(optimum)


def test_cp_scheduling_runtime_enforces_no_overlap() -> None:
    pytest.importorskip("ortools.sat.python.cp_model")
    starts = torch.arange(3)
    model = ModelIR(
        (VariableBlock("start", "integer", (3,), 0, 8),),
        ObjectiveIR((LinearFactor(starts, torch.ones(3)),)),
        (
            ConstraintIR(
                "machine",
                ObjectiveIR((NoOverlapFactor(starts, torch.tensor([2.0, 2.0, 2.0])),)),
                "<=",
                0.0,
            ),
        ),
    )
    result = solve_cp_model_ir(model, time_limit=5)
    assert result.proven
    ordered = sorted(result.best_sol.tolist())
    assert all(right - left >= 2 for left, right in zip(ordered, ordered[1:], strict=False))


def test_checkpoint_is_pickle_free_and_checksum_protected(tmp_path) -> None:
    target = tmp_path / "state.qqacp"
    save_checkpoint(Checkpoint("abc", {"profile": "fast"}, 3, {"x": torch.arange(3)}, {}), target)
    loaded = load_checkpoint(target)
    assert torch.equal(loaded.tensors["x"], torch.arange(3))
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(target, "a") as bundle,
    ):
        bundle.writestr("tensors/x.npy", b"tampered")
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        load_checkpoint(target)


def test_anneal_checkpoint_can_resume_without_embedding_its_path(tmp_path) -> None:
    target = tmp_path / "resume.qqacp"
    problem = qqa.MaxCut(__import__("networkx").cycle_graph(5))
    qqa.anneal(
        problem,
        sol_size=4,
        num_epochs=2,
        optimizer="lightweight-adamw",
        checkpoint_path=str(target),
        checkpoint_interval=1,
        polish=False,
        verbose=False,
    )
    resumed = qqa.anneal(
        problem,
        sol_size=4,
        num_epochs=4,
        optimizer="lightweight-adamw",
        resume_from=str(target),
        polish=False,
        verbose=False,
    )
    assert resumed.diagnostics["resumed"] is True
    assert resumed.diagnostics["completed_epochs"] == 4


def test_untrusted_python_and_private_metadata_are_fail_closed() -> None:
    with pytest.raises(PermissionError):
        qqa.user_problem_from_source("def loss_fn(x): return x.sum(-1)", num_vars=2)
    with pytest.raises(ValueError, match="local path"):
        validate_portable_payload({"note": "/private/machine/path"})
    with pytest.raises(ValueError, match="sensitive"):
        validate_portable_payload({"api_key": "not-a-real-key"})


def test_service_policy_rejects_code_and_unknown_options() -> None:
    manager = JobManager(policy=ServicePolicy(maximum_variables=2), workers=1)
    payload = {
        "variables": [{"name": "x", "domain": "binary", "shape": [2]}],
        "objective": {"factors": []},
    }
    try:
        with pytest.raises(ValueError, match="not allowed"):
            manager.submit(payload, {"python_source": "raise SystemExit"})
    finally:
        manager.shutdown(wait=False)


def test_ood_gate_handles_one_feature_and_decision_explorer_counterfactuals() -> None:
    gate = OODGate(torch.tensor([[0.0], [0.5], [1.0]]))
    assert torch.isfinite(gate.score(torch.tensor([0.25])))
    model = ModelIR(
        (VariableBlock("x", "binary", (2,)),),
        ObjectiveIR((LinearFactor(torch.arange(2), torch.tensor([1.0, 2.0])),)),
    )
    result = qqa.solve(model, profile="fast", replicas=4, epochs=1, exact_backend="none")
    rows = decision_explorer(result, model)
    assert len(rows) == 2
    assert all(row["objective_delta"] is not None for row in rows)
