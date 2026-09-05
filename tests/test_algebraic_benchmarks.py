"""Sparse algebraic IR and MIPLIB/QPLIB hybrid regression tests."""

from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch
from scipy import sparse

import qqa
from qqa.algebraic import (
    AlgebraicConstraint,
    AlgebraicModel,
    SparseQuadratic,
    VariableType,
)
from qqa.benchmarking import (
    BenchmarkResult,
    compare_benchmark_solvers,
    merge_benchmark_campaigns,
    publish_benchmark_campaigns,
    run_benchmark_instance,
    run_benchmark_suite,
)
from qqa.benchmarking import download as benchmark_download
from qqa.benchmarking.algebraic_runner import (
    _classify_outcome,
    _comparison_solver_order,
    _configure_scip_threads,
    _default_worker_timeout,
    _native_process_error_type,
    _qqa_applicability_hint,
    _qqa_is_applicable,
)
from qqa.benchmarking.cli import _named_paths
from qqa.benchmarking.metrics import (
    BenchmarkFailure,
    IncumbentPoint,
    SCIPProgressTracker,
    normalised_primal_error,
    primal_integral,
    relative_gap,
    summarise_comparison,
)
from qqa.cli import build_parser
from qqa.decomposition import (
    CompletionResult,
    complete_integer_assignment,
    complete_integer_assignment_dive,
    create_completion_template,
)
from qqa.hybrid import scip_heuristic as scip_heuristic_module
from qqa.hybrid.core_selector import CoreSelection, select_uncertain_integer_core
from qqa.hybrid.heuristic_types import QQAHeuristicConfig
from qqa.hybrid.neighborhood import IntegerNeighborhood
from qqa.hybrid.nonconvex import dc_decomposition, linearize_concave_part
from qqa.hybrid.scip_heuristic import QQAHeuristic
from qqa.hybrid.surrogate import build_core_surrogate, generate_surrogate_candidates
from qqa.io import load_mps, load_qplib
from qqa.mixed import ConstraintArchive
from qqa.mixed.augmented_lagrangian import AdaptiveAugmentedLagrangian
from qqa.presolve import SCIPState, build_scip_model, scaled_model
from qqa.presolve.scaling import ScalingFactors


def _algebraic_fixture() -> AlgebraicModel:
    objective = SparseQuadratic(
        sparse.coo_matrix(([2.0, 2.0], ([0, 1], [1, 0])), shape=(3, 3)),
        sparse.csr_matrix(([3.0], ([0], [2])), shape=(1, 3)),
        4.0,
    )
    row = AlgebraicConstraint(
        "capacity",
        SparseQuadratic.linear_expression(
            sparse.csr_matrix(([1.0, -2.0], ([0, 0], [0, 2])), shape=(1, 3))
        ),
        lower=-math.inf,
        upper=2.0,
    )
    return AlgebraicModel(
        name="portable-model",
        variable_names=("binary", "integer", "continuous"),
        variable_types=(VariableType.BINARY, VariableType.INTEGER, VariableType.CONTINUOUS),
        lower_bounds=(0.0, -2.0, -math.inf),
        upper_bounds=(1.0, 4.0, math.inf),
        objective=objective,
        constraints=(row,),
        metadata={"source_name": "portable.qplib", "source_sha256": "abc"},
    )


def test_sparse_algebraic_ir_evaluates_without_dense_row_storage():
    model = _algebraic_fixture()
    assert sparse.isspmatrix_csr(model.objective.linear)
    assert sparse.isspmatrix_csr(model.constraints[0].expression.linear)
    point = np.asarray([1.0, 2.0, 0.5])
    evaluation = model.evaluate(point)
    assert evaluation.objective == pytest.approx(9.5)
    assert evaluation.constraint_values.tolist() == pytest.approx([0.0])
    assert evaluation.maximum_infeasibility == 0.0
    assert model.summary()["constraint_linear_nonzeros"] == 2

    fractional = model.evaluate(np.asarray([0.8, 2.4, 0.5]))
    assert fractional.integrality_violation == pytest.approx(0.4)
    assert fractional.maximum_infeasibility == pytest.approx(0.4)


def test_qqa_static_size_gate_avoids_registering_out_of_scope_plugins():
    algebraic = _algebraic_fixture()
    assert _qqa_is_applicable(
        algebraic,
        QQAHeuristicConfig(maximum_problem_variables=3, minimum_core_size=1),
    )
    assert not _qqa_is_applicable(
        algebraic,
        QQAHeuristicConfig(maximum_problem_variables=2, minimum_core_size=1),
    )
    assert _qqa_is_applicable(
        algebraic,
        QQAHeuristicConfig(maximum_integer_variables=2, minimum_core_size=1),
    )
    assert not _qqa_is_applicable(
        algebraic,
        QQAHeuristicConfig(maximum_integer_variables=1, minimum_core_size=1),
    )
    assert not _qqa_is_applicable(
        algebraic,
        QQAHeuristicConfig(minimum_core_size=3),
    )
    qplib = replace(algebraic, problem_type="QML")
    assert _qqa_is_applicable(
        qplib,
        QQAHeuristicConfig(allowed_qplib_problem_types=("qml",), minimum_core_size=1),
    )
    assert not _qqa_is_applicable(
        qplib,
        QQAHeuristicConfig(allowed_qplib_problem_types=("LIQ",), minimum_core_size=1),
    )
    with pytest.raises(ValueError, match="valid three-character"):
        QQAHeuristicConfig(allowed_qplib_problem_types=("QXX",))
    with pytest.raises(ValueError, match="minimum_runtime_startup_time"):
        QQAHeuristicConfig(minimum_runtime_startup_time=-1.0)


def test_linear_rows_share_one_structural_zero_hessian():
    first = SparseQuadratic.linear_expression(np.ones(10_000))
    second = SparseQuadratic.linear_expression(np.arange(10_000))
    assert first.quadratic is second.quadratic
    assert first.quadratic.nnz == 0
    assert first.quadratic.indptr.nbytes < 100_000


def test_benchmark_thread_limit_includes_the_lp_solver():
    pyscipopt = pytest.importorskip("pyscipopt")
    model = pyscipopt.Model()
    _configure_scip_threads(model, 1)
    assert model.getParam("parallel/maxnthreads") == 1
    assert model.getParam("lp/threads") == 1


def test_paired_benchmark_balances_solver_execution_order():
    solvers = ("scip-aggressive", "sg-cqqa")
    assert (
        _comparison_solver_order(
            solvers,
            execution_order="balanced",
            seed=0,
            instance_index=0,
        )
        == solvers
    )
    assert _comparison_solver_order(
        solvers,
        execution_order="balanced",
        seed=0,
        instance_index=1,
    ) == tuple(reversed(solvers))
    assert _comparison_solver_order(
        solvers,
        execution_order="balanced",
        seed=1,
        instance_index=0,
    ) == tuple(reversed(solvers))


def test_balanced_solver_order_is_shard_invariant_and_seed_symmetric():
    solvers = ("scip-aggressive", "sg-cqqa")
    first = _comparison_solver_order(
        solvers,
        execution_order="balanced",
        seed=0,
        instance_index=0,
        instance_name="portable-instance.mps.gz",
    )
    assert (
        _comparison_solver_order(
            solvers,
            execution_order="balanced",
            seed=0,
            instance_index=999,
            instance_name="portable-instance.mps.gz",
        )
        == first
    )
    assert _comparison_solver_order(
        solvers,
        execution_order="balanced",
        seed=1,
        instance_index=0,
        instance_name="portable-instance.mps.gz",
    ) == tuple(reversed(first))


def test_algebraic_metadata_rejects_private_environment_fields():
    model = _algebraic_fixture()
    payload = json.dumps(model.summary(), sort_keys=True)
    assert "portable.qplib" in payload
    assert "/mnt/" not in payload
    with pytest.raises(ValueError, match="Private environment"):
        AlgebraicModel(
            name=model.name,
            variable_names=model.variable_names,
            variable_types=model.variable_types,
            lower_bounds=model.lower_bounds,
            upper_bounds=model.upper_bounds,
            objective=model.objective,
            metadata={"hostname": "private-machine"},
        )
    with pytest.raises(ValueError, match="Private environment"):
        AlgebraicModel(
            name=model.name,
            variable_names=model.variable_names,
            variable_types=model.variable_types,
            lower_bounds=model.lower_bounds,
            upper_bounds=model.upper_bounds,
            objective=model.objective,
            metadata={"source": "/private/work/model.qplib"},
        )
    with pytest.raises(ValueError, match="Private environment"):
        AlgebraicModel(
            name=model.name,
            variable_names=model.variable_names,
            variable_types=model.variable_types,
            lower_bounds=model.lower_bounds,
            upper_bounds=model.upper_bounds,
            objective=model.objective,
            metadata={"source_url": "http://localhost/model.qplib"},
        )


def test_sparse_scaling_is_reversible_and_preserves_model_values():
    model = _algebraic_fixture()
    scaled, factors = scaled_model(model)
    original = np.asarray([1.0, 2.0, 0.5])
    transformed = factors.to_scaled(original)
    np.testing.assert_allclose(factors.to_original(transformed), original)
    assert scaled.objective.value(transformed) == pytest.approx(model.objective.value(original))
    for index, (before, after) in enumerate(
        zip(model.constraints, scaled.constraints, strict=True)
    ):
        assert after.expression.value(transformed) == pytest.approx(
            factors.rows[index] * before.expression.value(original)
        )


def test_sparse_scaling_preserves_the_discrete_lattice():
    model = _algebraic_fixture()
    scaled, factors = scaled_model(model)
    np.testing.assert_array_equal(factors.columns[:2], np.ones(2))
    assert factors.preserves_integrality
    assert scaled.lower_bounds[:2].tolist() == model.lower_bounds[:2].tolist()
    assert scaled.upper_bounds[:2].tolist() == model.upper_bounds[:2].tolist()
    unsafe = ScalingFactors(np.asarray([2.0, 1.0, 1.0]), np.ones(1))
    with pytest.raises(ValueError, match="Integral variable columns"):
        scaled_model(model, unsafe)


@pytest.mark.parametrize("lower,upper", [(0, 1), (-2, 3), (0, 15), (-100, 100)])
def test_adaptive_integer_encodings_round_trip_every_local_value(lower, upper):
    plan = qqa.mixed.choose_integer_encoding(lower, upper)
    for value in range(lower, upper + 1):
        encoded = qqa.mixed.encode_integer(value, plan)
        assert qqa.mixed.decode_integer(encoded, plan) == value


def test_dynamic_convexification_reconstructs_and_is_tangent():
    matrix = sparse.csr_matrix([[1.0, 0.0], [0.0, -3.0]])
    decomposition = dc_decomposition(matrix)
    np.testing.assert_allclose(
        (decomposition.convex - decomposition.concave).toarray(),
        matrix.toarray(),
        atol=1e-8,
    )
    assert np.linalg.eigvalsh(decomposition.convex.toarray()).min() >= -1e-7
    point = np.asarray([0.5, -2.0])
    convex, gradient, constant = linearize_concave_part(decomposition, point)
    np.testing.assert_allclose(convex.toarray(), decomposition.convex.toarray())
    expected = -0.5 * point @ decomposition.concave.dot(point)
    affine = gradient @ point + constant
    assert affine == pytest.approx(expected)


def test_rens_and_rins_core_selection_uses_local_integer_domains():
    state = SCIPState(
        variables=tuple(range(4)),
        names=("a", "b", "c", "y"),
        variable_types=("BINARY", "INTEGER", "INTEGER", "CONTINUOUS"),
        lp_values=np.asarray([0.45, 100.3, 2.0, 0.0]),
        incumbent_values=None,
        local_lower=np.asarray([0.0, -1000.0, 2.0, -math.inf]),
        local_upper=np.asarray([1.0, 1000.0, 2.0, math.inf]),
        reduced_costs=np.asarray([0.0, 2.0, 0.0, 0.0]),
        pseudocosts=np.asarray([1.0, 1.0, 0.0, 0.0]),
        node_number=1,
        depth=0,
    )
    selection = select_uncertain_integer_core(state, max_core_size=2)
    assert selection.mode == "rens"
    assert set(selection.core_indices.tolist()) == {0, 1}
    position = selection.core_indices.tolist().index(1)
    # General integers retain a real local-search neighbourhood rather than
    # being collapsed to the LP point's floor/ceil pair.
    assert selection.local_lower[position] == 98
    assert selection.local_upper[position] == 103
    assert 2 in selection.fixed_indices

    rins_state = replace(
        state,
        incumbent_values=np.asarray([1.0, 101.0, 2.0, 0.0]),
    )
    assert select_uncertain_integer_core(rins_state, max_core_size=2).mode == "rins"


def test_qqa_reference_pool_tracks_node_disagreement_and_previous_values():
    heuristic = QQAHeuristic(QQAHeuristicConfig(reference_pool_size=2))
    state = SCIPState(
        variables=tuple(range(3)),
        names=("binary", "integer", "continuous"),
        variable_types=("BINARY", "INTEGER", "CONTINUOUS"),
        lp_values=np.asarray([0.1, 2.0, 0.5]),
        incumbent_values=None,
        local_lower=np.asarray([0.0, 0.0, 0.0]),
        local_upper=np.asarray([1.0, 4.0, 1.0]),
        reduced_costs=np.zeros(3),
        pseudocosts=np.zeros(3),
        node_number=1,
        depth=0,
    )
    heuristic._remember_reference(state)
    later = replace(
        state,
        lp_values=np.asarray([0.9, 4.0, 0.1]),
        node_number=2,
    )
    np.testing.assert_allclose(
        heuristic._reference_disagreement(later),
        np.asarray([0.8, 0.5, 0.0]),
    )
    previous = heuristic._reference_initial_values(
        later,
        np.asarray([0, 1]),
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 4.0]),
    )
    assert len(previous) == 1
    np.testing.assert_allclose(previous[0], np.asarray([0.1, 2.0]))
    heuristic._remember_reference(later)
    heuristic._remember_reference(later)
    assert len(heuristic._reference_pool) == 2


def test_core_surrogate_uses_original_objective_and_active_lp_rows():
    class Variable:
        def __init__(self, name, objective):
            self.name = name
            self._objective = objective

        def getObj(self):
            return self._objective

    class Column:
        def __init__(self, variable):
            self._variable = variable

        def getVar(self):
            return self._variable

    class Row:
        def __init__(self, variables):
            self._columns = [Column(variable) for variable in variables]

        def getCols(self):
            return self._columns

        def getVals(self):
            return [1.0, 1.0]

        def getConstant(self):
            return 0.0

        def getLhs(self):
            return -1e20

        def getRhs(self):
            return 1.0

    variables = (Variable("t_x0", -3.0), Variable("t_x1", -1.0))
    state = SCIPState(
        variables=variables,
        names=("x0", "x1"),
        variable_types=("BINARY", "BINARY"),
        lp_values=np.asarray([0.5, 0.5]),
        incumbent_values=None,
        local_lower=np.zeros(2),
        local_upper=np.ones(2),
        reduced_costs=np.zeros(2),
        pseudocosts=np.zeros(2),
        node_number=1,
        depth=0,
    )
    selection = CoreSelection(
        core_indices=np.asarray([0, 1]),
        fixed_indices=np.empty(0, dtype=np.int64),
        fixed_values=np.empty(0),
        local_lower=np.zeros(2),
        local_upper=np.ones(2),
        scores=np.ones(2),
        mode="rens",
    )
    algebraic = AlgebraicModel(
        name="surrogate",
        variable_names=("x0", "x1"),
        variable_types=(VariableType.BINARY, VariableType.BINARY),
        lower_bounds=np.zeros(2),
        upper_bounds=np.ones(2),
        objective=SparseQuadratic(
            sparse.csr_matrix([[0.0, 2.0], [2.0, 0.0]]),
            np.asarray([-3.0, -1.0]),
        ),
    )
    model = SimpleNamespace(infinity=lambda: 1e20, getLPRowsData=lambda: [Row(variables)])
    surrogate = build_core_surrogate(
        model,
        state,
        selection,
        [0, 1],
        algebraic=algebraic,
    )
    assert surrogate.objective_source == "algebraic"
    assert surrogate.num_rows == 1
    assert surrogate.row_violations(np.asarray([1.0, 1.0]))[0, 0] > 0
    candidates = generate_surrogate_candidates(
        surrogate,
        target=state.lp_values,
        lower=np.zeros(2),
        upper=np.ones(2),
        max_candidates=4,
        seed=0,
    )
    np.testing.assert_array_equal(candidates[0], np.asarray([1.0, 0.0]))
    core_problem, _ = scip_heuristic_module._core_problem(
        state,
        selection,
        [0, 1],
        surrogate,
        QQAHeuristicConfig(),
        adaptive_rows=True,
    )
    assert len(core_problem.constraints) == 1
    assert core_problem.constraints[0].sense == "<="
    violating = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    expected = surrogate.merit_values(
        violating.numpy(),
        target=state.lp_values,
        span=np.ones(2),
        row_penalty=20.0,
        proximity_weight=0.02,
    )[0]
    assert core_problem.loss_fn(violating).item() == pytest.approx(expected)
    cached_rows = next(iter(core_problem._row_device_cache.values()))
    core_problem.constraint_values(violating)
    assert next(iter(core_problem._row_device_cache.values())) is cached_rows
    static_problem, _ = scip_heuristic_module._core_problem(
        state,
        selection,
        [0, 1],
        surrogate,
        QQAHeuristicConfig(),
    )
    assert static_problem.constraints == ()
    assert static_problem.loss_fn(violating).item() == pytest.approx(expected)
    result = core_problem.solve(
        sol_size=4,
        num_epochs=4,
        initial_state=violating.expand(4, -1),
        adaptive_augmented_lagrangian=True,
        al_update_interval=1,
        calibrate_penalty=False,
        repair=False,
        polish=False,
        return_population=True,
        verbose=False,
    )
    assert result.diagnostics["adaptive_augmented_lagrangian"]["updates"] == 4
    assert result.diagnostics["constraint_archive"]["observations"] > 0


def _constrained_real_problem() -> qqa.MixedProblem:
    return qqa.MixedProblem(
        [qqa.Real("x", 0.0, 2.0)],
        lambda values: values["x"].square(),
        constraints=[
            qqa.Constraint(
                lambda values: values["x"],
                sense=">=",
                rhs=1.0,
                scale=1.0,
                tolerance=1e-6,
                name="minimum",
            ),
            qqa.Constraint(
                lambda values: values["x"],
                sense="<=",
                rhs=2.0,
                scale=1.0,
                tolerance=1e-6,
                name="maximum",
            ),
        ],
        dtype=torch.float64,
    )


def test_constraint_archives_and_row_specific_augmented_lagrangian_updates():
    problem = _constrained_real_problem()
    archive = ConstraintArchive()
    archive.update(problem, torch.tensor([[0.0]], dtype=torch.float64))
    assert archive.objective_solution is None
    archive.update(problem, torch.tensor([[2.0], [1.0]], dtype=torch.float64))
    assert archive.objective_solution.item() == pytest.approx(1.0)
    assert archive.feasibility_solution.item() == pytest.approx(1.0)

    controller = AdaptiveAugmentedLagrangian.for_problem(problem)
    values = torch.tensor([[0.0]], dtype=torch.float64)
    controller.update(problem, values)
    initial = controller.rho.clone()
    controller.update(problem, values)
    assert controller.rho[0] > initial[0]
    assert controller.rho[1] == initial[1]


def test_elastic_repair_and_mixed_solver_archive_restore_feasibility():
    problem = _constrained_real_problem()
    repaired = qqa.repair_mixed_solution(
        problem,
        torch.tensor([0.0], dtype=torch.float64),
        max_steps=120,
    )
    assert repaired.feasible
    assert repaired.solution.item() == pytest.approx(1.0, abs=2e-2)

    result = problem.solve(
        sol_size=8,
        num_epochs=20,
        repair_steps=40,
        verbose=False,
    )
    assert result.diagnostics["constraint_archive"]["observations"] > 0
    assert result.score["feasible"]


def test_real_decode_retains_inward_gradient_at_declared_bounds():
    problem = _constrained_real_problem()
    latent = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    problem.space.decode(latent).sum().backward()
    assert latent.grad is not None
    assert latent.grad.item() == pytest.approx(2.0)


def test_conditional_heuristic_uses_qqa_guided_partial_fixings_for_lns_repair(monkeypatch):
    calls = []

    def fake_completion(template, variable_names, values, **kwargs):
        calls.append((tuple(variable_names), tuple(values)))
        repaired = tuple(variable_names) == ("complement", "core")
        return CompletionResult(
            feasible=repaired,
            accepted=repaired,
            improved_incumbent=repaired,
            status="optimal" if repaired else "infeasible",
            objective=-1.0 if repaired else None,
            values=np.asarray([1.0]) if repaired else None,
            runtime=0.01,
            fixed_variables=len(variable_names),
        )

    monkeypatch.setattr(
        scip_heuristic_module,
        "complete_integer_assignment",
        fake_completion,
    )
    heuristic = QQAHeuristic(
        QQAHeuristicConfig(
            completion_time=1.0,
            maximum_overhead_fraction=0.5,
            use_dive_completion=False,
            subscip_repair=True,
        ),
        completion_template=object(),
    )
    pyscipopt = pytest.importorskip("pyscipopt")
    active_model = pyscipopt.Model()
    active_model.hideOutput()
    active_model.setRealParam("limits/time", 10.0)
    heuristic.model = active_model
    state = SimpleNamespace(
        lp_values=np.asarray([0.4, 0.0]),
        incumbent_values=np.asarray([0.0, 0.0]),
        names=("core", "complement"),
        local_lower=np.asarray([0.0, 0.0]),
        local_upper=np.asarray([1.0, 1.0]),
    )
    selection = SimpleNamespace(
        core_indices=np.asarray([0]),
        fixed_indices=np.asarray([1]),
        fixed_values=np.asarray([0.0]),
        local_lower=np.asarray([0.0]),
        local_upper=np.asarray([1.0]),
        scores=np.asarray([1.0]),
    )
    neighborhood = IntegerNeighborhood(
        core_indices=np.asarray([0]),
        lower=np.asarray([0.0]),
        upper=np.asarray([1.0]),
        fixed_indices=np.asarray([1]),
        fixed_values=np.asarray([0.0]),
    )
    problem = qqa.MixedProblem(
        [qqa.Binary("x")],
        lambda values: -values["x"],
        dtype=torch.float64,
    )
    accepted, improved = heuristic._complete_population(
        [np.asarray([1.0])],
        problem,
        state,
        selection,
        [0],
        np.asarray([0]),
        neighborhood,
        set(),
        source="qqa",
    )
    assert calls == [
        (("core", "complement"), (1.0, 0.0)),
        (("complement", "core"), (0.0, 1.0)),
    ]
    assert accepted and improved
    assert heuristic.stats.lns_repair_attempts == 1
    assert heuristic.stats.lns_repair_feasible == 1
    assert heuristic.stats.lns_repair_accepted == 1
    active_model.free()


def test_qqa_repair_beam_retains_jointly_improving_changes():
    problem = qqa.MixedProblem(
        [qqa.Binary("x0"), qqa.Binary("x1"), qqa.Binary("x2")],
        lambda values: (
            2.0 * values["x0"]
            + 2.0 * values["x1"]
            - 5.0 * values["x0"] * values["x1"]
            + 0.1 * values["x2"]
        ),
        dtype=torch.float64,
    )
    selected = scip_heuristic_module._select_repair_positions(
        problem,
        np.zeros(3),
        np.ones(3),
        [10, 20, 30],
        max_changes=2,
        beam_width=2,
    )
    assert selected == [10, 20]
    assert (
        scip_heuristic_module._select_repair_positions(
            problem,
            np.zeros(3),
            np.asarray([0.0, 0.0, 1.0]),
            [10, 20, 30],
            max_changes=1,
            beam_width=2,
        )
        == []
    )


def test_conditional_heuristic_partially_fixes_an_infeasible_core_lns(monkeypatch):
    fixed_counts = []

    def fake_completion(template, variable_names, values, **kwargs):
        fixed_counts.append(len(variable_names))
        feasible = len(variable_names) == 2
        return CompletionResult(
            feasible=feasible,
            accepted=feasible,
            improved_incumbent=feasible,
            status="optimal" if feasible else "infeasible",
            objective=-1.0 if feasible else None,
            values=np.asarray(values) if feasible else None,
            runtime=0.01,
            fixed_variables=len(variable_names),
        )

    monkeypatch.setattr(
        scip_heuristic_module,
        "complete_integer_assignment",
        fake_completion,
    )
    heuristic = QQAHeuristic(
        QQAHeuristicConfig(
            completion_time=1.0,
            maximum_overhead_fraction=0.5,
            use_dive_completion=False,
            subscip_repair=True,
        ),
        completion_template=object(),
    )
    pyscipopt = pytest.importorskip("pyscipopt")
    active_model = pyscipopt.Model()
    active_model.hideOutput()
    active_model.setRealParam("limits/time", 10.0)
    heuristic.model = active_model
    core = np.arange(4)
    state = SimpleNamespace(
        lp_values=np.full(5, 0.4),
        incumbent_values=np.zeros(5),
        names=tuple(f"x_{index}" for index in range(5)),
        local_lower=np.zeros(5),
        local_upper=np.ones(5),
    )
    selection = SimpleNamespace(
        core_indices=core,
        fixed_indices=np.asarray([4]),
        fixed_values=np.asarray([0.0]),
        local_lower=np.zeros(4),
        local_upper=np.ones(4),
        scores=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    neighborhood = IntegerNeighborhood(
        core_indices=core,
        lower=np.zeros(4),
        upper=np.ones(4),
        fixed_indices=np.asarray([4]),
        fixed_values=np.asarray([0.0]),
    )
    problem = qqa.MixedProblem(
        [qqa.Binary(f"x_{index}") for index in range(4)],
        lambda values: -sum(values.values()),
        dtype=torch.float64,
    )
    accepted, improved = heuristic._complete_population(
        [np.ones(4)],
        problem,
        state,
        selection,
        list(range(4)),
        core,
        neighborhood,
        set(),
        source="qqa",
    )
    assert fixed_counts == [5, 2]
    assert accepted and improved
    assert heuristic.stats.partial_lns_attempts == 0
    assert heuristic.stats.lns_repair_feasible == 1
    assert heuristic.stats.lns_repair_accepted == 1
    active_model.free()


def test_conditional_heuristic_repairs_without_an_incumbent(monkeypatch):
    fixed_counts = []

    def fake_completion(template, variable_names, values, **kwargs):
        fixed_counts.append(len(variable_names))
        feasible = len(variable_names) == 2
        return CompletionResult(
            feasible=feasible,
            accepted=feasible,
            improved_incumbent=feasible,
            status="optimal" if feasible else "infeasible",
            objective=-1.0 if feasible else None,
            values=np.asarray(values) if feasible else None,
            runtime=0.01,
            fixed_variables=len(variable_names),
        )

    monkeypatch.setattr(
        scip_heuristic_module,
        "complete_integer_assignment",
        fake_completion,
    )
    heuristic = QQAHeuristic(
        QQAHeuristicConfig(
            completion_time=1.0,
            maximum_overhead_fraction=0.5,
            use_dive_completion=False,
            subscip_repair=True,
            require_incumbent=False,
        ),
        completion_template=object(),
    )
    pyscipopt = pytest.importorskip("pyscipopt")
    active_model = pyscipopt.Model()
    active_model.hideOutput()
    active_model.setRealParam("limits/time", 10.0)
    heuristic.model = active_model
    core = np.arange(4)
    state = SimpleNamespace(
        lp_values=np.full(5, 0.4),
        incumbent_values=None,
        names=tuple(f"x_{index}" for index in range(5)),
        local_lower=np.zeros(5),
        local_upper=np.ones(5),
    )
    selection = SimpleNamespace(
        core_indices=core,
        fixed_indices=np.asarray([4]),
        fixed_values=np.asarray([0.0]),
        local_lower=np.zeros(4),
        local_upper=np.ones(4),
        scores=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    neighborhood = IntegerNeighborhood(
        core_indices=core,
        lower=np.zeros(4),
        upper=np.ones(4),
        fixed_indices=np.asarray([4]),
        fixed_values=np.asarray([0.0]),
    )
    problem = qqa.MixedProblem(
        [qqa.Binary(f"x_{index}") for index in range(4)],
        lambda values: -sum(values.values()),
        dtype=torch.float64,
    )
    accepted, improved = heuristic._complete_population(
        [np.ones(4)],
        problem,
        state,
        selection,
        list(range(4)),
        core,
        neighborhood,
        set(),
        source="qqa",
    )
    assert fixed_counts == [5, 2]
    assert accepted and improved
    assert heuristic.stats.partial_lns_attempts == 1
    assert heuristic.stats.partial_lns_feasible == 1
    assert heuristic.stats.partial_lns_accepted == 1
    assert heuristic.stats.partial_lns_incumbent_improvements == 1
    assert heuristic.stats.lns_repair_attempts == 0
    active_model.free()


def test_conditional_heuristic_suppresses_repeated_unproductive_qqa_calls():
    heuristic = QQAHeuristic(QQAHeuristicConfig())
    assert not heuristic._qqa_has_stalled()
    heuristic.stats.qqa_calls = 1
    assert heuristic._qqa_has_stalled()
    heuristic.stats.qqa_incumbent_improvements = 1
    assert not heuristic._qqa_has_stalled()
    ablation = QQAHeuristic(QQAHeuristicConfig(stop_qqa_after_nonimproving_call=False))
    ablation.stats.qqa_calls = 1
    assert not ablation._qqa_has_stalled()


def test_conditional_heuristic_accounts_complete_callback_overhead(monkeypatch):
    heuristic = QQAHeuristic(QQAHeuristicConfig(maximum_overhead_fraction=0.05))
    pyscipopt = pytest.importorskip("pyscipopt")
    active_model = pyscipopt.Model()
    active_model.hideOutput()
    active_model.setRealParam("limits/time", 10.0)
    heuristic.model = active_model
    heuristic.stats.callback_runtime = 0.4
    assert heuristic._remaining_overhead_budget() == pytest.approx(0.1)

    heuristic.stats.callback_runtime = 0.0
    heuristic.stats.numerical_runtime_initialisation = 0.6
    assert heuristic._remaining_overhead_budget() == 0.0

    times = iter((1.0, 1.25))
    monkeypatch.setattr(scip_heuristic_module, "perf_counter", lambda: next(times))

    def execute(heurtiming, nodeinfeasible):
        assert (heurtiming, nodeinfeasible) == ("timing", False)
        assert heuristic._active_callback_started_at == 1.0
        return {"result": "complete"}

    monkeypatch.setattr(heuristic, "_heurexec_impl", execute)
    assert heuristic.heurexec("timing", False) == {"result": "complete"}
    assert heuristic.stats.callback_runtime == pytest.approx(0.25)
    assert heuristic._active_callback_started_at is None
    active_model.free()


def test_conditional_heuristic_requires_cold_start_overhead_reserve():
    heuristic = QQAHeuristic(
        QQAHeuristicConfig(
            maximum_overhead_fraction=0.05,
            minimum_runtime_startup_time=8.0,
        )
    )
    pyscipopt = pytest.importorskip("pyscipopt")
    active_model = pyscipopt.Model()
    active_model.hideOutput()
    active_model.setRealParam("limits/time", 30.0)
    heuristic.model = active_model
    assert heuristic._remaining_overhead_budget() == pytest.approx(1.5)
    assert not heuristic._runtime_startup_is_affordable()
    active_model.free()


@pytest.mark.parametrize(
    ("available", "expected"),
    [(1.0, 0.5), (10.0, 1.5), (100.0, 2.0)],
)
def test_conditional_heuristic_reserves_bounded_callback_deadline_safety(available, expected):
    assert QQAHeuristic._callback_deadline_safety(available) == pytest.approx(expected)


def test_completion_improvement_threshold_is_validated_before_solver_use():
    with pytest.raises(ValueError, match="minimum_relative_improvement"):
        complete_integer_assignment(
            object(),
            [],
            [],
            minimum_relative_improvement=1.0,
        )
    with pytest.raises(ValueError, match="minimum_relative_improvement"):
        complete_integer_assignment_dive(
            object(),
            [],
            [],
            minimum_relative_improvement=-0.1,
        )


def test_qqa_priority_preserves_mip_order_and_opens_qplib_early_window():
    pytest.importorskip("pyscipopt")

    class RecordingModel:
        def __init__(self):
            self.priority = None

        def includeHeur(self, heuristic, *args, **kwargs):  # noqa: N802, ARG002
            self.priority = kwargs["priority"]

    linear_model = _algebraic_fixture()
    mip = RecordingModel()
    scip_heuristic_module.include_qqa_heuristic(
        mip,
        QQAHeuristicConfig(subscip_repair=False),
        algebraic=linear_model,
    )
    assert mip.priority == -1_200_000

    qplib = RecordingModel()
    scip_heuristic_module.include_qqa_heuristic(
        qplib,
        QQAHeuristicConfig(subscip_repair=False),
        algebraic=replace(linear_model, problem_type="QML"),
    )
    assert qplib.priority == -1_100_000


def test_primary_hybrid_escalates_only_after_a_completable_fast_candidate():
    primary = QQAHeuristic(QQAHeuristicConfig(fast_candidates=2))
    assert not primary._fast_path_supports_qqa(0)
    primary.stats.completion_feasible = 1
    assert primary._fast_path_supports_qqa(0)
    ablation = QQAHeuristic(QQAHeuristicConfig(fast_candidates=0))
    assert ablation._fast_path_supports_qqa(0)


def test_anneal_honours_optional_wall_clock_deadline():
    problem = qqa.MaxCut(nx.cycle_graph(8))
    result = qqa.anneal(
        problem,
        sol_size=8,
        num_epochs=100_000,
        time_limit=1e-4,
        polish=False,
        verbose=False,
    )
    assert result.diagnostics["deadline_reached"]
    assert result.diagnostics["completed_epochs"] < 100_000


def test_qplib_lower_triangle_conversion_is_checked_at_nonzero_point(tmp_path, monkeypatch):
    pyqplib = pytest.importorskip("pyqplib")

    class Lower:
        diag_rows = np.asarray([], dtype=np.int64)
        diag_vals = np.asarray([], dtype=np.float64)
        subdiag_rows = np.asarray([1], dtype=np.int64)
        subdiag_cols = np.asarray([0], dtype=np.int64)
        subdiag_vals = np.asarray([4.0], dtype=np.float64)

        def __bool__(self):
            return True

    objective = SimpleNamespace(
        mat=Lower(),
        lin=np.asarray([1.0, 2.0]),
        offset=3.0,
        sense=pyqplib.Sense.MINIMIZE,
    )
    parsed = SimpleNamespace(
        name="QPLIB_TEST",
        num_vars=2,
        num_cons=0,
        obj=objective,
        constraints=None,
        var_types=[pyqplib.VarType.CONTINUOUS, pyqplib.VarType.CONTINUOUS],
        var_lb=np.asarray([-1.0, -1.0]),
        var_ub=np.asarray([2.0, 2.0]),
        cons_lb=np.asarray([]),
        cons_ub=np.asarray([]),
        description=SimpleNamespace(
            obj_type=SimpleNamespace(name="GENERAL"),
            var_type=SimpleNamespace(name="CONTINUOUS"),
            cons_type=SimpleNamespace(name="UNCONSTRAINED"),
        ),
        x0=np.asarray([0.0, 0.0]),
        obj_val=lambda x: 2.0 * x[0] * x[1] + x[0] + 2.0 * x[1] + 3.0,
        cons_val=lambda x: np.asarray([]),
    )
    monkeypatch.setattr(pyqplib, "read_problem", lambda path: parsed)
    path = tmp_path / "QPLIB_TEST.qplib"
    path.write_text("synthetic parser fixture\n", encoding="utf-8")
    model = load_qplib(path)
    assert model.objective.value(np.asarray([1.0, 2.0])) == pytest.approx(12.0)
    assert model.objective.quadratic[0, 1] == pytest.approx(2.0)


def _write_tiny_mps(path):
    pyscipopt = pytest.importorskip("pyscipopt")
    model = pyscipopt.Model("portable-mip")
    model.hideOutput()
    x = model.addVar("x", vtype="B")
    y = model.addVar("y", vtype="C", lb=0.0, ub=10.0)
    model.addCons(y >= 2.0 * x, name="link")
    model.addCons(y >= 1.0, name="minimum")
    model.setObjective(y - 3.0 * x, "minimize")
    model.writeProblem(str(path))


def test_mps_import_scip_roundtrip_completion_and_benchmark(tmp_path):
    path = tmp_path / "portable.mps"
    _write_tiny_mps(path)
    algebraic = load_mps(path)
    assert algebraic.summary()["constraint_linear_nonzeros"] == 3
    assert sparse.isspmatrix_csr(algebraic.constraints[0].expression.linear)
    assert str(tmp_path) not in json.dumps(algebraic.summary())
    lightweight = load_mps(path, include_constraints=False)
    assert lightweight.num_constraints == 0
    assert lightweight.variable_names == algebraic.variable_names
    np.testing.assert_allclose(
        lightweight.objective.linear.toarray(), algebraic.objective.linear.toarray()
    )
    with pytest.raises(TypeError, match="include_constraints"):
        load_mps(path, include_constraints=1)

    model, variables = build_scip_model(algebraic)
    template = create_completion_template(model)
    completion = complete_integer_assignment(
        template,
        ["x"],
        [1.0],
        time_limit=2.0,
        node_limit=50,
    )
    assert completion.feasible
    assert completion.objective == pytest.approx(-1.0)

    model.optimize()
    best = model.getBestSol()
    point = np.asarray([model.getSolVal(best, variable) for variable in variables])
    assert algebraic.evaluate(point).maximum_infeasibility <= 1e-7

    reference = tmp_path / "benchmark.solu"
    reference.write_text("=opt= portable -1\n", encoding="utf-8")
    benchmark = run_benchmark_instance(
        path,
        solver="scip",
        time_limit=2.0,
        reference_file=reference,
    )
    assert benchmark.status == "optimal"
    assert benchmark.feasible
    assert benchmark.objective == pytest.approx(-1.0)
    assert benchmark.reference_objective == pytest.approx(-1.0)
    assert benchmark.provenance["model_statistics"] == {
        "num_variables": 2,
        "num_constraints": 2,
        "variable_counts": {
            "binary": 1,
            "integer": 0,
            "continuous": 1,
            "implicit_integer": 0,
        },
        "objective_linear_nonzeros": 2,
        "objective_quadratic_nonzeros": 0,
        "constraint_linear_nonzeros": None,
        "constraint_quadratic_nonzeros": None,
    }
    assert str(tmp_path) not in json.dumps(benchmark.to_dict())

    suite = run_benchmark_suite([path, path], solver="scip", time_limit=1.0)
    assert suite.summary["overall"]["instances"] == 2
    assert suite.summary["overall"]["feasible_rate"] == 1.0
    assert suite.summary["by_problem_type"]["MIPLIB"]["instances"] == 2

    checkpoint = tmp_path / "comparison.json"
    comparison = compare_benchmark_solvers(
        [path],
        solvers=("scip", "sg-cqqa"),
        seeds=(0,),
        baseline_solver="scip",
        time_limit=1.0,
        checkpoint_file=checkpoint,
    )
    paired = comparison.summary["pairwise"]["sg-cqqa"]
    assert paired["paired_runs"] == 1
    assert sum(paired["primal_quality"].values()) == 1
    assert comparison.comparison_config["instances"] == ["portable.mps"]
    assert "seed" not in comparison.comparison_config["qqa_config"]
    assert str(tmp_path) not in json.dumps(comparison.to_dict())
    resumed = compare_benchmark_solvers(
        [path],
        solvers=("scip", "sg-cqqa"),
        seeds=(0,),
        baseline_solver="scip",
        time_limit=1.0,
        checkpoint_file=checkpoint,
        resume=True,
    )
    assert len(resumed.results) == 2
    assert resumed.summary["campaign"]["completed_runs"] == 2


def test_comparison_reuses_exact_aggressive_result_for_structural_bypass(tmp_path):
    path = tmp_path / "bypassed.mps"
    _write_tiny_mps(path)
    comparison = compare_benchmark_solvers(
        [path],
        solvers=("scip-aggressive", "sg-cqqa"),
        seeds=(0,),
        baseline_solver="scip-aggressive",
        qqa_config=QQAHeuristicConfig(maximum_problem_variables=1),
        time_limit=1.0,
    )
    baseline = next(row for row in comparison.results if row.solver == "scip-aggressive")
    hybrid = next(row for row in comparison.results if row.solver == "sg-cqqa")
    assert hybrid.objective == baseline.objective
    assert hybrid.trajectory == baseline.trajectory
    assert hybrid.run_config["equivalent_baseline_reuse"] is True
    assert hybrid.run_config["qqa_plugin_active"] is False
    assert comparison.summary["pairwise"]["sg-cqqa"]["primal_quality"] == {
        "losses": 0,
        "ties": 1,
        "wins": 0,
    }


def test_qplib_applicability_hint_uses_public_header(tmp_path):
    path = tmp_path / "small.qplib"
    path.write_text("small\nQML\nminimize\n60 # variables\n", encoding="utf-8")
    assert _qqa_applicability_hint(
        path,
        "qplib",
        QQAHeuristicConfig(maximum_problem_variables=64),
        algebraic=None,
    )
    assert not _qqa_applicability_hint(
        path,
        "qplib",
        QQAHeuristicConfig(maximum_problem_variables=32),
        algebraic=None,
    )
    pure_binary = tmp_path / "binary.qplib"
    pure_binary.write_text("binary\nQBL\nminimize\n60 # variables\n", encoding="utf-8")
    assert not _qqa_applicability_hint(
        pure_binary,
        "qplib",
        QQAHeuristicConfig(maximum_integer_variables=32),
        algebraic=None,
    )
    small_binary = tmp_path / "small-binary.qplib"
    small_binary.write_text("small\nQBL\nminimize\n8 # variables\n", encoding="utf-8")
    assert not _qqa_applicability_hint(
        small_binary,
        "qplib",
        QQAHeuristicConfig(minimum_core_size=16),
        algebraic=None,
    )
    assert not _qqa_applicability_hint(
        path,
        "qplib",
        QQAHeuristicConfig(allowed_qplib_problem_types=("LIQ",)),
        algebraic=None,
    )


def test_structural_bypass_reuses_aggressive_failure(tmp_path, monkeypatch):
    path = tmp_path / "continuous.qplib"
    path.write_text("continuous\nQCL\nminimize\n10 # variables\n", encoding="utf-8")
    attempted = []

    def fail_once(source, *, solver, **kwargs):  # noqa: ARG001
        attempted.append(solver)
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(
        "qqa.benchmarking.algebraic_runner._run_isolated_benchmark_instance",
        fail_once,
    )
    comparison = compare_benchmark_solvers(
        [path],
        solvers=("scip-aggressive", "sg-cqqa"),
        baseline_solver="scip-aggressive",
        format="qplib",
        time_limit=0.1,
        continue_on_error=True,
    )
    assert attempted == ["scip-aggressive"]
    assert [failure.solver for failure in comparison.failures] == [
        "scip-aggressive",
        "sg-cqqa",
    ]
    assert {failure.error_type for failure in comparison.failures} == {"RuntimeError"}


def test_independent_structural_bypass_preserves_balanced_order(tmp_path, monkeypatch):
    path = tmp_path / "continuous.qplib"
    path.write_text("continuous\nQCL\nminimize\n10 # variables\n", encoding="utf-8")
    attempted = []

    def fail_both(source, *, solver, **kwargs):  # noqa: ARG001
        attempted.append(solver)
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(
        "qqa.benchmarking.algebraic_runner._run_isolated_benchmark_instance",
        fail_both,
    )
    solvers = ("scip-aggressive", "sg-cqqa")
    compare_benchmark_solvers(
        [path],
        solvers=solvers,
        baseline_solver="scip-aggressive",
        format="qplib",
        time_limit=0.1,
        continue_on_error=True,
        reuse_equivalent_baseline=False,
    )
    assert attempted == list(
        _comparison_solver_order(
            solvers,
            execution_order="balanced",
            seed=0,
            instance_index=0,
            instance_name=path.name,
        )
    )


def test_disposable_native_benchmark_worker_roundtrip(tmp_path):
    from qqa.benchmarking.algebraic_runner import _run_isolated_benchmark_instance
    from qqa.hybrid import QQAHeuristicConfig

    path = tmp_path / "isolated.mps"
    _write_tiny_mps(path)
    result = _run_isolated_benchmark_instance(
        path,
        resolved_format="miplib",
        solver="scip",
        seed=0,
        qqa_config=QQAHeuristicConfig(),
        reference_records=None,
        run_kwargs={
            "format": "miplib",
            "time_limit": 2.0,
            "relative_gap_limit": 0.0,
            "threads": 1,
            "reference_file": None,
            "verbose": False,
        },
        common_import=False,
        worker_timeout=float(os.environ.get("QQA_TEST_WORKER_TIMEOUT_SECONDS", "300")),
    )
    assert result.status == "optimal"
    assert result.feasible
    assert result.objective == pytest.approx(-1.0)
    assert result.run_config["metric_clock"] == "total_wall_clock"
    assert result.stage_timings["setup_and_plugin"] > 0.01
    assert str(tmp_path) not in json.dumps(result.to_dict())


def test_sg_cqqa_continuous_model_uses_matched_aggressive_scip_fallback(tmp_path, monkeypatch):
    pytest.importorskip("pyscipopt")
    path = tmp_path / "continuous.qplib"
    path.write_text("synthetic continuous fixture\n", encoding="utf-8")
    continuous = AlgebraicModel(
        name="continuous",
        variable_names=("x",),
        variable_types=(VariableType.CONTINUOUS,),
        lower_bounds=(0.0,),
        upper_bounds=(1.0,),
        objective=SparseQuadratic.linear_expression(
            sparse.csr_matrix(([-1.0], ([0], [0])), shape=(1, 1))
        ),
        metadata={"source_name": path.name},
    )

    def unexpected_plugin(*args, **kwargs):
        raise AssertionError("continuous models must not construct the QQA plugin")

    monkeypatch.setattr(
        "qqa.benchmarking.algebraic_runner.include_qqa_heuristic",
        unexpected_plugin,
    )
    result = run_benchmark_instance(
        path,
        format="qplib",
        solver="sg-cqqa",
        time_limit=1.0,
        _algebraic=continuous,
    )
    assert result.feasible
    assert result.objective == pytest.approx(-1.0)
    assert result.qqa is None
    assert result.run_config["qqa_applicable"] is False
    assert result.run_config["qqa_plugin_active"] is False
    assert result.run_config["torch_threads"] is None


def test_sg_cqqa_skips_plugin_import_without_qqa_time_reserve(tmp_path, monkeypatch):
    path = tmp_path / "short.mps"
    _write_tiny_mps(path)

    def unexpected_plugin(*args, **kwargs):
        raise AssertionError("insufficient budgets must not construct the QQA plugin")

    monkeypatch.setattr(
        "qqa.benchmarking.algebraic_runner.include_qqa_heuristic",
        unexpected_plugin,
    )
    result = run_benchmark_instance(
        path,
        format="miplib",
        solver="sg-cqqa",
        time_limit=0.1,
        qqa_config=QQAHeuristicConfig(minimum_core_size=1, minimum_qqa_time=1.0),
    )
    assert result.run_config["qqa_structurally_applicable"] is True
    assert result.run_config["qqa_budget_applicable"] is False
    assert result.run_config["qqa_applicable"] is False
    assert result.run_config["qqa_plugin_active"] is False
    assert result.run_config["torch_threads"] is None


def test_sg_cqqa_skips_plugin_import_without_runtime_startup_reserve(tmp_path, monkeypatch):
    path = tmp_path / "startup-reserve.mps"
    _write_tiny_mps(path)

    def unexpected_plugin(*args, **kwargs):
        raise AssertionError("insufficient startup reserve must not construct the QQA plugin")

    monkeypatch.setattr(
        "qqa.benchmarking.algebraic_runner.include_qqa_heuristic",
        unexpected_plugin,
    )
    result = run_benchmark_instance(
        path,
        format="miplib",
        solver="sg-cqqa",
        time_limit=30.0,
        qqa_config=QQAHeuristicConfig(
            minimum_core_size=1,
            minimum_qqa_time=1.0,
            maximum_overhead_fraction=0.05,
            minimum_runtime_startup_time=8.0,
        ),
    )
    assert result.run_config["qqa_structurally_applicable"] is True
    assert result.run_config["qqa_budget_applicable"] is False
    assert result.run_config["qqa_applicable"] is False
    assert result.run_config["qqa_plugin_active"] is False
    assert result.run_config["torch_threads"] is None


def test_benchmark_campaign_records_path_free_failures_and_continues(tmp_path):
    missing = tmp_path / "not-present.mps.gz"
    result = compare_benchmark_solvers(
        [missing],
        solvers=("scip", "sg-cqqa"),
        baseline_solver="scip",
        format="miplib",
        time_limit=0.1,
        continue_on_error=True,
    )
    assert not result.results
    assert len(result.failures) == 2
    assert result.summary["campaign"] == {
        "requested_runs": 2,
        "completed_runs": 0,
        "failed_runs": 2,
        "failures_by_solver": {"scip": 1, "sg-cqqa": 1},
        "failures_by_type": {"FileNotFoundError": 2},
        "failures_by_outcome": {"backend_failure": 2},
    }
    payload = json.dumps(result.to_dict())
    assert str(tmp_path) not in payload
    assert "FileNotFoundError" in payload


def test_disjoint_campaign_shards_merge_and_recompute_aggregates(tmp_path):
    config = {
        "instances": [],
        "solvers": ["scip-aggressive", "sg-cqqa"],
        "seeds": [0],
        "baseline_solver": "scip-aggressive",
        "format": "miplib",
        "time_limit": 1.0,
        "relative_gap_limit": 0.0,
        "threads": 1,
        "thread_policy": {"scip_parallel": 1, "scip_lp": 1, "torch_sg_cqqa": 1},
        "metric_clock": "solver_wall_clock_after_common_import",
        "reference_name": None,
        "qqa_config": {"seed": 0},
    }

    def result(instance, solver, objective, *, seed=0):
        return BenchmarkResult(
            instance=instance,
            format="miplib",
            solver=solver,
            objective_sense="minimize",
            status="timelimit",
            runtime=1.0,
            solving_time=1.0,
            nodes=1,
            objective=objective,
            dual_bound=0.0,
            gap=None,
            feasible=True,
            maximum_infeasibility=0.0,
            time_to_first_feasible=0.1,
            primal_integral=objective,
            reference_objective=None,
            primal_error=None,
            problem_type="MIPLIB",
            run_config={"seed": seed},
            provenance={"source_name": instance},
        ).to_dict()

    shards = []
    for index, instance in enumerate(("first.mps.gz", "second.mps.gz")):
        path = tmp_path / f"shard-{index}.json{'.gz' if index else ''}"
        shard_config = {**config, "instances": [instance]}
        payload = json.dumps(
            {
                "results": [
                    result(instance, "scip-aggressive", 2.0),
                    result(instance, "sg-cqqa", 1.0),
                ],
                "summary": {},
                "comparison_config": shard_config,
                "failures": [],
            }
        )
        if path.suffix == ".gz":
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(payload)
        else:
            path.write_text(payload, encoding="utf-8")
        shards.append(path)
    merged = merge_benchmark_campaigns(shards)
    assert merged.comparison_config["instances"] == ["first.mps.gz", "second.mps.gz"]
    assert merged.summary["campaign"]["requested_runs"] == 4
    assert merged.summary["campaign"]["completed_runs"] == 4
    assert merged.summary["pairwise"]["sg-cqqa"]["primal_quality"]["wins"] == 2
    with pytest.raises(ValueError, match="duplicate request cell"):
        merge_benchmark_campaigns([shards[0], shards[0]])

    seed_shard = tmp_path / "seed-1.json"
    seed_config = {
        **config,
        "instances": ["first.mps.gz", "second.mps.gz"],
        "seeds": [1],
        "qqa_config": {"seed": 1},
    }
    seed_shard.write_text(
        json.dumps(
            {
                "results": [
                    result(instance, solver, objective, seed=1)
                    for instance in ("first.mps.gz", "second.mps.gz")
                    for solver, objective in (("scip-aggressive", 2.0), ("sg-cqqa", 1.0))
                ],
                "summary": {},
                "comparison_config": seed_config,
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    multiseed = merge_benchmark_campaigns([*shards, seed_shard])
    assert multiseed.comparison_config["seeds"] == [0, 1]
    assert multiseed.comparison_config["qqa_config"] == {}
    assert multiseed.summary["campaign"]["requested_runs"] == 8
    assert multiseed.summary["pairwise"]["sg-cqqa"]["primal_quality"]["wins"] == 4

    outside = json.loads(shards[0].read_text(encoding="utf-8"))
    outside["results"][0]["provenance"]["source_name"] = "outside.mps.gz"
    shards[0].write_text(json.dumps(outside), encoding="utf-8")
    with pytest.raises(ValueError, match="is outside"):
        merge_benchmark_campaigns([shards[0]])


def test_merged_campaign_retains_failure_outcome_taxonomy(tmp_path):
    source = tmp_path / "failures.json"
    source.write_text(
        json.dumps(
            {
                "results": [],
                "summary": {},
                "comparison_config": {
                    "instances": ["hard.qplib"],
                    "solvers": ["scip-aggressive", "sg-cqqa"],
                    "seeds": [0],
                    "baseline_solver": "scip-aggressive",
                    "format": "qplib",
                    "qqa_config": {"seed": 0},
                },
                "failures": [
                    BenchmarkFailure(
                        "hard.qplib", "qplib", "scip-aggressive", 0, "WorkerTimeout"
                    ).to_dict(),
                    BenchmarkFailure("hard.qplib", "qplib", "sg-cqqa", 0, "MemoryError").to_dict(),
                ],
            }
        ),
        encoding="utf-8",
    )
    campaign = merge_benchmark_campaigns([source]).summary["campaign"]
    assert campaign["failures_by_outcome"] == {"out_of_memory": 1, "timeout": 1}


def test_comparison_stratifies_actual_qqa_execution():
    def result(instance, solver, objective, *, qqa=None):
        return BenchmarkResult(
            instance=instance,
            format="miplib",
            solver=solver,
            objective_sense="minimize",
            status="timelimit",
            runtime=1.0,
            solving_time=1.0,
            nodes=1,
            objective=objective,
            dual_bound=0.0,
            gap=None,
            feasible=True,
            maximum_infeasibility=0.0,
            time_to_first_feasible=0.1,
            primal_integral=objective,
            reference_objective=None,
            primal_error=None,
            problem_type="MIPLIB",
            qqa=qqa,
            run_config={"seed": 0},
        )

    summary = summarise_comparison(
        [
            result("executed", "scip-aggressive", 2.0),
            result(
                "executed",
                "sg-cqqa",
                1.0,
                qqa={"calls": 1, "qqa_calls": 1, "qqa_incumbent_improvements": 1},
            ),
            result("bypassed", "scip-aggressive", 1.0),
            result("bypassed", "sg-cqqa", 2.0),
        ],
        baseline_solver="scip-aggressive",
    )
    intervention = summary["pairwise"]["sg-cqqa"]["qqa_intervention"]
    assert intervention["heuristic_invoked_pairs"] == 1
    assert intervention["qqa_executed_pairs"] == 1
    assert intervention["qqa_incumbent_improvement_pairs"] == 1
    assert intervention["executed"]["primal_quality"] == {
        "losses": 0,
        "ties": 0,
        "wins": 1,
    }
    assert intervention["not_executed"]["primal_quality"] == {
        "losses": 1,
        "ties": 0,
        "wins": 0,
    }
    inference = summary["pairwise"]["sg-cqqa"]["inference"]
    assert inference["confidence_unit"] == "instance_after_seed_median"
    assert inference["primal_quality"]["eligible_instances"] == 2
    assert summary["anytime_ecdf"]["sg-cqqa"]["runs"] == 2


def test_public_campaign_artifacts_are_deterministic_and_path_free(tmp_path):
    campaign = {
        "results": [
            {
                "solver": "scip",
                "trajectory": [{"time": 0.1, "primal_bound": 1.0}],
                "solution_sha256": "abc",
                "solution_values": [1.0, 0.0],
            }
        ],
        "summary": {"campaign": {"requested_runs": 0, "completed_runs": 0}},
        "comparison_config": {
            "instances": ["public.mps.gz"],
            "solvers": ["scip"],
            "seeds": [0],
        },
        "failures": [],
    }
    snapshot = {
        "library": "miplib",
        "snapshot": "public-v1",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "files": [
            {
                "name": "benchmark.zip",
                "url": "https://miplib.zib.de/downloads/benchmark.zip",
                "sha256": "abc",
                "size": 1,
            }
        ],
        "extracted_files": ["public.mps.gz"],
    }
    campaign_path = tmp_path / "campaign.json"
    snapshot_path = tmp_path / "snapshot.json"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = publish_benchmark_campaigns(
        {"miplib": campaign_path},
        {"miplib": snapshot_path},
        first,
        implementation_revision="a" * 40,
    )
    publish_benchmark_campaigns(
        {"miplib": campaign_path},
        {"miplib": snapshot_path},
        second,
        implementation_revision="a" * 40,
    )
    assert manifest["implementation_revision"] == "a" * 40
    assert manifest["libraries"]["miplib"]["snapshot"]["instance_count"] == 1
    assert (first / "miplib-campaign.json.gz").read_bytes() == (
        second / "miplib-campaign.json.gz"
    ).read_bytes()
    with gzip.open(first / "miplib-campaign.json.gz", "rt", encoding="utf-8") as stream:
        assert json.load(stream) == campaign
    compact = json.loads((first / "miplib-results.json").read_text(encoding="utf-8"))
    assert "solution_values" not in compact["results"][0]
    assert compact["results"][0]["solution_value_count"] == 2
    assert compact["results"][0]["trajectory_points"] == 1
    assert str(tmp_path) not in json.dumps(manifest)

    campaign["comparison_config"]["source"] = str(tmp_path / "private.mps")
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="Private environment"):
        publish_benchmark_campaigns(
            {"miplib": campaign_path},
            {"miplib": snapshot_path},
            tmp_path / "rejected",
        )
    with pytest.raises(ValueError, match="implementation_revision"):
        publish_benchmark_campaigns(
            {"miplib": campaign_path},
            {"miplib": snapshot_path},
            tmp_path / "bad-revision",
            implementation_revision="not-a-commit",
        )


def test_publication_cli_named_paths_are_explicit_and_unique():
    assert _named_paths(["MIPLIB=campaign.json"], option="--campaign") == {
        "miplib": Path("campaign.json")
    }
    with pytest.raises(ValueError, match="NAME=PATH"):
        _named_paths(["campaign.json"], option="--campaign")
    with pytest.raises(ValueError, match="duplicate"):
        _named_paths(["miplib=one.json", "MIPLIB=two.json"], option="--campaign")


def test_benchmark_metrics_handle_infinite_scip_gap_and_primal_integral():
    assert relative_gap(20.0, 0.0) is None
    assert relative_gap(15.0, 12.0) == pytest.approx(0.25)
    trajectory = [IncumbentPoint(2.0, 20.0), IncumbentPoint(6.0, 11.0)]
    assert primal_integral(trajectory, reference=10.0, horizon=10.0) == pytest.approx(6.4)
    assert primal_integral([], reference=10.0, horizon=10.0) == pytest.approx(10.0)
    assert normalised_primal_error(9.0, 10.0) == 0.0
    assert (
        normalised_primal_error(
            11.0,
            10.0,
            objective_sense="maximize",
        )
        == 0.0
    )


def test_original_objective_tracker_keeps_best_feasible_incumbent():
    evaluations = {
        "first": SimpleNamespace(objective=5.0, maximum_infeasibility=0.0),
        "better": SimpleNamespace(objective=8.0, maximum_infeasibility=0.0),
        "slack_auxiliary": SimpleNamespace(objective=6.0, maximum_infeasibility=0.0),
    }
    active = {"solution": "first"}
    model = SimpleNamespace(getBestSol=lambda: active["solution"])
    tracker = SCIPProgressTracker(
        solution_evaluator=lambda _model, solution: (
            evaluations[solution],
            np.asarray([evaluations[solution].objective]),
        ),
        objective_sense="maximize",
    )
    assert tracker._record_original_evaluation(model) == 5.0
    active["solution"] = "better"
    assert tracker._record_original_evaluation(model) == 8.0
    active["solution"] = "slack_auxiliary"
    assert tracker._record_original_evaluation(model) is None
    assert tracker.best_evaluation.objective == 8.0
    np.testing.assert_array_equal(tracker.best_values, np.asarray([8.0]))


def test_progress_tracker_includes_pre_solve_wall_clock_offset_and_horizon():
    model = SimpleNamespace(
        getSolvingTime=lambda: 2.0,
        getPrimalbound=lambda: 7.0,
        getDualbound=lambda: 5.0,
        infinity=lambda: 1e20,
    )
    tracker = SCIPProgressTracker(time_offset=3.5, time_horizon=6.0)
    tracker.callback(model, None)
    assert tracker.time_to_first_feasible == pytest.approx(5.5)
    assert tracker.trajectory[0].time == pytest.approx(5.5)
    tracker.time_offset = 4.5
    tracker.callback(model, None)
    assert len(tracker.trajectory) == 1


def test_single_qplib_download_normalises_public_identifier(tmp_path, monkeypatch):
    captured = {}

    def fake_download(url, destination, *, overwrite):
        captured.update(url=url, destination=destination, overwrite=overwrite)
        return benchmark_download.DownloadedFile(destination.name, url, "abc", 12)

    monkeypatch.setattr(benchmark_download, "_download", fake_download)
    metadata = benchmark_download.fetch_instance("qplib", "31", tmp_path)
    assert captured["destination"].name == "QPLIB_0031.qplib"
    assert captured["url"].endswith("/QPLIB_0031.qplib")
    assert metadata["file"]["sha256"] == "abc"


def test_benchmark_compare_cli_has_portable_conservative_defaults():
    args = build_parser().parse_args(
        [
            "benchmark",
            "compare",
            "public-instance.mps.gz",
            "--output",
            "comparison.json",
        ]
    )
    assert args.solvers == ("scip-aggressive", "sg-cqqa")
    assert args.baseline_solver == "scip-aggressive"
    assert args.execution_order == "balanced"
    assert args.seeds == (0,)
    assert args.maximum_problem_variables == 32
    assert args.maximum_call_time == pytest.approx(0.15)
    assert args.min_qqa_time == pytest.approx(20.0)
    assert args.minimum_runtime_startup_time == pytest.approx(8.0)
    assert args.fast_candidates == 0
    assert args.maximum_overhead_fraction == pytest.approx(0.05)
    assert args.worker_timeout is None
    assert args.implementation_revision is None
    assert not args.no_equivalent_baseline_reuse
    assert not args.isolate_all
    assert not args.include_import_in_budget
    assert not args.include_solution_values
    assert not args.resume
    assert not args.continue_on_error


@pytest.mark.parametrize(
    ("error_type", "outcome"),
    [
        ("WorkerTimeout", "timeout"),
        ("MemoryError", "out_of_memory"),
        ("UnsupportedFactor", "unsupported"),
        ("NativeSolverProcessError", "backend_failure"),
    ],
)
def test_benchmark_failures_have_portable_outcomes(error_type, outcome):
    failure = BenchmarkFailure("instance", "miplib", "sg-cqqa", 0, error_type)
    assert failure.outcome == outcome


def test_native_worker_exit_types_preserve_only_portable_diagnostics():
    assert _native_process_error_type(None) == "NativeSolverProcessError"
    assert _native_process_error_type(-11) == "NativeSolverSignal11"
    assert _native_process_error_type(2) == "NativeSolverExit2"


def test_default_worker_timeout_has_bounded_budget_relative_grace():
    assert _default_worker_timeout(1.0) == pytest.approx(16.0)
    assert _default_worker_timeout(30.0) == pytest.approx(45.0)
    assert _default_worker_timeout(300.0) == pytest.approx(330.0)
    assert _default_worker_timeout(10_000.0) == pytest.approx(10_060.0)


def test_zero_gap_without_optimal_status_is_not_promoted_to_a_certificate():
    assert _classify_outcome(status="timelimit", feasible=True) == "feasible"
    assert (
        _classify_outcome(status="optimal", feasible=True) == "optimal_with_qualified_certificate"
    )
