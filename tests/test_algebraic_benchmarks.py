"""Sparse algebraic IR and MIPLIB/QPLIB hybrid regression tests."""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import replace
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
from qqa.benchmarking.algebraic_runner import _configure_scip_threads
from qqa.benchmarking.metrics import (
    IncumbentPoint,
    SCIPProgressTracker,
    normalised_primal_error,
    primal_integral,
    relative_gap,
)
from qqa.cli import build_parser
from qqa.decomposition import (
    CompletionResult,
    complete_integer_assignment,
    create_completion_template,
)
from qqa.hybrid import scip_heuristic as scip_heuristic_module
from qqa.hybrid.core_selector import CoreSelection, select_uncertain_integer_core
from qqa.hybrid.neighborhood import IntegerNeighborhood
from qqa.hybrid.nonconvex import dc_decomposition, linearize_concave_part
from qqa.hybrid.scip_heuristic import QQAHeuristic, QQAHeuristicConfig
from qqa.hybrid.surrogate import build_core_surrogate, generate_surrogate_candidates
from qqa.io import load_mps, load_qplib
from qqa.mixed import ConstraintArchive
from qqa.mixed.augmented_lagrangian import AdaptiveAugmentedLagrangian
from qqa.presolve import SCIPState, build_scip_model, scaled_model


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
    assert selection.local_lower[position] == 100
    assert selection.local_upper[position] == 101
    assert 2 in selection.fixed_indices

    rins_state = replace(
        state,
        incumbent_values=np.asarray([1.0, 101.0, 2.0, 0.0]),
    )
    assert select_uncertain_integer_core(rins_state, max_core_size=2).mode == "rins"


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
    )
    assert [row.sense for row in core_problem.constraints] == ["<="]
    assert core_problem.constraints[0].rhs == pytest.approx(1.0)
    controller = AdaptiveAugmentedLagrangian.for_problem(core_problem)
    violating = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    before = controller.penalty(core_problem, violating).item()
    controller.update(core_problem, violating)
    assert controller.penalty(core_problem, violating).item() > before


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


def test_conditional_heuristic_releases_fixed_complement_for_lns_repair(monkeypatch):
    calls = []

    def fake_completion(template, variable_names, values, **kwargs):
        calls.append((tuple(variable_names), tuple(values)))
        repaired = len(variable_names) == 1
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
        incumbent_values=None,
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
    assert calls == [(("core", "complement"), (1.0, 0.0)), (("core",), (1.0,))]
    assert accepted and improved
    assert heuristic.stats.lns_repair_attempts == 1
    assert heuristic.stats.lns_repair_feasible == 1
    assert heuristic.stats.lns_repair_accepted == 1
    active_model.free()


def test_conditional_heuristic_broadens_an_infeasible_core_lns(monkeypatch):
    fixed_counts = []

    def fake_completion(template, variable_names, values, **kwargs):
        fixed_counts.append(len(variable_names))
        feasible = len(variable_names) == 1
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
        QQAHeuristicConfig(completion_time=1.0, maximum_overhead_fraction=0.5),
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
    assert fixed_counts == [5, 4, 1]
    assert accepted and improved
    assert heuristic.stats.partial_lns_attempts == 1
    assert heuristic.stats.partial_lns_feasible == 1
    assert heuristic.stats.partial_lns_accepted == 1
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
        common_import=True,
    )
    assert result.status == "optimal"
    assert result.feasible
    assert result.objective == pytest.approx(-1.0)
    assert str(tmp_path) not in json.dumps(result.to_dict())


def test_sg_cqqa_continuous_model_uses_matched_aggressive_scip_fallback(tmp_path, monkeypatch):
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
        "qqa_config": {},
    }

    def result(instance, solver, objective):
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
            run_config={"seed": 0},
            provenance={"source_name": instance},
        ).to_dict()

    shards = []
    for index, instance in enumerate(("first.mps.gz", "second.mps.gz")):
        path = tmp_path / f"shard-{index}.json"
        shard_config = {**config, "instances": [instance]}
        path.write_text(
            json.dumps(
                {
                    "results": [
                        result(instance, "scip-aggressive", 2.0),
                        result(instance, "sg-cqqa", 1.0),
                    ],
                    "summary": {},
                    "comparison_config": shard_config,
                    "failures": [],
                }
            ),
            encoding="utf-8",
        )
        shards.append(path)
    merged = merge_benchmark_campaigns(shards)
    assert merged.comparison_config["instances"] == ["first.mps.gz", "second.mps.gz"]
    assert merged.summary["campaign"]["requested_runs"] == 4
    assert merged.summary["campaign"]["completed_runs"] == 4
    assert merged.summary["pairwise"]["sg-cqqa"]["primal_quality"]["wins"] == 2
    with pytest.raises(ValueError, match="duplicate instance"):
        merge_benchmark_campaigns([shards[0], shards[0]])
    outside = json.loads(shards[0].read_text(encoding="utf-8"))
    outside["results"][0]["provenance"]["source_name"] = "outside.mps.gz"
    shards[0].write_text(json.dumps(outside), encoding="utf-8")
    with pytest.raises(ValueError, match="is outside"):
        merge_benchmark_campaigns([shards[0]])


def test_public_campaign_artifacts_are_deterministic_and_path_free(tmp_path):
    campaign = {
        "results": [],
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
    assert args.solvers == ("scip", "scip-aggressive", "sg-cqqa")
    assert args.baseline_solver == "scip"
    assert args.seeds == (0,)
    assert args.min_qqa_time == 20.0
    assert args.fast_candidates == 2
    assert args.maximum_overhead_fraction == 0.1
    assert not args.resume
    assert not args.continue_on_error
