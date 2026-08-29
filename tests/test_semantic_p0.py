"""Exhaustive semantic guards for the Phase-0 correctness contract."""

from __future__ import annotations

import itertools
import time

import pytest
import torch

import qqa
from qqa.benchmarking import builtin_benchmark_manifest, holm_adjust, paired_metric_summary
from qqa.blackbox import (
    AsynchronousEvaluationScheduler,
    BlackBoxProblem,
    EvaluationDatabase,
    EvaluationStatus,
)
from qqa.compile import SparseQUBO
from qqa.exact import solve_sat_model_ir
from qqa.model import ModelIR, ObjectiveIR, VariableBlock


def _slow_objective(_point) -> float:
    time.sleep(10.0)
    return 0.0


def test_conditioned_subqubo_preserves_every_completion_energy() -> None:
    qubo = SparseQUBO(
        torch.tensor([1.0, -2.0, 0.5, -0.25]),
        torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]]),
        torch.tensor([3.0, -1.0, 2.0, 0.75]),
        1.25,
    )
    conditioned = qubo.conditioned_subqubo(torch.tensor([1, 3]), torch.tensor([1.0, 0.0]))
    for free_values in itertools.product((0.0, 1.0), repeat=2):
        free = torch.tensor([free_values])
        full = torch.tensor([[free_values[0], 1.0, free_values[1], 0.0]])
        assert conditioned.energy(free).item() == pytest.approx(qubo.energy(full).item())


def test_sat_deadline_rejects_invalid_values_before_optional_import() -> None:
    model = ModelIR((VariableBlock("x", "binary", (1,)),), ObjectiveIR(()))
    with pytest.raises(ValueError, match="time_limit"):
        solve_sat_model_ir(model, time_limit=0.0)


def test_blackbox_cache_identity_includes_seed_fidelity_replicate_and_version(tmp_path) -> None:
    problem = BlackBoxProblem(
        [qqa.Binary("x")],
        lambda point: float(point["x"]),
        evaluator_version="v2",
    )
    database = EvaluationDatabase(tmp_path / "observations.sqlite")
    point = torch.tensor([1.0])
    with AsynchronousEvaluationScheduler(
        problem,
        database=database,
        seed=7,
        fidelity="high",
    ) as scheduler:
        first = scheduler.submit(point, replicate=0).result()
        repeated = scheduler.submit(point, replicate=1).result()
    assert first.status is EvaluationStatus.COMPLETED
    assert repeated.status is EvaluationStatus.COMPLETED
    assert database.get(
        problem.name,
        point,
        seed=7,
        fidelity="high",
        replicate=0,
        evaluator_version="v2",
    ) == first
    assert database.get(
        problem.name,
        point,
        seed=8,
        fidelity="high",
        replicate=0,
        evaluator_version="v2",
    ) is None


def test_blackbox_hard_timeout_terminates_an_isolated_evaluation() -> None:
    problem = BlackBoxProblem([qqa.Binary("x")], _slow_objective)
    started = time.perf_counter()
    with AsynchronousEvaluationScheduler(problem, timeout=0.05) as scheduler:
        record = scheduler.submit(torch.tensor([0.0])).result()
    assert record.status is EvaluationStatus.TIMED_OUT
    # Leave ample process-startup headroom on network filesystems while still
    # proving that the ten-second evaluator was terminated rather than joined.
    assert time.perf_counter() - started < 5.0


def test_study_ask_tell_and_qqa_campaign_contract(tmp_path) -> None:
    problem = BlackBoxProblem(
        [qqa.Real("x", -1.0, 1.0)],
        lambda point: (point["x"] - 0.2) ** 2,
        name="study-contract",
    )
    study = qqa.create_study(
        problem,
        storage=tmp_path / "study.sqlite",
        seed=3,
    )
    trial = study.ask(fidelity="coarse")
    value, violations, _ = problem.evaluate_one(trial.packed)
    study.tell(trial, value=value, violations=tuple(violations))
    assert study.best_trial is trial
    result = study.optimize(
        budget=6,
        initial_points=4,
        batch_size=2,
        candidate_pool=32,
        qqa_acquisition_epochs=2,
        qqa_acquisition_replicas=4,
    )
    assert result.metadata["acquisition_optimizer"] == "qqa"
    assert len(study.trials) == result.evaluations


def test_benchmark_hub_manifest_and_paired_statistics_are_portable() -> None:
    manifest = builtin_benchmark_manifest()
    assert manifest.track.value == "qqa-primal"
    assert len(manifest.instances) == 2
    summary = paired_metric_summary(
        (1.0, 2.0, 3.0, 4.0),
        (2.0, 2.5, 2.5, 5.0),
        bootstrap_samples=200,
        seed=4,
    )
    assert summary.wins == 3
    assert summary.losses == 1
    assert 0.0 <= summary.sign_test_pvalue <= 1.0
    adjusted = holm_adjust({"quality": 0.01, "runtime": 0.04})
    assert adjusted == {"quality": 0.02, "runtime": 0.04}
