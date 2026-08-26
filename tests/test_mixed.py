"""Correctness and validation tests for mixed-variable optimisation."""

from __future__ import annotations

import pytest
import torch

import qqa


def test_variable_space_round_trip_and_named_views():
    problem = qqa.MixedProblem(
        [
            qqa.Binary("enabled", size=2),
            qqa.Integer("units", lower=-2, upper=5),
            qqa.Real("temperature", lower=10.0, upper=20.0),
        ],
        lambda v: v["enabled"].sum(dim=-1) + v["units"] + v["temperature"],
    )
    packed = problem.pack(
        {"enabled": [1, 0], "units": 3, "temperature": 12.5},
    )
    assert packed.tolist() == [1.0, 0.0, 3.0, 12.5]
    named = problem.unpack(packed)
    assert named["enabled"].tolist() == [1.0, 0.0]
    assert named["units"].item() == 3
    assert named["temperature"].item() == 12.5

    latent = problem.relaxation.encode(packed)
    torch.testing.assert_close(problem.relaxation.project(latent), packed)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: qqa.Binary("bad-name"),
        lambda: qqa.Integer("x", 2, 2),
        lambda: qqa.Real("x", float("-inf"), 1.0),
    ],
)
def test_variable_declarations_reject_invalid_domains(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_variable_space_rejects_nonfinite_solution_values():
    space = qqa.VariableSpace((qqa.Real("x", 0.0, 1.0),))
    with pytest.raises(ValueError, match="NaN or infinity"):
        space.validate(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="NaN or infinity"):
        space.validate(torch.tensor([float("inf")]))


def test_variable_kind_cannot_be_overridden():
    with pytest.raises(TypeError):
        qqa.Binary("x", kind="real")


def test_real_problem_converges_without_rounding():
    qqa.fix_seed(0)
    problem = qqa.MixedProblem(
        [qqa.Real("x", -5.0, 5.0), qqa.Real("y", -5.0, 5.0)],
        lambda v: (v["x"] - 1.25).square() + (v["y"] + 2.5).square(),
        name="convex-real",
    )
    result = problem.solve(sol_size=16, num_epochs=250, verbose=False)
    assert result.best_obj < 1e-4
    assert result.best_sol[0].item() == pytest.approx(1.25, abs=0.02)
    assert result.best_sol[1].item() == pytest.approx(-2.5, abs=0.02)


def test_integer_problem_projects_to_exact_grid_optimum():
    qqa.fix_seed(1)
    problem = qqa.MixedProblem(
        [qqa.Integer("quantity", lower=-10, upper=10)],
        lambda v: (v["quantity"] - 3).square(),
        name="integer-quadratic",
    )
    result = problem.solve(sol_size=8, num_epochs=100, verbose=False)
    assert result.best_obj == 0.0
    assert result.best_sol.item() == 3.0


def test_float64_preserves_large_integer_grid_points():
    problem = qqa.MixedProblem(
        [qqa.Integer("identifier", 100_000_000, 100_000_010)],
        lambda v: (v["identifier"] - 100_000_003).square(),
        dtype=torch.float64,
    )
    packed = problem.pack({"identifier": 100_000_003})
    assert packed.dtype == torch.float64
    projected = problem.relaxation.project(problem.relaxation.encode(packed))
    assert projected.item() == 100_000_003


def test_practical_mixed_factory_problem_finds_known_optimum():
    """Binary activation + integer batches + real overtime (MINLP)."""
    qqa.fix_seed(0)
    problem = qqa.MixedProblem(
        [
            qqa.Binary("machine", size=2),
            qqa.Integer("batches", lower=0, upper=6, size=2),
            qqa.Real("overtime", lower=0.0, upper=4.0),
        ],
        lambda v: (
            10 * v["machine"].sum(dim=-1)
            + 3 * v["batches"].sum(dim=-1)
            + 2 * v["overtime"].square()
        ),
        constraints=[
            qqa.Constraint(
                lambda v: 4 * v["batches"].sum(dim=-1) + v["overtime"],
                sense=">=",
                rhs=28,
                weight=100,
                name="demand",
            ),
            qqa.Constraint(
                lambda v: (v["batches"] - 6 * v["machine"]).clamp_min(0).sum(dim=-1),
                sense="<=",
                rhs=0,
                weight=100,
                name="activation_link",
            ),
        ],
        name="factory-planning",
        objective_label="cost",
        objective_unit="kUSD",
    )
    result = problem.solve(sol_size=128, num_epochs=600, verbose=False)

    assert result.best_obj == pytest.approx(41.0)
    assert result.score["value"] == pytest.approx(41.0)
    assert result.score["feasible"] is True
    assert result.score["extra"]["variables"]["machine"] == [1.0, 1.0]
    assert sum(result.score["extra"]["variables"]["batches"]) == 7.0
    assert result.score["extra"]["variables"]["overtime"] == pytest.approx(0.0)


def test_mixed_solver_prefers_a_feasible_replica_and_does_not_mutate_problem(monkeypatch):
    from qqa.annealing import AnnealResult

    problem = qqa.MixedProblem(
        [qqa.Real("x", 0.0, 1.0)],
        lambda v: v["x"],
        constraints=[
            qqa.Constraint(
                lambda v: v["x"],
                sense=">=",
                rhs=0.5,
                weight=1.0,
                name="minimum",
            )
        ],
    )
    problem.penalty_multiplier = 17.0
    seen = {}

    def fake_anneal(solving_problem, **kwargs):
        seen["problem"] = solving_problem
        seen["multiplier"] = solving_problem.penalty_multiplier
        return AnnealResult(
            best_sol=torch.tensor([0.0]),
            best_obj=0.25,
            runtime=0.0,
            final_population=torch.tensor([[0.0], [0.5], [0.8]]),
        )

    monkeypatch.setattr("qqa.mixed.solve.anneal", fake_anneal)
    result = qqa.solve_mixed(problem, calibrate_penalty=False)
    assert seen["problem"] is not problem
    assert seen["multiplier"] == 1.0
    assert problem.penalty_multiplier == 17.0
    assert result.best_sol.item() == pytest.approx(0.5)
    assert result.score["feasible"]
    assert result.final_population is None


def test_penalty_calibration_is_invariant_to_large_objective_offsets(monkeypatch):
    from qqa.annealing import AnnealResult

    captured = []

    def fake_anneal(solving_problem, **kwargs):
        captured.append(solving_problem.penalty_multiplier)
        return AnnealResult(
            best_sol=torch.tensor([0.8], dtype=torch.float64),
            best_obj=1e9,
            runtime=0.0,
            final_population=torch.tensor([[0.8], [0.5]], dtype=torch.float64),
        )

    monkeypatch.setattr("qqa.mixed.solve.anneal", fake_anneal)
    for offset in (0.0, 1e9):
        problem = qqa.MixedProblem(
            [qqa.Real("x", 0.0, 1.0)],
            lambda v, offset=offset: offset + (v["x"] - 0.8).square(),
            constraints=[
                qqa.Constraint(
                    lambda v: v["x"],
                    sense=">=",
                    rhs=0.4,
                    weight=1.0,
                    name="minimum",
                )
            ],
            dtype=torch.float64,
        )
        qqa.solve_mixed(problem, calibration_points=32)
    assert captured[0] == pytest.approx(captured[1], rel=1e-8)
    assert captured[1] < 1e5


def test_mixed_objective_must_preserve_population_axis():
    problem = qqa.MixedProblem(
        [qqa.Real("x", 0.0, 1.0)],
        lambda v: v["x"].sum(),
    )
    with pytest.raises(ValueError, match="leading population"):
        problem.loss_fn(torch.rand(4, 1))


def test_anneal_validates_dangerous_numeric_options():
    problem = qqa.MixedProblem([qqa.Real("x", 0.0, 1.0)], lambda v: v["x"].square())
    with pytest.raises(ValueError, match="curve_rate"):
        qqa.anneal(problem, curve_rate=0, num_epochs=0, verbose=False)
    with pytest.raises(ValueError, match="check_interval"):
        qqa.anneal(problem, check_interval=0, num_epochs=0, verbose=False)
    with pytest.raises(ValueError, match="div_param"):
        qqa.anneal(problem, div_param=1.1, num_epochs=0, verbose=False)
    with pytest.raises(ValueError, match="sol_size"):
        qqa.anneal(problem, sol_size=1.5, num_epochs=0, verbose=False)
    with pytest.raises(ValueError, match="num_epochs"):
        qqa.anneal(problem, num_epochs=1.5, verbose=False)
    with pytest.raises(ValueError, match="learning_rate"):
        qqa.anneal(problem, learning_rate=float("nan"), num_epochs=0, verbose=False)
    with pytest.raises(ValueError, match="weight_decay"):
        qqa.anneal(problem, weight_decay=-1, num_epochs=0, verbose=False)
    with pytest.raises(ValueError, match="optimizer"):
        qqa.anneal(problem, optimizer="unknown", num_epochs=0, verbose=False)
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        qqa.anneal(problem, gradient_clip_norm=0, num_epochs=0, verbose=False)
    with pytest.raises(ValueError, match="restart_patience"):
        qqa.anneal(problem, restart_patience=0, num_epochs=0, verbose=False)
    with pytest.raises(ValueError, match="restart_fraction"):
        qqa.anneal(problem, restart_fraction=1, num_epochs=0, verbose=False)


def test_anneal_adaptive_restarts_preserve_incumbent_and_report_diagnostics():
    problem = qqa.MixedProblem(
        [qqa.Real("x", 0.0, 1.0)],
        lambda v: 0.0 * v["x"],
    )
    result = qqa.anneal(
        problem,
        sol_size=8,
        learning_rate=0.1,
        num_epochs=4,
        restart_patience=1,
        restart_fraction=0.5,
        restart_jitter=0.05,
        return_population=True,
        polish=False,
        verbose=False,
    )
    assert result.best_obj == pytest.approx(0.0)
    assert result.history["restart_count"] > 0
    assert result.diagnostics["restart_events"] == len(result.history["restart_epochs"])
    assert result.diagnostics["weight_decay"] == 0.0
    assert result.final_population.shape == (8, 1)


def test_lightweight_adamw_runs_short_hybrid_anneals_without_optimizer_discovery():
    problem = qqa.MixedProblem(
        [qqa.Binary("x"), qqa.Binary("y")],
        lambda values: -values["x"] - 2.0 * values["y"],
    )
    result = qqa.anneal(
        problem,
        sol_size=4,
        learning_rate=0.05,
        num_epochs=4,
        optimizer="lightweight-adamw",
        polish=False,
        verbose=False,
    )
    assert result.diagnostics["optimizer"] == "lightweight-adamw"
    assert result.best_sol.shape == (2,)


def test_qubo_polish_handles_non_symmetric_user_matrix():
    class NonSymmetricQUBO:
        Q_mat = torch.tensor([[-3.0, -3.0], [-1.0, 2.0]])

        def loss_fn(self, x):
            return torch.einsum("bi,ij,bj->b", x, self.Q_mat, x)

    start = torch.tensor([1.0, 0.0])
    polished = qqa.polish.greedy_one_flip(NonSymmetricQUBO(), start)
    assert polished.tolist() == [1.0, 1.0]
    assert NonSymmetricQUBO().loss_fn(polished[None]).item() == pytest.approx(-5.0)


@pytest.mark.parametrize(
    "diversity,expected",
    [
        (0.1, 0.8),  # below target -> stronger diversity pressure
        (0.5, 0.4),  # above target -> weaker diversity pressure
    ],
)
def test_auto_div_tuner_is_population_invariant_negative_feedback(diversity, expected):
    problem = qqa.MixedProblem([qqa.Real("x", 0.0, 1.0)], lambda v: v["x"].square())
    tuner = qqa.AutoDivTuner(target=0.4, lr=1.0)
    state = qqa.CallbackState(
        epoch=0,
        num_epochs=1,
        bg=0.0,
        x=torch.zeros(100, 1),
        losses=torch.zeros(100),
        penalties=torch.zeros(100),
        diversity=torch.tensor(diversity),
        best_obj=0.0,
        hyperparams={"div_param": 0.5},
        problem=problem,
        relaxation=problem.relaxation,
    )
    tuner.on_epoch_end(state)
    assert state.hyperparams["div_param"] == pytest.approx(expected)
