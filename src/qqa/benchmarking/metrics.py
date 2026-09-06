"""Reproducible MIP/QP benchmark metrics and incumbent tracking."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, cast

from qqa.benchmarking.hub import paired_metric_summary


@dataclass(frozen=True, slots=True)
class IncumbentPoint:
    time: float
    primal_bound: float
    dual_bound: float | None = None


def relative_gap(primal: float | None, dual: float | None) -> float | None:
    if primal is None or dual is None or not math.isfinite(primal) or not math.isfinite(dual):
        return None
    if abs(primal - dual) <= 1e-12:
        return 0.0
    # Match SCIP's public gap definition. Opposite signs, or a zero bound
    # paired with a nonzero bound, produce an infinite gap; JSON-facing
    # results represent that state as None rather than a non-finite number.
    if primal * dual <= 0:
        return None
    return abs(primal - dual) / min(abs(primal), abs(dual))


def primal_integral(
    trajectory: Sequence[IncumbentPoint],
    *,
    reference: float,
    horizon: float,
    objective_sense: str = "minimize",
) -> float | None:
    """Integrate the normalised primal error over wall-clock time."""
    if not math.isfinite(reference) or not math.isfinite(horizon) or horizon <= 0:
        return None
    if objective_sense not in {"minimize", "maximize"}:
        raise ValueError("objective_sense must be minimize or maximize.")
    points = sorted(
        (point for point in trajectory if point.time <= horizon), key=lambda row: row.time
    )
    if not points:
        return float(horizon)
    scale = max(1.0, abs(reference))
    integral = 0.0
    previous_time = 0.0
    previous_error = 1.0
    for point in points:
        current_time = max(previous_time, float(point.time))
        integral += (current_time - previous_time) * previous_error
        difference = (
            point.primal_bound - reference
            if objective_sense == "minimize"
            else reference - point.primal_bound
        )
        previous_error = min(1.0, max(0.0, difference) / scale)
        previous_time = current_time
    integral += max(0.0, horizon - previous_time) * previous_error
    return float(integral)


def normalised_primal_error(
    objective: float | None,
    reference: float | None,
    *,
    objective_sense: str = "minimize",
) -> float | None:
    """Return a directional reference error in scale-independent units."""
    if objective_sense not in {"minimize", "maximize"}:
        raise ValueError("objective_sense must be minimize or maximize.")
    if (
        objective is None
        or reference is None
        or not math.isfinite(objective)
        or not math.isfinite(reference)
    ):
        return None
    difference = objective - reference if objective_sense == "minimize" else reference - objective
    return max(0.0, difference) / max(1.0, abs(reference))


@dataclass(slots=True)
class SCIPProgressTracker:
    solution_evaluator: Callable[[Any, Any], Any] | None = None
    objective_sense: str = "minimize"
    time_offset: float = 0.0
    time_horizon: float | None = None
    trajectory: list[IncumbentPoint] = field(default_factory=list)
    time_to_first_feasible: float | None = None
    best_evaluation: Any | None = None
    best_values: Any | None = None

    def __post_init__(self) -> None:
        if self.objective_sense not in {"minimize", "maximize"}:
            raise ValueError("objective_sense must be minimize or maximize.")
        if not math.isfinite(self.time_offset) or self.time_offset < 0:
            raise ValueError("time_offset must be finite and >= 0.")
        if self.time_horizon is not None and (
            not math.isfinite(self.time_horizon) or self.time_horizon <= 0
        ):
            raise ValueError("time_horizon must be finite and > 0, or None.")

    def _record_original_evaluation(self, model) -> float | None:
        if self.solution_evaluator is None:
            return None
        solution = model.getBestSol()
        if solution is None:
            return None
        evaluated = self.solution_evaluator(model, solution)
        if isinstance(evaluated, tuple) and len(evaluated) == 2:
            evaluation, values = evaluated
        else:
            evaluation, values = evaluated, None
        if not math.isfinite(float(evaluation.objective)):
            return None
        if float(evaluation.maximum_infeasibility) > 1e-6:
            return None
        if self.best_evaluation is None:
            self.best_evaluation = evaluation
            self.best_values = values
            return float(evaluation.objective)
        current = float(self.best_evaluation.objective)
        candidate = float(evaluation.objective)
        tolerance = 1e-10 * max(1.0, abs(current), abs(candidate))
        improved = (
            candidate > current + tolerance
            if self.objective_sense == "maximize"
            else candidate < current - tolerance
        )
        if improved:
            self.best_evaluation = evaluation
            self.best_values = values
            return candidate
        return None

    def callback(self, model, event) -> None:  # noqa: ARG002 - callback API
        time = self.time_offset + float(model.getSolvingTime())
        if self.time_horizon is not None and time > self.time_horizon:
            return
        primal = float(model.getPrimalbound())
        if self.solution_evaluator is not None:
            try:
                original = self._record_original_evaluation(model)
                if original is None:
                    return
                primal = original
            except Exception:
                # The transformed auxiliary objective is not a safe fallback
                # for nonlinear QPLIB models. Omit an unverified event.
                return
        dual: float | None
        try:
            dual = float(model.getDualbound())
            if not math.isfinite(dual):
                dual = None
        except Exception:
            dual = None
        infinity = abs(float(model.infinity()))
        if not math.isfinite(primal) or abs(primal) >= 0.99 * infinity:
            return
        if self.time_to_first_feasible is None:
            self.time_to_first_feasible = time
        self.trajectory.append(IncumbentPoint(time, primal, dual))

    def attach(self, model) -> None:
        from pyscipopt import SCIP_EVENTTYPE

        model.attachEventHandlerCallback(
            self.callback,
            [SCIP_EVENTTYPE.BESTSOLFOUND],
            name="qqa_benchmark_progress",
            description="Track incumbent timing for reproducible benchmark metrics",
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    instance: str
    format: str
    solver: str
    objective_sense: str
    status: str
    runtime: float
    solving_time: float
    nodes: int
    objective: float | None
    dual_bound: float | None
    gap: float | None
    feasible: bool
    maximum_infeasibility: float | None
    time_to_first_feasible: float | None
    primal_integral: float | None
    reference_objective: float | None
    primal_error: float | None
    problem_type: str | None
    trajectory: tuple[IncumbentPoint, ...] = ()
    qqa: dict[str, Any] | None = None
    run_config: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    time_to_target: float | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    peak_memory_mb: dict[str, float] = field(default_factory=dict)
    outcome: str = "unknown"
    solution_sha256: str | None = None
    solution_values: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trajectory"] = [asdict(point) for point in self.trajectory]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkResult:
        """Restore a portable checkpoint row without trusting extra fields."""
        values = dict(payload)
        values["trajectory"] = tuple(
            IncumbentPoint(**point) for point in values.get("trajectory", ())
        )
        values["solution_values"] = tuple(values.get("solution_values", ()))
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in allowed})


def _failure_outcome(error_type: str) -> str:
    normalised = error_type.lower()
    if "timeout" in normalised:
        return "timeout"
    if "outofmemory" in normalised or "memoryerror" in normalised or "oom" in normalised:
        return "out_of_memory"
    if "unsupported" in normalised or "notimplemented" in normalised:
        return "unsupported"
    return "backend_failure"


@dataclass(frozen=True, slots=True)
class BenchmarkFailure:
    """Path-free record for one failed campaign run."""

    instance: str
    format: str
    solver: str
    seed: int
    error_type: str
    outcome: str = "backend_failure"

    def __post_init__(self) -> None:
        if self.outcome == "backend_failure":
            object.__setattr__(self, "outcome", _failure_outcome(self.error_type))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkFailure:
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in payload.items() if key in allowed})


def _median(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(median(finite)) if finite else None


def summarise_benchmarks(results: Sequence[BenchmarkResult]) -> dict[str, object]:
    """Aggregate portable metrics overall and by QPLIB ``PROBTYPE``."""

    def one_group(rows: Sequence[BenchmarkResult]) -> dict[str, object]:
        qqa_rows = [row.qqa for row in rows if row.qqa is not None]
        plugin_active_runs = sum(bool(row.run_config.get("qqa_plugin_active")) for row in rows)
        reused_runs = sum(bool(row.run_config.get("equivalent_baseline_reuse")) for row in rows)
        qqa_executed_runs = sum(int(row.get("qqa_calls", 0)) > 0 for row in qqa_rows)
        candidates = sum(int(row.get("candidates", 0)) for row in qqa_rows)
        completed = sum(int(row.get("completion_feasible", 0)) for row in qqa_rows)
        accepted = sum(int(row.get("accepted", 0)) for row in qqa_rows)
        improvements = sum(int(row.get("incumbent_improvements", 0)) for row in qqa_rows)
        qqa_candidates = sum(int(row.get("qqa_candidates", 0)) for row in qqa_rows)
        qqa_completed = sum(int(row.get("qqa_completion_feasible", 0)) for row in qqa_rows)
        qqa_accepted = sum(int(row.get("qqa_accepted", 0)) for row in qqa_rows)
        qqa_improvements = sum(int(row.get("qqa_incumbent_improvements", 0)) for row in qqa_rows)
        repair_attempts = sum(int(row.get("lns_repair_attempts", 0)) for row in qqa_rows)
        repair_feasible = sum(int(row.get("lns_repair_feasible", 0)) for row in qqa_rows)
        repair_accepted = sum(int(row.get("lns_repair_accepted", 0)) for row in qqa_rows)
        repair_improvements = sum(
            int(row.get("lns_repair_incumbent_improvements", 0)) for row in qqa_rows
        )
        partial_attempts = sum(int(row.get("partial_lns_attempts", 0)) for row in qqa_rows)
        partial_feasible = sum(int(row.get("partial_lns_feasible", 0)) for row in qqa_rows)
        partial_accepted = sum(int(row.get("partial_lns_accepted", 0)) for row in qqa_rows)
        partial_improvements = sum(
            int(row.get("partial_lns_incumbent_improvements", 0)) for row in qqa_rows
        )
        return {
            "instances": len(rows),
            "feasible": sum(int(row.feasible) for row in rows),
            "feasible_rate": (sum(int(row.feasible) for row in rows) / len(rows) if rows else 0.0),
            "median_runtime": _median([row.runtime for row in rows]),
            "median_time_to_first_feasible": _median([row.time_to_first_feasible for row in rows]),
            "median_time_to_target": _median([row.time_to_target for row in rows]),
            "median_gap": _median([row.gap for row in rows]),
            "median_primal_error": _median([row.primal_error for row in rows]),
            "median_primal_integral": _median([row.primal_integral for row in rows]),
            "median_maximum_infeasibility": _median([row.maximum_infeasibility for row in rows]),
            "median_peak_process_memory_mb": _median(
                [row.peak_memory_mb.get("process_rss") for row in rows]
            ),
            "outcomes": {
                outcome: sum(row.outcome == outcome for row in rows)
                for outcome in sorted({row.outcome for row in rows})
            },
            "independently_executed_runs": len(rows) - reused_runs,
            "equivalent_baseline_reuse_runs": reused_runs,
            "qqa_plugin_active_runs": plugin_active_runs,
            "qqa_executed_runs": qqa_executed_runs,
            "qqa_active_rate": qqa_executed_runs / len(rows) if rows else 0.0,
            "hybrid_candidates": candidates,
            "hybrid_completion_rate": completed / candidates if candidates else 0.0,
            "hybrid_acceptance_rate": accepted / candidates if candidates else 0.0,
            "hybrid_incumbent_improvement_rate": (improvements / candidates if candidates else 0.0),
            "qqa_candidates": qqa_candidates,
            "qqa_completion_rate": qqa_completed / qqa_candidates if qqa_candidates else 0.0,
            "qqa_acceptance_rate": qqa_accepted / qqa_candidates if qqa_candidates else 0.0,
            "qqa_incumbent_improvement_rate": (
                qqa_improvements / qqa_candidates if qqa_candidates else 0.0
            ),
            "lns_repair_attempts": repair_attempts,
            "lns_repair_feasibility_rate": (
                repair_feasible / repair_attempts if repair_attempts else 0.0
            ),
            "lns_repair_acceptance_rate": (
                repair_accepted / repair_attempts if repair_attempts else 0.0
            ),
            "lns_repair_incumbent_improvement_rate": (
                repair_improvements / repair_attempts if repair_attempts else 0.0
            ),
            "partial_lns_attempts": partial_attempts,
            "partial_lns_feasibility_rate": (
                partial_feasible / partial_attempts if partial_attempts else 0.0
            ),
            "partial_lns_acceptance_rate": (
                partial_accepted / partial_attempts if partial_attempts else 0.0
            ),
            "partial_lns_incumbent_improvement_rate": (
                partial_improvements / partial_attempts if partial_attempts else 0.0
            ),
        }

    rows = tuple(results)
    groups: dict[str, list[BenchmarkResult]] = {}
    for row in rows:
        key = row.problem_type or row.format.upper()
        groups.setdefault(key, []).append(row)
    return {
        "overall": one_group(rows),
        "by_problem_type": {key: one_group(group) for key, group in sorted(groups.items())},
    }


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteResult:
    results: tuple[BenchmarkResult, ...]
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "results": [result.to_dict() for result in self.results],
            "summary": self.summary,
        }


def _compare_primary(candidate: BenchmarkResult, baseline: BenchmarkResult) -> int:
    if candidate.feasible != baseline.feasible:
        return 1 if candidate.feasible else -1
    if not candidate.feasible:
        return 0
    if candidate.primal_error is not None and baseline.primal_error is not None:
        left = candidate.primal_error
        right = baseline.primal_error
        tolerance = 1e-8 * max(1.0, abs(left), abs(right))
        return 1 if left < right - tolerance else -1 if left > right + tolerance else 0
    if candidate.objective is None or baseline.objective is None:
        return 0
    tolerance = 1e-8 * max(1.0, abs(candidate.objective), abs(baseline.objective))
    difference = candidate.objective - baseline.objective
    if abs(difference) <= tolerance:
        return 0
    if candidate.objective_sense == "maximize":
        return 1 if difference > 0 else -1
    return 1 if difference < 0 else -1


def _compare_lower(candidate: float | None, baseline: float | None) -> int:
    if candidate is None or baseline is None:
        return 0
    tolerance = 1e-8 * max(1.0, abs(candidate), abs(baseline))
    return 1 if candidate < baseline - tolerance else -1 if candidate > baseline + tolerance else 0


def _empty_outcome_bucket() -> dict[str, object]:
    return {
        "paired_runs": 0,
        "primal_quality": {"losses": 0, "ties": 0, "wins": 0},
        "primal_integral": {"losses": 0, "ties": 0, "wins": 0},
    }


def _record_outcome(
    bucket: dict[str, object],
    *,
    primary: int,
    integral: int,
) -> None:
    labels = ("losses", "ties", "wins")
    bucket["paired_runs"] = cast(int, bucket["paired_runs"]) + 1
    primal_quality = cast(dict[str, int], bucket["primal_quality"])
    primal_integral_counts = cast(dict[str, int], bucket["primal_integral"])
    primal_quality[labels[primary + 1]] += 1
    primal_integral_counts[labels[integral + 1]] += 1


def _paired_instance_effects(
    grouped: dict[tuple[str, int], dict[str, BenchmarkResult]],
    *,
    solver: str,
    baseline_solver: str,
    metric: str,
) -> tuple[list[float], int]:
    """Return seed-median effects, with each instance counted only once."""
    by_instance: dict[str, list[float]] = {}
    seed_pairs = 0
    for (instance, _seed), group in grouped.items():
        if baseline_solver not in group or solver not in group:
            continue
        candidate = group[solver]
        baseline = group[baseline_solver]
        if metric == "primal_integral":
            left, right = candidate.primal_integral, baseline.primal_integral
            if left is None or right is None:
                continue
            effect = float(left) - float(right)
        else:
            if not candidate.feasible or not baseline.feasible:
                continue
            if candidate.primal_error is not None and baseline.primal_error is not None:
                effect = float(candidate.primal_error) - float(baseline.primal_error)
            elif candidate.objective is not None and baseline.objective is not None:
                scale = max(1.0, abs(float(baseline.objective)))
                difference = float(candidate.objective) - float(baseline.objective)
                effect = (
                    difference / scale
                    if candidate.objective_sense == "minimize"
                    else -difference / scale
                )
            else:
                continue
        if not math.isfinite(effect):
            continue
        by_instance.setdefault(instance, []).append(effect)
        seed_pairs += 1
    return [float(median(values)) for _, values in sorted(by_instance.items())], seed_pairs


def _statistical_summary(
    grouped: dict[tuple[str, int], dict[str, BenchmarkResult]],
    *,
    solver: str,
    baseline_solver: str,
) -> dict[str, object]:
    """Build deterministic inference with instance—not seed—as the unit."""
    output: dict[str, object] = {
        "confidence_unit": "instance_after_seed_median",
        "bootstrap_samples": 2000,
    }
    for metric in ("primal_quality", "primal_integral"):
        effects, seed_pairs = _paired_instance_effects(
            grouped,
            solver=solver,
            baseline_solver=baseline_solver,
            metric=metric,
        )
        if not effects:
            output[metric] = {
                "eligible_instances": 0,
                "eligible_seed_pairs": 0,
                "median_candidate_minus_baseline": None,
                "confidence_interval": None,
                "sign_test_pvalue": None,
            }
            continue
        inference = paired_metric_summary(
            effects,
            [0.0] * len(effects),
            lower_is_better=True,
            bootstrap_samples=2000,
            seed=0,
        )
        output[metric] = {
            "eligible_instances": inference.pairs,
            "eligible_seed_pairs": seed_pairs,
            "wins": inference.wins,
            "ties": inference.ties,
            "losses": inference.losses,
            "median_candidate_minus_baseline": inference.median_difference,
            "confidence_interval": list(inference.confidence_interval),
            "sign_test_pvalue": inference.sign_test_pvalue,
        }
    return output


def _anytime_ecdf(rows: Sequence[BenchmarkResult]) -> dict[str, object]:
    """Return a fixed-grid ECDF of time to first verified feasible solution."""
    output: dict[str, object] = {}
    for solver in sorted({row.solver for row in rows}):
        selected = [row for row in rows if row.solver == solver]
        fractions = (0.01, 0.03, 0.1, 0.3, 1.0)
        points = []
        for fraction in fractions:
            reached = 0
            for row in selected:
                horizon = float(row.run_config.get("time_limit", row.runtime))
                cutoff = fraction * horizon
                reached += int(
                    row.time_to_first_feasible is not None
                    and float(row.time_to_first_feasible) <= cutoff
                )
            points.append(
                {
                    "normalised_time": fraction,
                    "verified_feasible_fraction": reached / len(selected) if selected else 0.0,
                }
            )
        output[solver] = {"runs": len(selected), "points": points}
    return output


def summarise_comparison(
    results: Sequence[BenchmarkResult],
    *,
    baseline_solver: str = "scip",
) -> dict[str, object]:
    """Create paired objective and anytime win/tie/loss counts by solver."""
    rows = tuple(results)
    solvers = sorted({row.solver for row in rows})
    grouped: dict[tuple[str, int], dict[str, BenchmarkResult]] = {}
    for row in rows:
        seed = int(row.run_config.get("seed", 0))
        grouped.setdefault((row.instance, seed), {})[row.solver] = row

    pairwise = {}
    for solver in solvers:
        if solver == baseline_solver:
            continue
        primary = [0, 0, 0]
        integral = [0, 0, 0]
        paired = 0
        qqa_intervention: dict[str, Any] = {
            "heuristic_invoked_pairs": 0,
            "qqa_executed_pairs": 0,
            "qqa_incumbent_improvement_pairs": 0,
            "executed": _empty_outcome_bucket(),
            "not_executed": _empty_outcome_bucket(),
        }
        for group in grouped.values():
            if baseline_solver not in group or solver not in group:
                continue
            paired += 1
            candidate = group[solver]
            baseline = group[baseline_solver]
            primary_outcome = _compare_primary(candidate, baseline)
            integral_outcome = _compare_lower(
                candidate.primal_integral,
                baseline.primal_integral,
            )
            primary[primary_outcome + 1] += 1
            integral[integral_outcome + 1] += 1
            if solver == "sg-cqqa":
                qqa = candidate.qqa or {}
                heuristic_invoked = int(qqa.get("calls", 0)) > 0
                qqa_executed = int(qqa.get("qqa_calls", 0)) > 0
                qqa_improved = int(qqa.get("qqa_incumbent_improvements", 0)) > 0
                qqa_intervention["heuristic_invoked_pairs"] += int(heuristic_invoked)
                qqa_intervention["qqa_executed_pairs"] += int(qqa_executed)
                qqa_intervention["qqa_incumbent_improvement_pairs"] += int(qqa_improved)
                bucket = qqa_intervention["executed" if qqa_executed else "not_executed"]
                _record_outcome(
                    bucket,
                    primary=primary_outcome,
                    integral=integral_outcome,
                )
        pairwise[solver] = {
            "baseline": baseline_solver,
            "paired_runs": paired,
            "primal_quality": {
                "losses": primary[0],
                "ties": primary[1],
                "wins": primary[2],
            },
            "primal_integral": {
                "losses": integral[0],
                "ties": integral[1],
                "wins": integral[2],
            },
        }
        if solver == "sg-cqqa":
            pairwise[solver]["qqa_intervention"] = qqa_intervention
        pairwise[solver]["inference"] = _statistical_summary(
            grouped,
            solver=solver,
            baseline_solver=baseline_solver,
        )
    return {
        "baseline_solver": baseline_solver,
        "by_solver": {
            solver: summarise_benchmarks([row for row in rows if row.solver == solver])["overall"]
            for solver in solvers
        },
        "pairwise": pairwise,
        "anytime_ecdf": _anytime_ecdf(rows),
    }


@dataclass(frozen=True, slots=True)
class BenchmarkComparisonResult:
    results: tuple[BenchmarkResult, ...]
    summary: dict[str, object]
    comparison_config: dict[str, Any]
    failures: tuple[BenchmarkFailure, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "results": [result.to_dict() for result in self.results],
            "summary": self.summary,
            "comparison_config": self.comparison_config,
            "failures": [failure.to_dict() for failure in self.failures],
        }


__all__ = [
    "BenchmarkResult",
    "BenchmarkFailure",
    "BenchmarkComparisonResult",
    "BenchmarkSuiteResult",
    "IncumbentPoint",
    "SCIPProgressTracker",
    "primal_integral",
    "normalised_primal_error",
    "relative_gap",
    "summarise_benchmarks",
    "summarise_comparison",
]
