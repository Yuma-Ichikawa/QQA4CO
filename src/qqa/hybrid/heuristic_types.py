"""Configuration and diagnostics for the opt-in SCIP-guided heuristic."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class QQAHeuristicConfig:
    core_size: int = 32
    maximum_problem_variables: int | None = 32
    maximum_integer_variables: int | None = 2048
    allowed_qplib_problem_types: tuple[str, ...] | None = None
    sol_size: int = 16
    epochs: int = 20
    max_calls: int = 1
    max_candidates: int = 1
    frequency: int = 10
    maximum_depth: int = 20
    minimum_core_size: int = 16
    maximum_core_saturation: float = 0.9
    completion_time: float = 0.25
    completion_nodes: int = 100
    dive_lp_iterations: int = 300
    dive_max_repair_changes: int = 12
    use_dive_completion: bool = True
    subscip_repair: bool = True
    qqa_fix_fraction: float = 0.25
    repair_beam_width: int = 16
    reference_pool_size: int = 3
    minimum_relative_improvement: float = 0.001
    minimum_call_time: float = 1.0
    minimum_qqa_time: float = 2.0
    maximum_call_time: float = 0.15
    maximum_call_time_fraction: float = 0.05
    fast_candidates: int = 0
    min_nodes_between_calls: int = 100
    local_branching_radius: int | None = None
    learning_rate: float = 0.05
    diversity: float = 0.05
    max_lp_rows: int = 64
    objective_weight: float = 1.0
    row_penalty: float = 20.0
    proximity_weight: float = 0.02
    reduced_cost_weight: float = 0.01
    require_surrogate_improvement: bool = True
    require_incumbent: bool = True
    adaptive_row_lagrangian: bool = True
    stop_qqa_after_nonimproving_call: bool = True
    maximum_overhead_fraction: float = 0.05
    threads: int = 1
    seed: int = 0
    device: str = "cpu"
    verbose: bool = False

    def __post_init__(self) -> None:
        for name in (
            "core_size",
            "sol_size",
            "max_calls",
            "max_candidates",
            "minimum_core_size",
            "completion_nodes",
            "dive_lp_iterations",
            "repair_beam_width",
            "reference_pool_size",
            "threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.maximum_problem_variables is not None and (
            isinstance(self.maximum_problem_variables, bool)
            or not isinstance(self.maximum_problem_variables, int)
            or self.maximum_problem_variables < 1
        ):
            raise ValueError("maximum_problem_variables must be a positive integer or None.")
        if self.maximum_integer_variables is not None and (
            isinstance(self.maximum_integer_variables, bool)
            or not isinstance(self.maximum_integer_variables, int)
            or self.maximum_integer_variables < 1
        ):
            raise ValueError("maximum_integer_variables must be a positive integer or None.")
        if self.allowed_qplib_problem_types is not None:
            if isinstance(self.allowed_qplib_problem_types, (str, bytes)):
                raise TypeError("allowed_qplib_problem_types must be a sequence of PROBTYPEs.")
            normalised_problem_types = tuple(
                dict.fromkeys(
                    str(problem_type).strip().upper()
                    for problem_type in self.allowed_qplib_problem_types
                )
            )
            if not normalised_problem_types or any(
                len(problem_type) != 3
                or problem_type[0] not in {"C", "L", "Q"}
                or problem_type[1] not in {"B", "C", "G", "I", "M"}
                or problem_type[2] not in {"B", "C", "D", "L", "N", "Q"}
                for problem_type in normalised_problem_types
            ):
                raise ValueError(
                    "allowed_qplib_problem_types must contain valid three-character PROBTYPEs."
                )
            object.__setattr__(
                self,
                "allowed_qplib_problem_types",
                normalised_problem_types,
            )
        if isinstance(self.frequency, bool) or not isinstance(self.frequency, int):
            raise ValueError("frequency must be an integer.")
        if self.frequency == 0 or self.frequency < -1:
            raise ValueError("frequency must be -1 or a positive integer.")
        if isinstance(self.maximum_depth, bool) or not isinstance(self.maximum_depth, int):
            raise ValueError("maximum_depth must be an integer.")
        if self.maximum_depth < -1:
            raise ValueError("maximum_depth must be >= -1.")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 0:
            raise ValueError("epochs must be a non-negative integer.")
        if (
            isinstance(self.fast_candidates, bool)
            or not isinstance(self.fast_candidates, int)
            or self.fast_candidates < 0
        ):
            raise ValueError("fast_candidates must be a non-negative integer.")
        if (
            isinstance(self.dive_max_repair_changes, bool)
            or not isinstance(self.dive_max_repair_changes, int)
            or self.dive_max_repair_changes < 0
        ):
            raise ValueError("dive_max_repair_changes must be a non-negative integer.")
        if (
            isinstance(self.max_lp_rows, bool)
            or not isinstance(self.max_lp_rows, int)
            or self.max_lp_rows < 0
        ):
            raise ValueError("max_lp_rows must be a non-negative integer.")
        if (
            isinstance(self.min_nodes_between_calls, bool)
            or not isinstance(self.min_nodes_between_calls, int)
            or self.min_nodes_between_calls < 0
        ):
            raise ValueError("min_nodes_between_calls must be a non-negative integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if self.local_branching_radius is not None and (
            isinstance(self.local_branching_radius, bool)
            or not isinstance(self.local_branching_radius, int)
            or self.local_branching_radius < 1
        ):
            raise ValueError("local_branching_radius must be a positive integer or None.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string.")
        if not isinstance(self.require_surrogate_improvement, bool):
            raise TypeError("require_surrogate_improvement must be a bool.")
        if not isinstance(self.require_incumbent, bool):
            raise TypeError("require_incumbent must be a bool.")
        if not isinstance(self.adaptive_row_lagrangian, bool):
            raise TypeError("adaptive_row_lagrangian must be a bool.")
        if not isinstance(self.stop_qqa_after_nonimproving_call, bool):
            raise TypeError("stop_qqa_after_nonimproving_call must be a bool.")
        if not isinstance(self.use_dive_completion, bool):
            raise TypeError("use_dive_completion must be a bool.")
        if not isinstance(self.subscip_repair, bool):
            raise TypeError("subscip_repair must be a bool.")
        if not self.use_dive_completion and not self.subscip_repair:
            raise ValueError("At least one completion strategy must be enabled.")
        if (
            not math.isfinite(self.maximum_overhead_fraction)
            or not 0 < self.maximum_overhead_fraction <= 1
        ):
            raise ValueError("maximum_overhead_fraction must be in (0, 1].")
        if (
            not math.isfinite(self.maximum_call_time_fraction)
            or not 0 < self.maximum_call_time_fraction <= 1
        ):
            raise ValueError("maximum_call_time_fraction must be in (0, 1].")
        if (
            not math.isfinite(self.maximum_core_saturation)
            or not 0 < self.maximum_core_saturation <= 1
        ):
            raise ValueError("maximum_core_saturation must be in (0, 1].")
        if self.minimum_core_size > self.core_size:
            raise ValueError("minimum_core_size must not exceed core_size.")
        if self.minimum_core_size / self.core_size > self.maximum_core_saturation:
            raise ValueError("minimum_core_size/core_size must not exceed maximum_core_saturation.")
        for name in (
            "completion_time",
            "minimum_call_time",
            "minimum_qqa_time",
            "maximum_call_time",
            "learning_rate",
            "objective_weight",
            "row_penalty",
            "proximity_weight",
            "reduced_cost_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0.")
        if (
            self.completion_time == 0
            or self.minimum_call_time == 0
            or self.minimum_qqa_time == 0
            or self.learning_rate == 0
        ):
            raise ValueError(
                "completion_time, minimum_call_time, minimum_qqa_time, and learning_rate "
                "must be > 0."
            )
        if not math.isfinite(self.diversity) or not 0 <= self.diversity <= 1:
            raise ValueError("diversity must be in [0, 1].")
        if not math.isfinite(self.qqa_fix_fraction) or not 0 <= self.qqa_fix_fraction <= 1:
            raise ValueError("qqa_fix_fraction must be in [0, 1].")
        if (
            not math.isfinite(self.minimum_relative_improvement)
            or not 0 <= self.minimum_relative_improvement < 1
        ):
            raise ValueError("minimum_relative_improvement must be in [0, 1).")


@dataclass(slots=True)
class QQAHeuristicStats:
    callbacks: int = 0
    node_infeasible_skips: int = 0
    call_limit_skips: int = 0
    lp_status_skips: int = 0
    incumbent_skips: int = 0
    node_spacing_skips: int = 0
    remaining_time_skips: int = 0
    late_call_skips: int = 0
    prechecks: int = 0
    small_core_skips: int = 0
    saturation_skips: int = 0
    inspection_runtime: float = 0.0
    lp_candidate_counts: list[int] = field(default_factory=list)
    calls: int = 0
    qqa_calls: int = 0
    qqa_runtime: float = 0.0
    qqa_al_updates: int = 0
    qqa_archive_observations: int = 0
    completion_runtime: float = 0.0
    dive_attempts: int = 0
    dive_feasible: int = 0
    dive_accepted: int = 0
    dive_incumbent_improvements: int = 0
    candidates: int = 0
    completion_feasible: int = 0
    accepted: int = 0
    incumbent_improvements: int = 0
    failures: int = 0
    fast_candidates: int = 0
    fast_completion_candidates: int = 0
    qqa_candidates: int = 0
    qqa_completion_feasible: int = 0
    qqa_accepted: int = 0
    qqa_incumbent_improvements: int = 0
    lns_repair_attempts: int = 0
    lns_repair_feasible: int = 0
    lns_repair_accepted: int = 0
    lns_repair_incumbent_improvements: int = 0
    partial_lns_attempts: int = 0
    partial_lns_feasible: int = 0
    partial_lns_accepted: int = 0
    partial_lns_incumbent_improvements: int = 0
    qqa_stopped_after_nonimprovement: bool = False
    filtered_candidates: int = 0
    core_sizes: list[int] = field(default_factory=list)
    surrogate_rows: list[int] = field(default_factory=list)
    objective_sources: dict[str, int] = field(default_factory=dict)
    failure_types: dict[str, int] = field(default_factory=dict)
    completion_statuses: dict[str, int] = field(default_factory=dict)
    call_nodes: list[int] = field(default_factory=list)
    call_times: list[float] = field(default_factory=list)
    completion_objectives: list[float | None] = field(default_factory=list)
    completion_incumbents_before: list[float | None] = field(default_factory=list)
    completion_relative_improvements: list[float | None] = field(default_factory=list)
    reference_pool_sizes: list[int] = field(default_factory=list)
    multi_reference_calls: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "callbacks": self.callbacks,
            "node_infeasible_skips": self.node_infeasible_skips,
            "call_limit_skips": self.call_limit_skips,
            "lp_status_skips": self.lp_status_skips,
            "incumbent_skips": self.incumbent_skips,
            "node_spacing_skips": self.node_spacing_skips,
            "remaining_time_skips": self.remaining_time_skips,
            "late_call_skips": self.late_call_skips,
            "prechecks": self.prechecks,
            "small_core_skips": self.small_core_skips,
            "saturation_skips": self.saturation_skips,
            "inspection_runtime": self.inspection_runtime,
            "lp_candidate_counts": list(self.lp_candidate_counts),
            "calls": self.calls,
            "qqa_calls": self.qqa_calls,
            "qqa_runtime": self.qqa_runtime,
            "qqa_al_updates": self.qqa_al_updates,
            "qqa_archive_observations": self.qqa_archive_observations,
            "completion_runtime": self.completion_runtime,
            "dive_attempts": self.dive_attempts,
            "dive_feasible": self.dive_feasible,
            "dive_accepted": self.dive_accepted,
            "dive_incumbent_improvements": self.dive_incumbent_improvements,
            "candidates": self.candidates,
            "completion_feasible": self.completion_feasible,
            "accepted": self.accepted,
            "incumbent_improvements": self.incumbent_improvements,
            "failures": self.failures,
            "fast_candidates": self.fast_candidates,
            "fast_completion_candidates": self.fast_completion_candidates,
            "qqa_candidates": self.qqa_candidates,
            "qqa_completion_feasible": self.qqa_completion_feasible,
            "qqa_accepted": self.qqa_accepted,
            "qqa_incumbent_improvements": self.qqa_incumbent_improvements,
            "lns_repair_attempts": self.lns_repair_attempts,
            "lns_repair_feasible": self.lns_repair_feasible,
            "lns_repair_accepted": self.lns_repair_accepted,
            "lns_repair_incumbent_improvements": self.lns_repair_incumbent_improvements,
            "partial_lns_attempts": self.partial_lns_attempts,
            "partial_lns_feasible": self.partial_lns_feasible,
            "partial_lns_accepted": self.partial_lns_accepted,
            "partial_lns_incumbent_improvements": (self.partial_lns_incumbent_improvements),
            "qqa_stopped_after_nonimprovement": self.qqa_stopped_after_nonimprovement,
            "filtered_candidates": self.filtered_candidates,
            "core_sizes": list(self.core_sizes),
            "surrogate_rows": list(self.surrogate_rows),
            "objective_sources": dict(self.objective_sources),
            "failure_types": dict(self.failure_types),
            "completion_statuses": dict(self.completion_statuses),
            "call_nodes": list(self.call_nodes),
            "call_times": list(self.call_times),
            "completion_objectives": list(self.completion_objectives),
            "completion_incumbents_before": list(self.completion_incumbents_before),
            "completion_relative_improvements": list(self.completion_relative_improvements),
            "reference_pool_sizes": list(self.reference_pool_sizes),
            "multi_reference_calls": self.multi_reference_calls,
            "completion_rate": (
                self.completion_feasible / self.candidates if self.candidates else 0.0
            ),
            "acceptance_rate": self.accepted / self.candidates if self.candidates else 0.0,
            "incumbent_improvement_rate": (
                self.incumbent_improvements / self.candidates if self.candidates else 0.0
            ),
            "qqa_completion_rate": (
                self.qqa_completion_feasible / self.qqa_candidates if self.qqa_candidates else 0.0
            ),
            "qqa_acceptance_rate": (
                self.qqa_accepted / self.qqa_candidates if self.qqa_candidates else 0.0
            ),
            "qqa_incumbent_improvement_rate": (
                self.qqa_incumbent_improvements / self.qqa_candidates
                if self.qqa_candidates
                else 0.0
            ),
            "lns_repair_feasibility_rate": (
                self.lns_repair_feasible / self.lns_repair_attempts
                if self.lns_repair_attempts
                else 0.0
            ),
            "lns_repair_acceptance_rate": (
                self.lns_repair_accepted / self.lns_repair_attempts
                if self.lns_repair_attempts
                else 0.0
            ),
            "partial_lns_feasibility_rate": (
                self.partial_lns_feasible / self.partial_lns_attempts
                if self.partial_lns_attempts
                else 0.0
            ),
            "partial_lns_acceptance_rate": (
                self.partial_lns_accepted / self.partial_lns_attempts
                if self.partial_lns_attempts
                else 0.0
            ),
        }


__all__ = ["QQAHeuristicConfig", "QQAHeuristicStats"]
