"""SCIP-guided conditional QQA primal heuristic.

QQA explores only a small uncertain integer core.  Every proposed integer
assignment is completed in a sub-SCIP, so continuous variables, feasibility,
dual bounds, and proof responsibility remain with SCIP.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from time import perf_counter

import numpy as np
import torch

from qqa.algebraic import AlgebraicModel
from qqa.decomposition import complete_integer_assignment, create_completion_template
from qqa.hybrid.core_selector import select_uncertain_integer_core
from qqa.hybrid.neighborhood import build_neighborhood, within_local_branching
from qqa.hybrid.surrogate import (
    CoreSurrogate,
    build_core_surrogate,
    generate_surrogate_candidates,
)
from qqa.mixed import Binary, Constraint, Integer, MixedProblem
from qqa.presolve import extract_scip_state

try:  # Keep importing qqa independent from the optional SCIP wheel.
    from pyscipopt import Heur as _HeurBase
except (ImportError, OSError):  # pragma: no cover - optional dependency

    class _HeurBase:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True, slots=True)
class QQAHeuristicConfig:
    core_size: int = 64
    sol_size: int = 32
    epochs: int = 120
    max_calls: int = 4
    max_candidates: int = 8
    completion_time: float = 1.0
    completion_nodes: int = 500
    minimum_call_time: float = 3.0
    minimum_qqa_time: float = 20.0
    fast_candidates: int = 2
    min_nodes_between_calls: int = 10
    local_branching_radius: int | None = None
    learning_rate: float = 0.05
    diversity: float = 0.05
    max_lp_rows: int = 128
    objective_weight: float = 1.0
    row_penalty: float = 20.0
    proximity_weight: float = 0.02
    reduced_cost_weight: float = 0.01
    require_surrogate_improvement: bool = True
    stop_qqa_after_nonimproving_call: bool = True
    maximum_overhead_fraction: float = 0.1
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
            "completion_nodes",
            "threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 0:
            raise ValueError("epochs must be a non-negative integer.")
        if (
            isinstance(self.fast_candidates, bool)
            or not isinstance(self.fast_candidates, int)
            or self.fast_candidates < 0
        ):
            raise ValueError("fast_candidates must be a non-negative integer.")
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
        if not isinstance(self.stop_qqa_after_nonimproving_call, bool):
            raise TypeError("stop_qqa_after_nonimproving_call must be a bool.")
        if (
            not math.isfinite(self.maximum_overhead_fraction)
            or not 0 < self.maximum_overhead_fraction <= 1
        ):
            raise ValueError("maximum_overhead_fraction must be in (0, 1].")
        for name in (
            "completion_time",
            "minimum_call_time",
            "minimum_qqa_time",
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


@dataclass(slots=True)
class QQAHeuristicStats:
    calls: int = 0
    qqa_calls: int = 0
    qqa_runtime: float = 0.0
    qqa_al_updates: int = 0
    qqa_archive_observations: int = 0
    completion_runtime: float = 0.0
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
    qqa_stopped_after_nonimprovement: bool = False
    filtered_candidates: int = 0
    core_sizes: list[int] = field(default_factory=list)
    surrogate_rows: list[int] = field(default_factory=list)
    objective_sources: dict[str, int] = field(default_factory=dict)
    failure_types: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "qqa_calls": self.qqa_calls,
            "qqa_runtime": self.qqa_runtime,
            "qqa_al_updates": self.qqa_al_updates,
            "qqa_archive_observations": self.qqa_archive_observations,
            "completion_runtime": self.completion_runtime,
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
            "qqa_stopped_after_nonimprovement": self.qqa_stopped_after_nonimprovement,
            "filtered_candidates": self.filtered_candidates,
            "core_sizes": list(self.core_sizes),
            "surrogate_rows": list(self.surrogate_rows),
            "objective_sources": dict(self.objective_sources),
            "failure_types": dict(self.failure_types),
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


@contextmanager
def _torch_thread_budget(threads: int):
    previous = torch.get_num_threads()
    if previous == threads:
        yield
        return
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@contextmanager
def _torch_seed(seed: int, device: str):
    target = torch.device(device)
    devices: list[int] = []
    if target.type == "cuda" and torch.cuda.is_available():
        devices = [target.index if target.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def _core_problem(
    state,
    selection,
    positions: list[int],
    surrogate: CoreSurrogate,
    config: QQAHeuristicConfig,
) -> tuple[MixedProblem, list[str]]:
    declarations = []
    names = []
    targets = []
    spans = []
    reduced = []
    weights = []
    for output_index, position in enumerate(positions):
        variable_index = int(selection.core_indices[position])
        name = f"z_{output_index}"
        lower = int(math.ceil(selection.local_lower[position] - 1e-9))
        upper = int(math.floor(selection.local_upper[position] + 1e-9))
        if state.variable_types[variable_index] == "BINARY":
            declarations.append(Binary(name))
        else:
            declarations.append(Integer(name, lower=lower, upper=upper))
        names.append(name)
        targets.append(float(state.lp_values[variable_index]))
        spans.append(float(max(1, upper - lower)))
        reduced.append(float(state.reduced_costs[variable_index]))
        weights.append(float(max(0.1, selection.scores[position])))

    target_vector = torch.tensor(targets, dtype=torch.float64)
    span_vector = torch.tensor(spans, dtype=torch.float64)
    reduced_vector = torch.tensor(reduced, dtype=torch.float64)
    reduced_vector /= reduced_vector.abs().amax().clamp_min(1.0)
    weight_vector = torch.tensor(weights, dtype=torch.float64)
    quadratic = torch.tensor(surrogate.quadratic, dtype=torch.float64)
    linear = torch.tensor(surrogate.linear, dtype=torch.float64)
    row_matrix = torch.tensor(surrogate.row_matrix, dtype=torch.float64)
    row_offset = torch.tensor(surrogate.row_offset, dtype=torch.float64)
    row_lower = torch.tensor(surrogate.row_lower, dtype=torch.float64)
    row_upper = torch.tensor(surrogate.row_upper, dtype=torch.float64)
    row_scale = torch.tensor(surrogate.row_scale, dtype=torch.float64)

    def objective(values):
        stacked = torch.stack([values[name] for name in names], dim=1)
        target = target_vector.to(stacked)
        span = span_vector.to(stacked)
        redcost = reduced_vector.to(stacked)
        weight = weight_vector.to(stacked)
        local_quadratic = quadratic.to(stacked)
        local_linear = linear.to(stacked)
        original = (
            0.5 * torch.einsum("bi,ij,bj->b", stacked, local_quadratic, stacked)
            + stacked @ local_linear
        )
        original = config.objective_weight * original / surrogate.objective_scale
        proximity = config.proximity_weight * (weight * ((stacked - target) / span).square()).mean(
            dim=1
        )
        direction = config.reduced_cost_weight * (redcost * (stacked - target) / span).mean(dim=1)
        return original + proximity + direction

    constraints = []
    row_weight = config.row_penalty / max(1, surrogate.num_rows)

    def row_function(coefficients: torch.Tensor, offset: float):
        def evaluate(values):
            stacked = torch.stack([values[name] for name in names], dim=1)
            return stacked @ coefficients.to(stacked) + offset

        return evaluate

    for row_index in range(surrogate.num_rows):
        coefficients = row_matrix[row_index].clone()
        offset = float(row_offset[row_index])
        lower = float(row_lower[row_index])
        upper = float(row_upper[row_index])
        scale = float(row_scale[row_index])
        function = row_function(coefficients, offset)
        tolerance = 1e-7 * scale
        if math.isfinite(lower) and math.isfinite(upper) and abs(upper - lower) <= 1e-12 * scale:
            constraints.append(
                Constraint(
                    function,
                    sense="==",
                    rhs=lower,
                    weight=row_weight,
                    scale=scale,
                    tolerance=tolerance,
                    name=f"surrogate_row_{row_index}",
                )
            )
            continue
        if math.isfinite(lower):
            constraints.append(
                Constraint(
                    function,
                    sense=">=",
                    rhs=lower,
                    weight=row_weight,
                    scale=scale,
                    tolerance=tolerance,
                    name=f"surrogate_row_{row_index}_lower",
                )
            )
        if math.isfinite(upper):
            constraints.append(
                Constraint(
                    function,
                    sense="<=",
                    rhs=upper,
                    weight=row_weight,
                    scale=scale,
                    tolerance=tolerance,
                    name=f"surrogate_row_{row_index}_upper",
                )
            )

    return MixedProblem(
        declarations,
        objective,
        constraints=constraints,
        name="scip-guided-integer-core",
        dtype=torch.float64,
    ), names


class QQAHeuristic(_HeurBase):
    """PySCIPOpt ``Heur`` implementing root/node conditional QQA-LNS."""

    def __init__(
        self,
        config: QQAHeuristicConfig | None = None,
        *,
        completion_template=None,
        algebraic: AlgebraicModel | None = None,
        incumbent_provider: Callable[[], np.ndarray | None] | None = None,
    ):
        self.config = config or QQAHeuristicConfig()
        self.completion_template = completion_template
        self.algebraic = algebraic
        self.incumbent_provider = incumbent_provider
        self.stats = QQAHeuristicStats()
        self._last_node = -(10**18)
        self._archive: set[bytes] = set()

    def _with_external_incumbent(self, state):
        if self.algebraic is None or self.incumbent_provider is None:
            return state
        external = self.incumbent_provider()
        if external is None:
            return state
        values = np.asarray(external, dtype=np.float64)
        if values.shape != (self.algebraic.num_variables,):
            return state
        by_name = {name: index for index, name in enumerate(self.algebraic.variable_names)}
        incumbent = (
            state.incumbent_values.copy()
            if state.incumbent_values is not None
            else state.lp_values.copy()
        )
        mapped = False
        for index, name in enumerate(state.names):
            original = by_name.get(name)
            if original is None:
                continue
            incumbent[index] = min(
                max(values[original], state.local_lower[index]),
                state.local_upper[index],
            )
            mapped = True
        return replace(state, incumbent_values=incumbent) if mapped else state

    def _external_incumbent_objective(self) -> float | None:
        """Evaluate the benchmark-visible incumbent before solution injection."""
        if self.algebraic is None or self.incumbent_provider is None:
            return None
        external = self.incumbent_provider()
        if external is None:
            return None
        values = np.asarray(external, dtype=np.float64)
        if values.shape != (self.algebraic.num_variables,):
            return None
        evaluation = self.algebraic.evaluate(values)
        if evaluation.maximum_infeasibility > 1e-6:
            return None
        return float(evaluation.objective)

    def _is_original_improvement(
        self,
        candidate: float | None,
        incumbent: float | None,
    ) -> bool:
        if candidate is None or incumbent is None or self.algebraic is None:
            return False
        tolerance = 1e-9 * max(1.0, abs(candidate), abs(incumbent))
        if self.algebraic.objective_sense == "maximize":
            return candidate > incumbent + tolerance
        return candidate < incumbent - tolerance

    def _remaining_time(self) -> float:
        try:
            limit = float(self.model.getParam("limits/time"))
        except Exception:
            limit = math.inf
        if not math.isfinite(limit) or limit >= 1e19:
            return self.config.completion_time * self.config.max_candidates + 1.0
        return max(0.0, limit - float(self.model.getSolvingTime()))

    def _remaining_overhead_budget(self) -> float:
        try:
            limit = float(self.model.getParam("limits/time"))
        except Exception:
            return math.inf
        if not math.isfinite(limit) or limit >= 1e19:
            return math.inf
        used = self.stats.qqa_runtime + self.stats.completion_runtime
        return max(0.0, self.config.maximum_overhead_fraction * limit - used)

    def _qqa_has_stalled(self) -> bool:
        return bool(
            self.config.stop_qqa_after_nonimproving_call
            and self.stats.qqa_calls > 0
            and self.stats.qqa_incumbent_improvements == 0
        )

    def _fast_path_supports_qqa(self, completion_feasible_before: int) -> bool:
        """Require completion evidence before escalating a primary hybrid call."""
        return bool(
            self.config.fast_candidates == 0
            or self.stats.completion_feasible > completion_feasible_before
        )

    def _rank_population(
        self,
        problem: MixedProblem,
        population: list[np.ndarray],
        state,
        selection,
        positions: list[int],
        selected_indices: np.ndarray,
    ) -> list[np.ndarray]:
        if not population:
            return []
        population_array = np.stack(population)
        with torch.no_grad():
            losses = (
                problem.loss_fn(torch.as_tensor(population_array, dtype=torch.float64))
                .detach()
                .cpu()
                .numpy()
            )
            incumbent_loss = None
            if self.config.require_surrogate_improvement and state.incumbent_values is not None:
                incumbent_core = np.minimum(
                    np.maximum(
                        np.rint(state.incumbent_values[selected_indices]),
                        selection.local_lower[positions],
                    ),
                    selection.local_upper[positions],
                )
                incumbent_loss = float(
                    problem.loss_fn(torch.as_tensor(incumbent_core, dtype=torch.float64))[0]
                )
        order = np.argsort(losses, kind="stable")
        if incumbent_loss is not None:
            retained = [index for index in order if float(losses[index]) < incumbent_loss - 1e-8]
            self.stats.filtered_candidates += len(order) - len(retained)
            order = np.asarray(retained, dtype=np.int64)
        return [population_array[index] for index in order]

    def _complete_population(
        self,
        population: list[np.ndarray],
        problem: MixedProblem,
        state,
        selection,
        positions: list[int],
        selected_indices: np.ndarray,
        neighborhood,
        seen_now: set[bytes],
        *,
        source: str,
    ) -> tuple[bool, bool]:
        if source not in {"fast", "qqa"}:
            raise ValueError("source must be 'fast' or 'qqa'.")
        accepted_any = False
        improved_any = False
        ranked = self._rank_population(
            problem,
            population,
            state,
            selection,
            positions,
            selected_indices,
        )
        for core_values_reduced in ranked:
            core_values = np.rint(state.lp_values[selection.core_indices]).astype(np.float64)
            core_values[positions] = np.rint(core_values_reduced)
            core_values = np.minimum(
                np.maximum(core_values, selection.local_lower),
                selection.local_upper,
            )
            if not within_local_branching(neighborhood, core_values):
                continue
            indices, assignments = neighborhood.complete_assignment(core_values, state)
            signature = np.stack([indices, assignments.astype(np.int64)]).tobytes()
            if signature in self._archive or signature in seen_now:
                continue
            seen_now.add(signature)
            self._archive.add(signature)
            self.stats.candidates += 1
            if source == "fast":
                self.stats.fast_completion_candidates += 1
            else:
                self.stats.qqa_candidates += 1
            remaining = self._remaining_time()
            overhead_remaining = self._remaining_overhead_budget()
            if remaining <= 0.05 or overhead_remaining <= 0.05:
                break
            external_before = self._external_incumbent_objective()
            completion_budget = min(
                self.config.completion_time,
                max(0.05, remaining * 0.5),
                overhead_remaining,
            )
            can_release_complement = bool(neighborhood.fixed_indices.size)
            can_release_core = len(neighborhood.core_indices) > 1
            first_budget = (
                max(0.05, (0.25 if can_release_core else 0.5) * completion_budget)
                if can_release_complement and completion_budget >= 0.1
                else completion_budget
            )
            completion_started = perf_counter()
            completed = complete_integer_assignment(
                self.completion_template,
                [state.names[index] for index in indices],
                assignments,
                main_model=self.model,
                heuristic=self,
                algebraic=self.algebraic,
                time_limit=first_budget,
                node_limit=(
                    max(1, self.config.completion_nodes // 2)
                    if can_release_complement
                    else self.config.completion_nodes
                ),
                seed=self.config.seed + self.stats.calls - 1,
                verbose=self.config.verbose,
            )
            first_runtime = perf_counter() - completion_started
            self.stats.completion_runtime += first_runtime
            repaired = False
            if not completed.feasible and can_release_complement:
                remaining = self._remaining_time()
                overhead_remaining = self._remaining_overhead_budget()
                repair_budget = min(
                    max(
                        0.0,
                        (completion_budget - first_runtime) * (0.5 if can_release_core else 1.0),
                    ),
                    max(0.0, remaining * 0.5),
                    overhead_remaining,
                )
                if repair_budget > 0.05:
                    self.stats.lns_repair_attempts += 1
                    repair_started = perf_counter()
                    completed = complete_integer_assignment(
                        self.completion_template,
                        [state.names[index] for index in neighborhood.core_indices],
                        core_values,
                        main_model=self.model,
                        heuristic=self,
                        algebraic=self.algebraic,
                        time_limit=repair_budget,
                        node_limit=max(1, self.config.completion_nodes // 2),
                        seed=self.config.seed + self.stats.calls - 1,
                        verbose=self.config.verbose,
                    )
                    self.stats.completion_runtime += perf_counter() - repair_started
                    repaired = True
            if not completed.feasible and repaired and can_release_core:
                remaining = self._remaining_time()
                overhead_remaining = self._remaining_overhead_budget()
                spent = perf_counter() - completion_started
                partial_budget = min(
                    max(0.0, completion_budget - spent),
                    max(0.0, remaining * 0.5),
                    overhead_remaining,
                )
                if partial_budget > 0.05:
                    keep = max(1, math.ceil(0.25 * len(neighborhood.core_indices)))
                    selected = np.argsort(selection.scores, kind="stable")[-keep:]
                    self.stats.partial_lns_attempts += 1
                    partial_started = perf_counter()
                    completed = complete_integer_assignment(
                        self.completion_template,
                        [state.names[index] for index in neighborhood.core_indices[selected]],
                        core_values[selected],
                        main_model=self.model,
                        heuristic=self,
                        algebraic=self.algebraic,
                        time_limit=partial_budget,
                        node_limit=max(1, self.config.completion_nodes // 3),
                        seed=self.config.seed + self.stats.calls - 1,
                        verbose=self.config.verbose,
                    )
                    self.stats.completion_runtime += perf_counter() - partial_started
                    self.stats.partial_lns_feasible += int(completed.feasible)
                    self.stats.partial_lns_accepted += int(completed.accepted)
            self.stats.completion_feasible += int(completed.feasible)
            self.stats.accepted += int(completed.accepted)
            original_improvement = (
                self._is_original_improvement(completed.objective, external_before)
                if external_before is not None
                else completed.improved_incumbent
            )
            self.stats.incumbent_improvements += int(original_improvement)
            if repaired:
                self.stats.lns_repair_feasible += int(completed.feasible)
                self.stats.lns_repair_accepted += int(completed.accepted)
                self.stats.lns_repair_incumbent_improvements += int(original_improvement)
            if source == "qqa":
                self.stats.qqa_completion_feasible += int(completed.feasible)
                self.stats.qqa_accepted += int(completed.accepted)
                self.stats.qqa_incumbent_improvements += int(original_improvement)
            accepted_any = accepted_any or completed.accepted
            improved_any = improved_any or original_improvement
            if source == "fast" and original_improvement:
                break
            if self.stats.candidates >= self.config.max_candidates * self.stats.calls:
                break
        return accepted_any, improved_any

    def heurexec(self, heurtiming, nodeinfeasible):  # noqa: ARG002 - SCIP callback signature
        from pyscipopt import SCIP_LPSOLSTAT, SCIP_RESULT

        if nodeinfeasible or self.stats.calls >= self.config.max_calls:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        if self.model.getLPSolstat() != SCIP_LPSOLSTAT.OPTIMAL:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        node_number = int(self.model.getNNodes())
        if node_number - self._last_node < self.config.min_nodes_between_calls:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        if self._remaining_time() <= max(
            self.config.minimum_call_time,
            self.config.completion_time,
        ):
            return {"result": SCIP_RESULT.DIDNOTRUN}

        self._last_node = node_number
        self.stats.calls += 1
        try:
            state = self._with_external_incumbent(extract_scip_state(self.model))
            selection = select_uncertain_integer_core(
                state,
                max_core_size=self.config.core_size,
            )
            positions = [
                position
                for position, (lower, upper) in enumerate(
                    zip(selection.local_lower, selection.local_upper, strict=True)
                )
                if math.ceil(lower - 1e-9) < math.floor(upper + 1e-9)
            ]
            if not positions:
                return {"result": SCIP_RESULT.DIDNOTFIND}
            selected_indices = selection.core_indices[positions]
            self.stats.core_sizes.append(len(selected_indices))
            surrogate = build_core_surrogate(
                self.model,
                state,
                selection,
                positions,
                algebraic=self.algebraic,
                max_lp_rows=self.config.max_lp_rows,
            )
            self.stats.surrogate_rows.append(surrogate.num_rows)
            self.stats.objective_sources[surrogate.objective_source] = (
                self.stats.objective_sources.get(surrogate.objective_source, 0) + 1
            )
            problem, _ = _core_problem(state, selection, positions, surrogate, self.config)
            initial_values = np.asarray(
                [state.lp_values[index] for index in selected_indices], dtype=np.float64
            )
            fast_population = list(
                generate_surrogate_candidates(
                    surrogate,
                    target=initial_values,
                    lower=selection.local_lower[positions],
                    upper=selection.local_upper[positions],
                    max_candidates=min(self.config.fast_candidates, self.config.max_candidates),
                    row_penalty=self.config.row_penalty,
                    proximity_weight=self.config.proximity_weight,
                    seed=self.config.seed + self.stats.calls - 1,
                )
            )
            self.stats.fast_candidates += len(fast_population)
            neighborhood = build_neighborhood(
                selection,
                state,
                local_branching_radius=self.config.local_branching_radius,
            )
            seen_now: set[bytes] = set()
            completion_feasible_before = self.stats.completion_feasible
            accepted_any, improved_fast = self._complete_population(
                fast_population,
                problem,
                state,
                selection,
                positions,
                selected_indices,
                neighborhood,
                seen_now,
                source="fast",
            )
            remaining = self._remaining_time()
            qqa_stalled = self._qqa_has_stalled()
            fast_path_supports_qqa = self._fast_path_supports_qqa(completion_feasible_before)
            if (
                not improved_fast
                and fast_path_supports_qqa
                and not qqa_stalled
                and remaining > self.config.minimum_qqa_time
                and self._remaining_overhead_budget() > 0.05
                and self.config.epochs > 0
            ):
                overhead_remaining = self._remaining_overhead_budget()
                completion_reserve = min(
                    self.config.completion_time,
                    0.5 * remaining,
                    max(0.05, 0.25 * overhead_remaining),
                )
                qqa_budget = min(
                    remaining - completion_reserve,
                    overhead_remaining - completion_reserve,
                )
                if qqa_budget <= 0.05:
                    return {
                        "result": (SCIP_RESULT.FOUNDSOL if accepted_any else SCIP_RESULT.DIDNOTFIND)
                    }
                local_lower = selection.local_lower[positions]
                local_upper = selection.local_upper[positions]
                initial_population = np.empty(
                    (self.config.sol_size, len(initial_values)), dtype=np.float64
                )
                initial_population[0] = initial_values
                next_row = 1
                if state.incumbent_values is not None and next_row < self.config.sol_size:
                    initial_population[next_row] = np.minimum(
                        np.maximum(
                            state.incumbent_values[selected_indices],
                            local_lower,
                        ),
                        local_upper,
                    )
                    next_row += 1
                for candidate in fast_population:
                    if next_row >= self.config.sol_size:
                        break
                    initial_population[next_row] = candidate
                    next_row += 1
                rng = np.random.default_rng(self.config.seed + self.stats.calls - 1)
                probability = np.clip(
                    (initial_values - local_lower) / np.maximum(1.0, local_upper - local_lower),
                    0.0,
                    1.0,
                )
                while next_row < self.config.sol_size:
                    local_probability = probability if next_row % 2 == 0 else 0.5
                    initial_population[next_row] = np.where(
                        rng.random(len(initial_values)) < local_probability,
                        local_upper,
                        local_lower,
                    )
                    next_row += 1
                initial = torch.as_tensor(initial_population, dtype=torch.float64)
                qqa_started = perf_counter()
                with (
                    _torch_thread_budget(self.config.threads),
                    _torch_seed(
                        self.config.seed + self.stats.calls - 1,
                        self.config.device,
                    ),
                ):
                    result = problem.solve(
                        sol_size=self.config.sol_size,
                        num_epochs=self.config.epochs,
                        learning_rate=self.config.learning_rate,
                        div_param=self.config.diversity,
                        initial_state=initial,
                        return_population=True,
                        calibrate_penalty=False,
                        adaptive_augmented_lagrangian=bool(problem.constraints),
                        al_update_interval=max(1, min(25, self.config.epochs // 4)),
                        repair=False,
                        polish=False,
                        restart_patience=max(20, self.config.epochs // 2),
                        time_limit=qqa_budget,
                        verbose=self.config.verbose,
                    )
                self.stats.qqa_calls += 1
                self.stats.qqa_runtime += perf_counter() - qqa_started
                al_diagnostics = result.diagnostics.get("adaptive_augmented_lagrangian")
                if isinstance(al_diagnostics, Mapping):
                    self.stats.qqa_al_updates += int(al_diagnostics.get("updates", 0))
                archive_diagnostics = result.diagnostics.get("constraint_archive")
                if isinstance(archive_diagnostics, Mapping):
                    self.stats.qqa_archive_observations += int(
                        archive_diagnostics.get("observations", 0)
                    )
                qqa_population = [result.best_sol.detach().cpu().numpy()]
                if result.final_population is not None:
                    qqa_population.extend(result.final_population.detach().cpu().numpy())
                accepted_qqa, _ = self._complete_population(
                    qqa_population,
                    problem,
                    state,
                    selection,
                    positions,
                    selected_indices,
                    neighborhood,
                    seen_now,
                    source="qqa",
                )
                accepted_any = accepted_any or accepted_qqa
            elif qqa_stalled:
                self.stats.qqa_stopped_after_nonimprovement = True
            return {"result": SCIP_RESULT.FOUNDSOL if accepted_any else SCIP_RESULT.DIDNOTFIND}
        except Exception as exc:
            self.stats.failures += 1
            failure = type(exc).__name__
            self.stats.failure_types[failure] = self.stats.failure_types.get(failure, 0) + 1
            return {"result": SCIP_RESULT.DIDNOTFIND}


def include_qqa_heuristic(
    model,
    config: QQAHeuristicConfig | None = None,
    *,
    algebraic: AlgebraicModel | None = None,
    incumbent_provider: Callable[[], np.ndarray | None] | None = None,
) -> QQAHeuristic:
    """Include conditional QQA at useful LP-node timings and return the plugin."""
    try:
        from pyscipopt import SCIP_HEURTIMING
    except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
        raise ImportError("SCIP-guided QQA requires `qqa[scip]`.") from exc
    completion_template = create_completion_template(model)
    heuristic = QQAHeuristic(
        config,
        completion_template=completion_template,
        algebraic=algebraic,
        incumbent_provider=incumbent_provider,
    )
    timing = SCIP_HEURTIMING.AFTERLPNODE | SCIP_HEURTIMING.AFTERLPPLUNGE
    model.includeHeur(
        heuristic,
        "qqa_conditional",
        "SCIP-guided conditional QQA with continuous completion",
        "Q",
        priority=5000,
        freq=1,
        maxdepth=20,
        timingmask=timing,
        usessubscip=True,
    )
    return heuristic


__all__ = [
    "QQAHeuristic",
    "QQAHeuristicConfig",
    "QQAHeuristicStats",
    "include_qqa_heuristic",
]
