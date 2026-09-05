"""SCIP-guided conditional QQA primal heuristic.

QQA explores only a small uncertain integer core.  Proposed assignments are
completed by an in-place SCIP dive and, when enabled, a tightly bounded LNS
repair.  SCIP remains the authority for feasibility, bounds, and proof.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING, Any

from qqa.hybrid.heuristic_types import QQAHeuristicConfig, QQAHeuristicStats

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from qqa.algebraic import AlgebraicModel
    from qqa.mixed import MixedProblem
else:
    AlgebraicModel = Any
    MixedProblem = Any
    NDArray = Any

try:  # Keep importing qqa independent from the optional SCIP wheel.
    from pyscipopt import Heur as _HeurBase
except (ImportError, OSError):  # pragma: no cover - optional dependency

    class _HeurBase:  # type: ignore[no-redef]
        pass


# Private compatibility wrappers for callers that exercised the pre-refactor
# test seams. Imports stay inside the wrappers so registering an inapplicable
# heuristic never pays Torch's process-startup cost.
def _core_problem(*args: Any, **kwargs: Any):
    from qqa.hybrid.core_problem import build_core_problem

    return build_core_problem(*args, **kwargs)


def _select_repair_positions(*args: Any, **kwargs: Any):
    from qqa.hybrid.heuristic_runtime import select_repair_positions

    return select_repair_positions(*args, **kwargs)


def _create_completion_template(model):
    from qqa.decomposition import create_completion_template

    return create_completion_template(model)


def complete_integer_assignment(*args: Any, **kwargs: Any):
    """Compatibility seam that keeps decomposition imports demand-driven."""
    from qqa.decomposition import complete_integer_assignment as complete

    return complete(*args, **kwargs)


def complete_integer_assignment_dive(*args: Any, **kwargs: Any):
    """Demand-driven wrapper around SCIP's in-place completion dive."""
    from qqa.decomposition import complete_integer_assignment_dive as complete

    return complete(*args, **kwargs)


class QQAHeuristic(_HeurBase):
    """PySCIPOpt ``Heur`` implementing root/node conditional QQA-LNS."""

    def __init__(
        self,
        config: QQAHeuristicConfig | None = None,
        *,
        completion_template=None,
        completion_template_factory: Callable[[], object] | None = None,
        algebraic: AlgebraicModel | None = None,
        incumbent_provider: Callable[[], NDArray | None] | None = None,
        feedback_bus=None,
    ):
        self.config = config or QQAHeuristicConfig()
        self.completion_template = completion_template
        self.completion_template_factory = completion_template_factory
        self.algebraic = algebraic
        self.incumbent_provider = incumbent_provider
        self.feedback_bus = feedback_bus
        self.stats = QQAHeuristicStats()
        self._last_node = -(10**18)
        self._archive: set[bytes] = set()
        self._lns_archive: set[bytes] = set()
        self._reference_pool: list[dict[str, float]] = []
        self._numerical_runtime_loaded = False
        self._active_callback_started_at: float | None = None

    def _reference_disagreement(self, state) -> NDArray:
        """Return node-aligned disagreement across recent LP references."""
        import numpy as np

        current = {
            state.names[index]: float(state.lp_values[index]) for index in state.integer_indices
        }
        references = [*self._reference_pool, current]
        disagreement = np.zeros(len(state.names), dtype=np.float64)
        if len(references) < 2:
            return disagreement
        span = np.maximum(1.0, state.local_upper - state.local_lower)
        for index in state.integer_indices:
            values = [
                reference.get(state.names[index], float(state.lp_values[index]))
                for reference in references
            ]
            disagreement[index] = (max(values) - min(values)) / span[index]
        return disagreement

    def _remember_reference(self, state) -> None:
        reference = {
            state.names[index]: float(state.lp_values[index]) for index in state.integer_indices
        }
        if self._reference_pool:
            previous = self._reference_pool[-1]
            shared = reference.keys() & previous.keys()
            if len(shared) == len(reference) == len(previous) and all(
                abs(reference[name] - previous[name]) <= 1e-9 for name in shared
            ):
                return
        self._reference_pool.append(reference)
        del self._reference_pool[: -self.config.reference_pool_size]

    def _reference_initial_values(
        self,
        state,
        selected_indices: NDArray,
        lower: NDArray,
        upper: NDArray,
    ) -> list[NDArray]:
        import numpy as np

        names = [state.names[index] for index in selected_indices]
        result: list[NDArray] = []
        for reference in reversed(self._reference_pool):
            if not all(name in reference for name in names):
                continue
            values = np.asarray([reference[name] for name in names], dtype=np.float64)
            result.append(np.minimum(np.maximum(values, lower), upper))
        return result

    def _with_external_incumbent(self, state):
        import numpy as np

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
        import numpy as np

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
        measured = self.stats.callback_runtime
        if self._active_callback_started_at is not None:
            measured += perf_counter() - self._active_callback_started_at
        # Component counters remain useful when private numerical helpers are
        # exercised directly, outside SCIP's registered callback wrapper.
        components = (
            self.stats.inspection_runtime
            + self.stats.numerical_runtime_initialisation
            + self.stats.qqa_runtime
            + self.stats.completion_runtime
        )
        used = max(measured, components)
        return max(0.0, self.config.maximum_overhead_fraction * limit - used)

    def _qqa_has_stalled(self) -> bool:
        return bool(
            self.config.stop_qqa_after_nonimproving_call
            and self.stats.qqa_calls > 0
            and self.stats.qqa_incumbent_improvements == 0
        )

    def _runtime_startup_is_affordable(self) -> bool:
        """Return whether the remaining plugin allowance can absorb cold start."""
        return self._remaining_overhead_budget() >= self.config.minimum_runtime_startup_time

    def _fast_path_supports_qqa(self, completion_feasible_before: int) -> bool:
        """Require completion evidence before escalating a primary hybrid call."""
        return bool(
            self.config.fast_candidates == 0
            or self.stats.completion_feasible > completion_feasible_before
        )

    def _rank_population(
        self,
        problem: MixedProblem,
        population: list[NDArray],
        state,
        selection,
        positions: list[int],
        selected_indices: NDArray,
        *,
        enforce_incumbent_filter: bool = True,
    ) -> list[NDArray]:
        import numpy as np
        import torch

        if not population:
            return []
        population_array = np.stack(population)
        incumbent_core = None
        if (
            enforce_incumbent_filter
            and self.config.require_surrogate_improvement
            and state.incumbent_values is not None
        ):
            incumbent_core = np.clip(
                np.rint(state.incumbent_values[selected_indices]),
                selection.local_lower[positions],
                selection.local_upper[positions],
            )
        evaluation_array = (
            np.concatenate([population_array, incumbent_core[None, :]], axis=0)
            if incumbent_core is not None
            else population_array
        )
        with torch.no_grad():
            evaluated = (
                problem.loss_fn(torch.as_tensor(evaluation_array, dtype=torch.float64))
                .detach()
                .cpu()
                .numpy()
            )
        losses = evaluated[: len(population_array)]
        incumbent_loss = float(evaluated[-1]) if incumbent_core is not None else None
        order = np.argsort(losses, kind="stable")
        if incumbent_loss is not None:
            retained = [index for index in order if float(losses[index]) < incumbent_loss - 1e-8]
            self.stats.filtered_candidates += len(order) - len(retained)
            order = np.asarray(retained, dtype=np.int64)
        return [population_array[index] for index in order]

    def _complete_population(
        self,
        population: list[NDArray],
        problem: MixedProblem,
        state,
        selection,
        positions: list[int],
        selected_indices: NDArray,
        neighborhood,
        seen_now: set[bytes],
        *,
        source: str,
    ) -> tuple[bool, bool]:
        import numpy as np

        from qqa.hybrid.heuristic_runtime import (
            rank_repair_candidates,
            select_repair_positions,
        )
        from qqa.hybrid.neighborhood import within_local_branching

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
            enforce_incumbent_filter=source != "qqa",
        )
        repair_plans: dict[bytes, list[int]] = {}
        if source == "qqa" and self.config.qqa_fix_fraction > 0:
            reference_values = (
                state.incumbent_values if state.incumbent_values is not None else state.lp_values
            )
            reference_core = np.minimum(
                np.maximum(
                    np.rint(reference_values[neighborhood.core_indices]),
                    neighborhood.lower,
                ),
                neighborhood.upper,
            )
            reference_reduced = reference_core[positions]
            maximum = max(
                1,
                math.ceil(self.config.qqa_fix_fraction * len(neighborhood.core_indices)),
            )
            ranked, repair_plans = rank_repair_candidates(
                problem,
                ranked,
                reference=reference_reduced,
                positions=positions,
                lower=selection.local_lower[positions],
                upper=selection.local_upper[positions],
                max_changes=maximum,
                beam_width=self.config.repair_beam_width,
                selector=select_repair_positions,
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
            qqa_fix_positions: list[int] = []
            if source == "qqa" and self.config.qqa_fix_fraction > 0:
                reference_values = (
                    state.incumbent_values
                    if state.incumbent_values is not None
                    else state.lp_values
                )
                reference_core = np.minimum(
                    np.maximum(
                        np.rint(reference_values[neighborhood.core_indices]),
                        neighborhood.lower,
                    ),
                    neighborhood.upper,
                )
                reduced_signature = np.asarray(
                    np.rint(core_values[positions]), dtype=np.int64
                ).tobytes()
                planned_positions = repair_plans.get(reduced_signature)
                qqa_fix_positions = (
                    planned_positions
                    if planned_positions is not None
                    else _select_repair_positions(
                        problem,
                        reference_core[positions],
                        core_values[positions],
                        positions,
                        max_changes=max(
                            1,
                            math.ceil(
                                self.config.qqa_fix_fraction * len(neighborhood.core_indices)
                            ),
                        ),
                        beam_width=self.config.repair_beam_width,
                    )
                )
            completion_started = perf_counter()
            if self.config.use_dive_completion:
                self.stats.dive_attempts += 1
                anchor_assignments = None
                if state.incumbent_values is not None:
                    anchor_assignments = np.minimum(
                        np.maximum(
                            np.rint(state.incumbent_values[indices]),
                            state.local_lower[indices],
                        ),
                        state.local_upper[indices],
                    )
                assignment_positions = {
                    int(index): position for position, index in enumerate(indices)
                }
                score_order = np.argsort(selection.scores, kind="stable")[::-1].tolist()
                full_change_order = list(dict.fromkeys([*qqa_fix_positions, *score_order]))
                change_order = [
                    assignment_positions[int(selection.core_indices[position])]
                    for position in full_change_order
                    if int(selection.core_indices[position]) in assignment_positions
                ]
                completed = complete_integer_assignment_dive(
                    self.model,
                    [state.variables[index] for index in indices],
                    assignments,
                    heuristic=self,
                    algebraic=self.algebraic,
                    lp_iterations=self.config.dive_lp_iterations,
                    anchor_values=anchor_assignments,
                    change_order=change_order,
                    max_repair_changes=self.config.dive_max_repair_changes,
                    minimum_relative_improvement=(self.config.minimum_relative_improvement),
                )
                self.stats.dive_feasible += int(completed.feasible)
                self.stats.dive_accepted += int(completed.accepted)
            else:
                completed = complete_integer_assignment(
                    self.completion_template,
                    [state.names[index] for index in indices],
                    assignments,
                    main_model=self.model,
                    heuristic=self,
                    algebraic=self.algebraic,
                    time_limit=min(
                        self.config.completion_time,
                        max(0.05, remaining * 0.5),
                        overhead_remaining,
                    ),
                    node_limit=self.config.completion_nodes,
                    seed=self.config.seed + self.stats.calls - 1,
                    minimum_relative_improvement=self.config.minimum_relative_improvement,
                    verbose=self.config.verbose,
                )
            first_runtime = perf_counter() - completion_started
            self.stats.completion_runtime += first_runtime
            repaired = False
            partial_repair = False
            repair_indices = np.concatenate(
                [
                    neighborhood.fixed_indices,
                    neighborhood.core_indices[qqa_fix_positions],
                ]
            )
            repair_values = np.concatenate(
                [
                    neighborhood.fixed_values,
                    core_values[qqa_fix_positions],
                ]
            )
            repair_signature = np.stack([repair_indices, repair_values.astype(np.int64)]).tobytes()
            if (
                not completed.improved_incumbent
                and self.config.subscip_repair
                and (
                    self.completion_template is not None
                    or self.completion_template_factory is not None
                )
                and qqa_fix_positions
                and repair_signature not in self._lns_archive
            ):
                remaining = self._remaining_time()
                overhead_remaining = self._remaining_overhead_budget()
                repair_budget = min(
                    self.config.completion_time,
                    max(0.0, remaining * 0.5),
                    overhead_remaining,
                )
                if repair_budget > 0.05:
                    repair_started = perf_counter()
                    if self.completion_template is None:
                        factory = self.completion_template_factory
                        if factory is None:
                            raise RuntimeError("Completion template factory is unavailable.")
                        self.completion_template = factory()
                    self._lns_archive.add(repair_signature)
                    partial_repair = state.incumbent_values is None
                    if partial_repair:
                        self.stats.partial_lns_attempts += 1
                    else:
                        self.stats.lns_repair_attempts += 1
                    completed = complete_integer_assignment(
                        self.completion_template,
                        [state.names[index] for index in repair_indices],
                        repair_values.tolist(),
                        main_model=self.model,
                        heuristic=self,
                        algebraic=self.algebraic,
                        time_limit=repair_budget,
                        node_limit=max(1, self.config.completion_nodes // 2),
                        seed=self.config.seed + self.stats.calls - 1,
                        minimum_relative_improvement=self.config.minimum_relative_improvement,
                        verbose=self.config.verbose,
                    )
                    self.stats.completion_runtime += perf_counter() - repair_started
                    repaired = True
            self.stats.completion_feasible += int(completed.feasible)
            self.stats.completion_statuses[completed.status] = (
                self.stats.completion_statuses.get(completed.status, 0) + 1
            )
            self.stats.completion_objectives.append(completed.objective)
            self.stats.completion_incumbents_before.append(external_before)
            relative_improvement = None
            if completed.objective is not None and external_before is not None:
                scale = max(1.0, abs(completed.objective), abs(external_before))
                relative_improvement = (
                    (completed.objective - external_before) / scale
                    if self.algebraic is not None and self.algebraic.objective_sense == "maximize"
                    else (external_before - completed.objective) / scale
                )
            self.stats.completion_relative_improvements.append(relative_improvement)
            self.stats.accepted += int(completed.accepted)
            original_improvement = bool(
                completed.accepted
                and (
                    self._is_original_improvement(completed.objective, external_before)
                    if external_before is not None
                    else completed.improved_incumbent
                )
            )
            self.stats.incumbent_improvements += int(original_improvement)
            if self.config.use_dive_completion and not repaired:
                self.stats.dive_incumbent_improvements += int(original_improvement)
            if repaired:
                if partial_repair:
                    self.stats.partial_lns_feasible += int(completed.feasible)
                    self.stats.partial_lns_accepted += int(completed.accepted)
                    self.stats.partial_lns_incumbent_improvements += int(original_improvement)
                else:
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

    def heurexec(self, heurtiming, nodeinfeasible):
        """Account for the complete plugin callback, including lazy startup."""
        callback_started = perf_counter()
        self._active_callback_started_at = callback_started
        try:
            return self._heurexec_impl(heurtiming, nodeinfeasible)
        finally:
            self.stats.callback_runtime += perf_counter() - callback_started
            self._active_callback_started_at = None

    def _heurexec_impl(
        self,
        heurtiming,
        nodeinfeasible,
    ):  # noqa: ARG002 - SCIP callback signature
        from pyscipopt import SCIP_LPSOLSTAT, SCIP_RESULT

        self.stats.callbacks += 1
        if nodeinfeasible:
            self.stats.node_infeasible_skips += 1
            return {"result": SCIP_RESULT.DIDNOTRUN}
        if self.stats.calls >= self.config.max_calls:
            self.stats.call_limit_skips += 1
            return {"result": SCIP_RESULT.DIDNOTRUN}
        if self.model.getLPSolstat() != SCIP_LPSOLSTAT.OPTIMAL:
            self.stats.lp_status_skips += 1
            return {"result": SCIP_RESULT.DIDNOTRUN}
        best_solution = self.model.getBestSol()
        if self.config.require_incumbent and best_solution is None:
            self.stats.incumbent_skips += 1
            return {"result": SCIP_RESULT.DIDNOTRUN}
        node_number = int(self.model.getNNodes())
        if node_number - self._last_node < self.config.min_nodes_between_calls:
            self.stats.node_spacing_skips += 1
            return {"result": SCIP_RESULT.DIDNOTRUN}
        if self._remaining_time() <= max(
            self.config.minimum_call_time,
            self.config.completion_time,
        ):
            self.stats.remaining_time_skips += 1
            return {"result": SCIP_RESULT.DIDNOTRUN}
        try:
            time_limit = float(self.model.getParam("limits/time"))
            elapsed = float(self.model.getSolvingTime())
        except Exception:
            time_limit = math.inf
            elapsed = 0.0
        if (
            math.isfinite(time_limit)
            and time_limit < 1e19
            and elapsed
            > min(
                self.config.maximum_call_time,
                self.config.maximum_call_time_fraction * time_limit,
            )
        ):
            self.stats.late_call_skips += 1
            return {"result": SCIP_RESULT.DIDNOTRUN}

        self._last_node = node_number
        self.stats.prechecks += 1
        try:
            branch_candidates = self.model.getLPBranchCands()
            lp_candidate_count = int(branch_candidates[3]) + int(branch_candidates[5])
        except Exception:
            lp_candidate_count = None
        if lp_candidate_count is not None:
            self.stats.lp_candidate_counts.append(lp_candidate_count)
            if lp_candidate_count < self.config.minimum_core_size:
                self.stats.small_core_skips += 1
                return {"result": SCIP_RESULT.DIDNOTRUN}
            if lp_candidate_count / self.config.core_size > self.config.maximum_core_saturation:
                self.stats.saturation_skips += 1
                return {"result": SCIP_RESULT.DIDNOTRUN}
        # Importing the optional numerical stack is an indivisible cold-start
        # operation. Do not begin it unless the complete callback-overhead
        # allowance has a conservative startup reserve. This keeps short
        # budgets on the native SCIP path instead of discovering only after a
        # costly import that the measured cap has already been consumed.
        if not self._runtime_startup_is_affordable():
            self.stats.runtime_budget_skips += 1
            return {"result": SCIP_RESULT.DIDNOTRUN}
        # Everything above is a cheap SCIP-only precheck. Load NumPy-based
        # state inspection only for a plausible callback; Torch remains lazy
        # until the selected core is known to satisfy every structural gate.
        import numpy as np

        from qqa.hybrid.core_selector import select_uncertain_integer_core
        from qqa.presolve import extract_scip_state

        inspection_started = perf_counter()
        try:
            state = self._with_external_incumbent(extract_scip_state(self.model))
            disagreement = self._reference_disagreement(state)
            if self.feedback_bus is not None:
                import torch

                self.feedback_bus.publish(
                    lp_primal=torch.as_tensor(state.lp_values, dtype=torch.float64),
                    reduced_costs=torch.as_tensor(state.reduced_costs, dtype=torch.float64),
                    branch_scores=torch.as_tensor(disagreement, dtype=torch.float64),
                    fractionalities=torch.as_tensor(
                        np.abs(state.lp_values - np.rint(state.lp_values)),
                        dtype=torch.float64,
                    ),
                    local_lower=torch.as_tensor(state.local_lower, dtype=torch.float64),
                    local_upper=torch.as_tensor(state.local_upper, dtype=torch.float64),
                    incumbent=(
                        None
                        if state.incumbent_values is None
                        else torch.as_tensor(state.incumbent_values, dtype=torch.float64)
                    ),
                    metadata={"node": node_number, "source": "scip-lp"},
                )
            references_used = min(
                self.config.reference_pool_size,
                len(self._reference_pool) + 1,
            )
            self.stats.reference_pool_sizes.append(references_used)
            self.stats.multi_reference_calls += int(references_used > 1)
            self._remember_reference(state)
            selection = select_uncertain_integer_core(
                state,
                max_core_size=self.config.core_size,
                entropy=disagreement,
            )
            positions = [
                position
                for position, (lower, upper) in enumerate(
                    zip(selection.local_lower, selection.local_upper, strict=True)
                )
                if math.ceil(lower - 1e-9) < math.floor(upper + 1e-9)
            ]
            self.stats.inspection_runtime += perf_counter() - inspection_started
            if not positions:
                return {"result": SCIP_RESULT.DIDNOTFIND}
            selected_indices = selection.core_indices[positions]
            if len(selected_indices) < self.config.minimum_core_size:
                self.stats.small_core_skips += 1
                return {"result": SCIP_RESULT.DIDNOTRUN}
            if len(selected_indices) / self.config.core_size > self.config.maximum_core_saturation:
                self.stats.saturation_skips += 1
                return {"result": SCIP_RESULT.DIDNOTRUN}
            # This is the first point at which a QQA call can be useful. The
            # heavy optimisation stack is deliberately imported here rather
            # than at plugin registration or at every SCIP-only callback.
            runtime_started = perf_counter()
            from qqa.hybrid.core_problem import build_core_problem
            from qqa.hybrid.heuristic_runtime import (
                build_initial_population,
                solve_core_problem,
            )
            from qqa.hybrid.neighborhood import build_neighborhood
            from qqa.hybrid.surrogate import (
                build_core_surrogate,
                generate_surrogate_candidates,
            )

            if not self._numerical_runtime_loaded:
                self.stats.numerical_runtime_loads += 1
                self.stats.numerical_runtime_initialisation += perf_counter() - runtime_started
                self._numerical_runtime_loaded = True

            self.stats.calls += 1
            self.stats.call_nodes.append(node_number)
            self.stats.call_times.append(float(self.model.getSolvingTime()))
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
            problem, _ = build_core_problem(
                state,
                selection,
                positions,
                surrogate,
                self.config,
                adaptive_rows=bool(
                    self.config.adaptive_row_lagrangian
                    and (
                        state.incumbent_values is None
                        or (self.algebraic is not None and self.algebraic.problem_type is not None)
                    )
                ),
            )
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
                initial_seeds = [initial_values]
                if state.incumbent_values is not None:
                    initial_seeds.append(state.incumbent_values[selected_indices])
                initial_seeds.extend(
                    self._reference_initial_values(
                        state,
                        selected_indices,
                        local_lower,
                        local_upper,
                    )
                )
                initial_seeds.extend(fast_population)
                initial_population = build_initial_population(
                    initial_seeds,
                    target=initial_values,
                    lower=local_lower,
                    upper=local_upper,
                    sol_size=self.config.sol_size,
                    seed=self.config.seed + self.stats.calls - 1,
                )
                qqa_started = perf_counter()
                result = solve_core_problem(
                    problem,
                    initial_population,
                    self.config,
                    seed=self.config.seed + self.stats.calls - 1,
                    time_limit=qqa_budget,
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
    incumbent_provider: Callable[[], NDArray | None] | None = None,
    completion_template_factory: Callable[[], object] | None = None,
    feedback_bus=None,
) -> QQAHeuristic:
    """Include conditional QQA at useful LP-node timings and return the plugin."""
    try:
        from pyscipopt import SCIP_HEURTIMING
    except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
        raise ImportError("SCIP-guided QQA requires `qqa[scip]`.") from exc
    resolved = config or QQAHeuristicConfig()
    needs_completion_template = resolved.subscip_repair or not resolved.use_dive_completion
    if not needs_completion_template:
        completion_template_factory = None
    completion_template = (
        _create_completion_template(model)
        if needs_completion_template and completion_template_factory is None
        else None
    )
    heuristic = QQAHeuristic(
        resolved,
        completion_template=completion_template,
        completion_template_factory=completion_template_factory,
        algebraic=algebraic,
        incumbent_provider=incumbent_provider,
        feedback_bus=feedback_bus,
    )
    timing = SCIP_HEURTIMING.AFTERLPNODE | SCIP_HEURTIMING.AFTERLPPLUNGE
    has_quadratic_algebraic_model = bool(
        algebraic is not None and algebraic.problem_type is not None
    )
    model.includeHeur(
        heuristic,
        "qqa_conditional",
        "SCIP-guided conditional QQA with continuous completion",
        "Q",
        # Linear MIPs benefit from SCIP's native fast/adaptive heuristics
        # running first.  QPLIB's quadratic objective/rows are unavailable to
        # those linear neighbourhood scores, so run QQA before the expensive
        # ALNS tier while the configured early-call window is still open.
        priority=(-1_100_000 if has_quadratic_algebraic_model else -1_200_000),
        freq=resolved.frequency,
        maxdepth=resolved.maximum_depth,
        timingmask=timing,
        usessubscip=(completion_template is not None or completion_template_factory is not None),
    )
    return heuristic


__all__ = [
    "QQAHeuristic",
    "QQAHeuristicConfig",
    "QQAHeuristicStats",
    "include_qqa_heuristic",
]
