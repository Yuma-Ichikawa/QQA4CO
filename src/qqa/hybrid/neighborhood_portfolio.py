"""Explainable LNS neighbourhood portfolio with online bandit allocation."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from qqa.hybrid.core_selector import CoreSelection
from qqa.hybrid.neighborhood import IntegerNeighborhood, build_neighborhood
from qqa.presolve.scip_bridge import SCIPState


@dataclass(frozen=True, slots=True)
class NeighborhoodBudget:
    max_variables: int = 64
    radius: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_variables, bool) or self.max_variables < 1:
            raise ValueError("max_variables must be a positive integer.")
        if self.radius is not None and (isinstance(self.radius, bool) or self.radius < 1):
            raise ValueError("radius must be a positive integer or None.")


class NeighborhoodGenerator(Protocol):
    name: str

    def propose(
        self,
        state: SCIPState,
        budget: NeighborhoodBudget,
    ) -> IntegerNeighborhood: ...


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    if not len(values):
        return values
    spread = np.ptp(values)
    return np.zeros_like(values) if spread <= 1e-12 else (values - values.min()) / spread


def _state_signal(state: SCIPState, name: str, size: int) -> np.ndarray:
    values = getattr(state, name, None)
    if values is None:
        return np.zeros(size, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    return array if array.shape == (size,) else np.zeros(size, dtype=np.float64)


@dataclass(slots=True)
class ScoredNeighborhoodGenerator:
    """Select a variable block from one or more model-state signals."""

    name: str
    signal: str
    external_scores: np.ndarray | None = None

    def _scores(self, state: SCIPState, integer: np.ndarray) -> np.ndarray:
        lp = state.lp_values[integer]
        if self.signal == "fractionality":
            return np.abs(lp - np.rint(lp))
        if self.signal == "incumbent-disagreement":
            if state.incumbent_values is None:
                return np.abs(lp - np.rint(lp))
            return np.abs(lp - state.incumbent_values[integer])
        if self.signal == "pseudocost":
            return np.abs(state.pseudocosts[integer])
        if self.signal == "reduced-cost":
            return np.abs(state.reduced_costs[integer])
        state_fields = {
            "conflict": "conflict_scores",
            "gradient": "gradient_scores",
            "history": "historical_scores",
        }
        if self.signal in state_fields:
            if self.external_scores is None or len(self.external_scores) != len(state.variables):
                scores = _state_signal(state, state_fields[self.signal], len(state.variables))
            else:
                scores = np.asarray(self.external_scores, dtype=np.float64)
            return scores[integer]
        if self.signal == "reference-disagreement":
            history = tuple(state.reference_history)
            if not history:
                return np.abs(lp - np.rint(lp))
            references = np.stack(
                [np.asarray(values, dtype=np.float64) for values in history],
                axis=0,
            )
            if references.shape[1:] != (len(state.variables),):
                return np.abs(lp - np.rint(lp))
            return references[:, integer].std(axis=0)
        raise ValueError(f"Unknown neighbourhood signal {self.signal!r}.")

    def propose(self, state: SCIPState, budget: NeighborhoodBudget) -> IntegerNeighborhood:
        integer = state.integer_indices
        active = integer[state.local_upper[integer] > state.local_lower[integer] + 1e-9]
        if not len(active):
            empty = np.empty(0, dtype=np.int64)
            selection = CoreSelection(
                empty,
                integer,
                state.local_lower[integer].copy(),
                np.empty(0),
                np.empty(0),
                np.empty(0),
                self.name,
            )
            return build_neighborhood(selection, state, local_branching_radius=budget.radius)
        scores = _normalise(self._scores(state, active))
        order = np.argsort(scores, kind="stable")[::-1]
        core = np.sort(active[order[: budget.max_variables]])
        selected = np.isin(integer, core)
        fixed = integer[~selected]
        reference = (
            state.incumbent_values if state.incumbent_values is not None else state.lp_values
        )
        fixed_values = np.rint(reference[fixed])
        fixed_values = np.minimum(
            np.maximum(fixed_values, state.local_lower[fixed]), state.local_upper[fixed]
        )
        selection = CoreSelection(
            core,
            fixed,
            fixed_values,
            state.local_lower[core].copy(),
            state.local_upper[core].copy(),
            scores[order[: budget.max_variables]],
            self.name,
        )
        return build_neighborhood(selection, state, local_branching_radius=budget.radius)


@dataclass(slots=True)
class GraphInducedNeighborhoodGenerator:
    """GINS-style connected variable block grown from the interaction graph."""

    name: str = "gins"

    def propose(self, state: SCIPState, budget: NeighborhoodBudget) -> IntegerNeighborhood:
        integer = state.integer_indices
        active = integer[state.local_upper[integer] > state.local_lower[integer] + 1e-9]
        if not len(active):
            return ScoredNeighborhoodGenerator(self.name, "fractionality").propose(state, budget)

        active_set = set(active.tolist())
        adjacency: dict[int, list[int]] = {index: [] for index in active_set}
        edges = state.interaction_edges
        if edges is not None:
            edge_array = np.asarray(edges, dtype=np.int64)
            if edge_array.ndim == 2 and edge_array.shape[0] == 2:
                for left, right in edge_array.T:
                    if int(left) in active_set and int(right) in active_set:
                        adjacency[int(left)].append(int(right))
                        adjacency[int(right)].append(int(left))

        fractionality = np.abs(state.lp_values[active] - np.rint(state.lp_values[active]))
        disagreement = (
            np.zeros(len(active))
            if state.incumbent_values is None
            else np.abs(state.lp_values[active] - state.incumbent_values[active])
        )
        priority = _normalise(fractionality) + _normalise(disagreement)
        seed_order = active[np.argsort(priority, kind="stable")[::-1]]
        selected: list[int] = []
        seen: set[int] = set()
        for seed in seed_order:
            if len(selected) >= budget.max_variables:
                break
            queue = deque([int(seed)])
            while queue and len(selected) < budget.max_variables:
                index = queue.popleft()
                if index in seen:
                    continue
                seen.add(index)
                selected.append(index)
                queue.extend(
                    sorted(
                        adjacency.get(index, ()),
                        key=lambda item: (-priority[np.where(active == item)[0][0]], item),
                    )
                )
        core = np.asarray(sorted(selected), dtype=np.int64)
        selected_mask = np.isin(integer, core)
        fixed = integer[~selected_mask]
        reference = (
            state.incumbent_values if state.incumbent_values is not None else state.lp_values
        )
        fixed_values = np.rint(reference[fixed])
        fixed_values = np.minimum(
            np.maximum(fixed_values, state.local_lower[fixed]),
            state.local_upper[fixed],
        )
        selection = CoreSelection(
            core,
            fixed,
            fixed_values,
            state.local_lower[core].copy(),
            state.local_upper[core].copy(),
            priority[np.searchsorted(active, core)] if len(core) else np.empty(0),
            self.name,
        )
        return build_neighborhood(selection, state, local_branching_radius=budget.radius)


@dataclass(slots=True)
class LocalBranchingNeighborhoodGenerator:
    """RINS core with an explicit incumbent-distance budget."""

    name: str = "local-branching"

    def propose(self, state: SCIPState, budget: NeighborhoodBudget) -> IntegerNeighborhood:
        radius = budget.radius or max(1, int(math.sqrt(budget.max_variables)))
        neighborhood = ScoredNeighborhoodGenerator(
            self.name,
            "incumbent-disagreement",
        ).propose(state, replace(budget, radius=radius))
        return replace(neighborhood, kind=self.name)


@dataclass(slots=True)
class TrustRegionNeighborhoodGenerator:
    """Integer neighbourhood with local bounds around LP/incumbent references."""

    name: str = "trust-region"

    def propose(self, state: SCIPState, budget: NeighborhoodBudget) -> IntegerNeighborhood:
        neighborhood = ScoredNeighborhoodGenerator(self.name, "fractionality").propose(
            state,
            budget,
        )
        radius = budget.radius or max(1, int(math.sqrt(budget.max_variables)))
        reference = (
            state.incumbent_values if state.incumbent_values is not None else state.lp_values
        )[neighborhood.core_indices]
        lower = np.maximum(neighborhood.lower, np.floor(reference - radius))
        upper = np.minimum(neighborhood.upper, np.ceil(reference + radius))
        return replace(neighborhood, lower=lower, upper=upper, kind=self.name)


@dataclass(slots=True)
class NeighborhoodStats:
    calls: int = 0
    feasible_solutions: int = 0
    accepted_incumbents: int = 0
    objective_gain: float = 0.0
    runtime: float = 0.0
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def reward(self) -> float:
        if self.calls == 0:
            return 0.0
        quality = self.objective_gain + 0.1 * self.accepted_incumbents
        return quality / max(self.runtime, 1e-6)


class NeighborhoodPortfolio:
    """UCB1 allocation over RENS/RINS/GINS/local model-state signals."""

    def __init__(
        self,
        generators: tuple[NeighborhoodGenerator, ...] | None = None,
        *,
        exploration: float = 1.0,
    ) -> None:
        if not math.isfinite(exploration) or exploration < 0:
            raise ValueError("exploration must be finite and non-negative.")
        if generators is None:
            generators = (
                ScoredNeighborhoodGenerator("rens", "fractionality"),
                ScoredNeighborhoodGenerator("rins", "incumbent-disagreement"),
                GraphInducedNeighborhoodGenerator(),
                LocalBranchingNeighborhoodGenerator(),
                TrustRegionNeighborhoodGenerator(),
                ScoredNeighborhoodGenerator("conflict", "conflict"),
                ScoredNeighborhoodGenerator("pseudocost", "pseudocost"),
                ScoredNeighborhoodGenerator("gradient", "gradient"),
                ScoredNeighborhoodGenerator("history", "history"),
                ScoredNeighborhoodGenerator("reduced-cost", "reduced-cost"),
            )
        if not generators or len({item.name for item in generators}) != len(generators):
            raise ValueError("Neighbourhood generators must be non-empty and uniquely named.")
        self.generators = tuple(generators)
        self.exploration = float(exploration)
        self.stats = {item.name: NeighborhoodStats() for item in generators}

    def select(self) -> NeighborhoodGenerator:
        unseen = [item for item in self.generators if self.stats[item.name].calls == 0]
        if unseen:
            return unseen[0]
        total = sum(record.calls for record in self.stats.values())
        return max(
            self.generators,
            key=lambda item: (
                self.stats[item.name].reward
                + self.exploration
                * math.sqrt(math.log(max(2, total)) / self.stats[item.name].calls),
                item.name,
            ),
        )

    def propose(
        self, state: SCIPState, budget: NeighborhoodBudget
    ) -> tuple[str, IntegerNeighborhood]:
        generator = self.select()
        return generator.name, generator.propose(state, budget)

    def update(
        self,
        name: str,
        *,
        runtime: float,
        feasible: bool,
        accepted: bool,
        objective_gain: float = 0.0,
        failure: str | None = None,
    ) -> None:
        if name not in self.stats:
            raise KeyError(f"Unknown neighbourhood generator {name!r}.")
        if not math.isfinite(runtime) or runtime < 0:
            raise ValueError("runtime must be finite and non-negative.")
        record = self.stats[name]
        record.calls += 1
        record.runtime += runtime
        record.feasible_solutions += int(feasible)
        record.accepted_incumbents += int(accepted)
        record.objective_gain += max(0.0, float(objective_gain))
        if failure:
            record.failures[failure] = record.failures.get(failure, 0) + 1

    def diagnostics(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "calls": row.calls,
                "feasible_solutions": row.feasible_solutions,
                "accepted_incumbents": row.accepted_incumbents,
                "objective_gain": row.objective_gain,
                "runtime": row.runtime,
                "failures": dict(row.failures),
            }
            for name, row in self.stats.items()
        }


__all__ = [
    "NeighborhoodBudget",
    "NeighborhoodGenerator",
    "NeighborhoodPortfolio",
    "NeighborhoodStats",
    "GraphInducedNeighborhoodGenerator",
    "LocalBranchingNeighborhoodGenerator",
    "ScoredNeighborhoodGenerator",
    "TrustRegionNeighborhoodGenerator",
]
