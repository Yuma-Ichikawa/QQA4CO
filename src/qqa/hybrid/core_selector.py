"""SCIP-guided uncertain integer core selection for QQA neighbourhoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qqa.presolve.scip_bridge import SCIPState


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


@dataclass(frozen=True, slots=True)
class CoreSelection:
    """Selected integer indices, fixed complement and node-local bounds."""

    core_indices: np.ndarray
    fixed_indices: np.ndarray
    fixed_values: np.ndarray
    local_lower: np.ndarray
    local_upper: np.ndarray
    scores: np.ndarray
    mode: str


def select_uncertain_integer_core(
    state: SCIPState,
    *,
    max_core_size: int = 64,
    integrality_tolerance: float = 1e-6,
    entropy: np.ndarray | None = None,
    conflict_scores: np.ndarray | None = None,
    fractionality_weight: float = 1.0,
    incumbent_weight: float = 0.6,
    pseudocost_weight: float = 0.3,
    reduced_cost_weight: float = 0.2,
    entropy_weight: float = 0.2,
    conflict_weight: float = 0.2,
    integer_radius: int = 2,
) -> CoreSelection:
    """Select a RENS/RINS-style integer neighbourhood from active SCIP state."""
    if isinstance(max_core_size, bool) or not isinstance(max_core_size, int) or max_core_size < 1:
        raise ValueError("max_core_size must be a positive integer.")
    if not 0 <= integrality_tolerance < 0.5:
        raise ValueError("integrality_tolerance must be in [0, 0.5).")
    if (
        isinstance(integer_radius, bool)
        or not isinstance(integer_radius, int)
        or integer_radius < 1
    ):
        raise ValueError("integer_radius must be a positive integer.")
    all_integer = state.integer_indices
    if all_integer.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return CoreSelection(
            empty, empty, np.empty(0), np.empty(0), np.empty(0), np.empty(0), "none"
        )
    active_mask = (
        state.local_upper[all_integer] - state.local_lower[all_integer] > integrality_tolerance
    )
    bound_fixed = all_integer[~active_mask]
    integer = all_integer[active_mask]
    if integer.size == 0:
        values = state.local_lower[bound_fixed].copy()
        empty = np.empty(0, dtype=np.int64)
        return CoreSelection(
            empty,
            bound_fixed,
            values,
            np.empty(0),
            np.empty(0),
            np.empty(0),
            "fixed",
        )

    lp = state.lp_values[integer]
    nearest = np.rint(lp)
    fractionality = np.abs(lp - nearest)
    span = np.maximum(1.0, state.local_upper[integer] - state.local_lower[integer])
    incumbent_delta = (
        np.abs(lp - state.incumbent_values[integer]) / span
        if state.incumbent_values is not None
        else np.zeros_like(lp)
    )
    local_entropy = np.zeros_like(lp) if entropy is None else np.asarray(entropy)[integer]
    conflicts = (
        np.zeros_like(lp) if conflict_scores is None else np.asarray(conflict_scores)[integer]
    )
    score = (
        fractionality_weight * _normalise(fractionality)
        + incumbent_weight * _normalise(incumbent_delta)
        + pseudocost_weight * _normalise(np.abs(state.pseudocosts[integer]))
        + reduced_cost_weight * _normalise(np.abs(state.reduced_costs[integer]))
        + entropy_weight * _normalise(local_entropy)
        + conflict_weight * _normalise(conflicts)
    )

    if state.incumbent_values is None:
        uncertain = fractionality > integrality_tolerance
        mode = "rens"
    else:
        incumbent = np.rint(state.incumbent_values[integer])
        uncertain = (fractionality > integrality_tolerance) | (np.abs(nearest - incumbent) > 0)
        mode = "rins"
    candidates = np.where(uncertain)[0]
    if candidates.size == 0:
        candidates = np.argsort(score)[-min(max_core_size, len(integer)) :]
    order = candidates[np.argsort(score[candidates], kind="stable")[::-1]]
    selected_positions = np.sort(order[:max_core_size])
    selected_mask = np.zeros(len(integer), dtype=bool)
    selected_mask[selected_positions] = True
    core = integer[selected_positions]
    fixed = np.concatenate([integer[~selected_mask], bound_fixed])

    reference = state.incumbent_values if state.incumbent_values is not None else state.lp_values
    fixed_values = np.rint(reference[fixed])
    fixed_values = np.minimum(
        np.maximum(fixed_values, state.local_lower[fixed]), state.local_upper[fixed]
    )

    local_lower = state.local_lower[core].copy()
    local_upper = state.local_upper[core].copy()
    for position, index in enumerate(core):
        if state.variable_types[index] == "BINARY":
            local_lower[position], local_upper[position] = 0.0, 1.0
            continue
        centre = state.lp_values[index]
        # Keep a genuine integer neighbourhood instead of collapsing every
        # general integer to floor/ceil of the LP point.  In RINS mode include
        # both LP and incumbent centres, then clip to active node bounds.
        centres = [centre]
        if state.incumbent_values is not None:
            centres.append(state.incumbent_values[index])
        local_lower[position] = max(local_lower[position], np.floor(min(centres)) - integer_radius)
        local_upper[position] = min(local_upper[position], np.ceil(max(centres)) + integer_radius)
        if local_lower[position] >= local_upper[position] and state.incumbent_values is not None:
            incumbent = round(float(state.incumbent_values[index]))
            local_lower[position] = max(state.local_lower[index], incumbent - 1)
            local_upper[position] = min(state.local_upper[index], incumbent + 1)

    return CoreSelection(
        core_indices=core,
        fixed_indices=fixed,
        fixed_values=fixed_values,
        local_lower=local_lower,
        local_upper=local_upper,
        scores=score[selected_positions],
        mode=mode,
    )


__all__ = ["CoreSelection", "select_uncertain_integer_core"]
