"""RENS/RINS and local-branching neighbourhood descriptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qqa.hybrid.core_selector import CoreSelection
from qqa.presolve.scip_bridge import SCIPState


@dataclass(frozen=True, slots=True)
class IntegerNeighborhood:
    core_indices: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fixed_indices: np.ndarray
    fixed_values: np.ndarray
    incumbent: np.ndarray | None = None
    radius: int | None = None
    kind: str = "rens"

    def complete_assignment(
        self, core_values: np.ndarray, state: SCIPState
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(core_values, dtype=np.float64)
        if values.shape != self.core_indices.shape:
            raise ValueError("core_values must align with core_indices.")
        values = np.rint(np.minimum(np.maximum(values, self.lower), self.upper))
        indices = np.concatenate([self.core_indices, self.fixed_indices])
        assignments = np.concatenate([values, self.fixed_values])
        order = np.argsort(indices)
        indices = indices[order]
        assignments = assignments[order]
        assignments = np.minimum(
            np.maximum(assignments, state.local_lower[indices]),
            state.local_upper[indices],
        )
        return indices, assignments


def build_neighborhood(
    selection: CoreSelection,
    state: SCIPState,
    *,
    local_branching_radius: int | None = None,
) -> IntegerNeighborhood:
    if local_branching_radius is not None and (
        isinstance(local_branching_radius, bool)
        or not isinstance(local_branching_radius, int)
        or local_branching_radius < 1
    ):
        raise ValueError("local_branching_radius must be a positive integer or None.")
    incumbent = (
        np.rint(state.incumbent_values[selection.core_indices])
        if state.incumbent_values is not None
        else None
    )
    return IntegerNeighborhood(
        core_indices=selection.core_indices,
        lower=selection.local_lower,
        upper=selection.local_upper,
        fixed_indices=selection.fixed_indices,
        fixed_values=selection.fixed_values,
        incumbent=incumbent,
        radius=local_branching_radius,
        kind=selection.mode,
    )


def within_local_branching(neighborhood: IntegerNeighborhood, values: np.ndarray) -> bool:
    if neighborhood.radius is None or neighborhood.incumbent is None:
        return True
    candidate = np.rint(np.asarray(values, dtype=np.float64))
    binary = (neighborhood.lower == 0) & (neighborhood.upper == 1)
    distance = np.abs(candidate[~binary] - neighborhood.incumbent[~binary]).sum()
    distance += np.count_nonzero(candidate[binary] != neighborhood.incumbent[binary])
    return bool(distance <= neighborhood.radius)


__all__ = ["IntegerNeighborhood", "build_neighborhood", "within_local_branching"]
