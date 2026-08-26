"""Incremental one-flip descent for sparse QUBOs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qqa.compile import SparseQUBO


@dataclass(frozen=True, slots=True)
class LocalSearchResult:
    solution: torch.Tensor
    objective: float
    flips: int
    passes: int


def _adjacency(qubo: SparseQUBO) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source, target = qubo.edge_index
    row = torch.cat((source, target))
    neighbour = torch.cat((target, source))
    weight = torch.cat((qubo.edge_weight, qubo.edge_weight))
    order = torch.argsort(row)
    row = row[order]
    neighbour = neighbour[order]
    weight = weight[order]
    counts = torch.bincount(row, minlength=qubo.num_variables)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))
    return offsets, neighbour, weight


@torch.no_grad()
def sparse_qubo_descent(
    qubo: SparseQUBO,
    solution: torch.Tensor,
    *,
    max_flips: int | None = None,
    tolerance: float = 1e-10,
) -> LocalSearchResult:
    """Best-improvement descent with degree-local delta updates.

    The initial deltas cost ``O(E)``.  After a flip only adjacent variables
    are updated, so sparse instances avoid repeated full objective calls.
    """
    if solution.ndim != 1 or solution.numel() != qubo.num_variables:
        raise ValueError(f"solution must have shape ({qubo.num_variables},).")
    if max_flips is None:
        max_flips = max(1, 10 * qubo.num_variables)
    if isinstance(max_flips, bool) or not isinstance(max_flips, int) or max_flips < 0:
        raise ValueError("max_flips must be a non-negative integer or None.")
    x = solution.to(device=qubo.linear.device, dtype=qubo.linear.dtype).round().clamp_(0, 1)
    delta = qubo.flip_delta(x)
    offsets, neighbours, weights = _adjacency(qubo)
    flips = 0
    for _ in range(max_flips):
        best_delta, selected = torch.min(delta, dim=0)
        if float(best_delta.item()) >= -tolerance:
            break
        index = int(selected.item())
        old_sign = 1.0 - 2.0 * x[index]
        start = int(offsets[index].item())
        stop = int(offsets[index + 1].item())
        adjacent = neighbours[start:stop]
        edge_weights = weights[start:stop]
        if adjacent.numel():
            delta[adjacent] += (1.0 - 2.0 * x[adjacent]) * edge_weights * old_sign
        x[index] = 1.0 - x[index]
        delta[index] = -best_delta
        flips += 1
    objective = float(qubo.energy(x).item())
    return LocalSearchResult(x, objective, flips, 1)


__all__ = ["LocalSearchResult", "sparse_qubo_descent"]
