"""Incremental GPU local-search primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qqa.compile import SparseQUBO


def binary_flip_delta(qubo: SparseQUBO, values: torch.Tensor) -> torch.Tensor:
    """Return exact QUBO energy differences for every possible bit flip."""
    if values.shape[-1] != qubo.num_variables:
        raise ValueError("values and QUBO dimensions do not align.")
    gradient = qubo.linear.to(values).expand_as(values).clone()
    if qubo.num_edges:
        source, target = qubo.edge_index.to(values.device)
        weights = qubo.edge_weight.to(values)
        shape = (1,) * (values.ndim - 1) + (len(weights),)
        source_index = source.reshape(shape).expand(*values.shape[:-1], -1)
        target_index = target.reshape(shape).expand(*values.shape[:-1], -1)
        gradient.scatter_add_(-1, source_index, values[..., target] * weights)
        gradient.scatter_add_(-1, target_index, values[..., source] * weights)
    return (1 - 2 * values) * gradient


@dataclass(frozen=True, slots=True)
class GPULocalSearchResult:
    solutions: torch.Tensor
    objectives: torch.Tensor
    moves: int


@torch.no_grad()
def gpu_k_flip_search(
    qubo: SparseQUBO,
    initial: torch.Tensor,
    *,
    maximum_moves: int = 100,
    tabu_tenure: int = 0,
) -> GPULocalSearchResult:
    """Run batched best-improvement one-flip/tabu search without host sync."""
    if maximum_moves < 0 or tabu_tenure < 0:
        raise ValueError("maximum_moves and tabu_tenure must be non-negative.")
    values = torch.as_tensor(initial).clone()
    if values.ndim == 1:
        values = values.unsqueeze(0)
    tabu = torch.zeros_like(values, dtype=torch.int64)
    for _ in range(maximum_moves):
        delta = binary_flip_delta(qubo, values)
        allowed = tabu <= 0
        candidate_delta = delta.masked_fill(~allowed, torch.inf)
        best_delta, index = candidate_delta.min(dim=-1)
        improve = best_delta < -1e-12
        proposal = values.clone()
        proposal.scatter_(-1, index.unsqueeze(-1), 1 - values.gather(-1, index.unsqueeze(-1)))
        values = torch.where(improve.unsqueeze(-1), proposal, values)
        tabu.sub_(1).clamp_min_(0)
        if tabu_tenure:
            tabu.scatter_(
                -1, index.unsqueeze(-1), torch.full_like(index.unsqueeze(-1), tabu_tenure)
            )
    return GPULocalSearchResult(values, qubo.energy(values), maximum_moves)


@torch.no_grad()
def gpu_two_opt(route: torch.Tensor, distances: torch.Tensor, *, passes: int = 4) -> torch.Tensor:
    """Apply bounded batched 2-opt route reversals on the active device."""
    values = torch.as_tensor(route, device=distances.device, dtype=torch.long).clone()
    if values.ndim == 1:
        values = values.unsqueeze(0)
    size = values.shape[-1]
    if size < 4:
        return values.squeeze(0) if route.ndim == 1 else values
    if distances.shape != (size, size):
        raise ValueError("distances must be square and match the route length.")
    pairs = torch.combinations(torch.arange(size, device=values.device), r=2)
    pairs = pairs[(pairs[:, 1] - pairs[:, 0]) > 1]
    for _ in range(passes):
        left, right = pairs[:, 0], pairs[:, 1]
        a = values[:, left]
        b = values[:, (left + 1) % size]
        c = values[:, right]
        d = values[:, (right + 1) % size]
        delta = distances[a, c] + distances[b, d] - distances[a, b] - distances[c, d]
        chosen = delta.argmin(dim=1)
        selected_pair = pairs[chosen]
        i, j = selected_pair[:, 0], selected_pair[:, 1]
        position = torch.arange(size, device=values.device).expand(len(values), -1)
        reversed_position = i[:, None] + j[:, None] + 1 - position
        gather_index = torch.where(
            (position > i[:, None]) & (position <= j[:, None]),
            reversed_position,
            position,
        )
        proposal = values.gather(1, gather_index)
        improve = delta.gather(1, chosen[:, None]).squeeze(1) < -1e-12
        values = torch.where(improve[:, None], proposal, values)
    return values.squeeze(0) if route.ndim == 1 else values


__all__ = ["GPULocalSearchResult", "binary_flip_delta", "gpu_k_flip_search", "gpu_two_opt"]
