"""Bounded advanced local-search portfolio for sparse QUBO incumbents."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qqa.compile import SparseQUBO
from qqa.local.sparse_qubo import sparse_qubo_descent


@dataclass(frozen=True, slots=True)
class QUBOLocalSearchResult:
    solution: torch.Tensor
    objective: float
    method: str
    moves: int


@torch.no_grad()
def tabu_search(
    qubo: SparseQUBO,
    solution: torch.Tensor,
    *,
    iterations: int = 500,
    tenure: int = 11,
) -> QUBOLocalSearchResult:
    """Best-admissible one-flip tabu search with aspiration."""
    if iterations < 0 or tenure < 1:
        raise ValueError("iterations must be non-negative and tenure positive.")
    current = solution.to(qubo.linear).round().clamp(0, 1)
    current_value = float(qubo.energy(current).item())
    best = current.clone()
    best_value = current_value
    tabu_until = torch.zeros(qubo.num_variables, dtype=torch.long, device=current.device)
    moves = 0
    for step in range(iterations):
        delta = qubo.flip_delta(current)
        candidate_values = current_value + delta
        admissible = (tabu_until <= step) | (candidate_values < best_value - 1e-12)
        if not bool(admissible.any()):
            break
        ranked = candidate_values.masked_fill(~admissible, torch.inf)
        index = int(torch.argmin(ranked).item())
        current[index] = 1 - current[index]
        current_value = float(candidate_values[index].item())
        tabu_until[index] = step + tenure
        moves += 1
        if current_value < best_value - 1e-12:
            best.copy_(current)
            best_value = current_value
    return QUBOLocalSearchResult(best, best_value, "tabu", moves)


@torch.no_grad()
def k_flip_search(
    qubo: SparseQUBO,
    solution: torch.Tensor,
    *,
    candidate_width: int = 24,
    max_rounds: int = 20,
) -> QUBOLocalSearchResult:
    """Batched best-improvement search over one- and two-flip moves."""
    if candidate_width < 2 or max_rounds < 0:
        raise ValueError("candidate_width must be >=2 and max_rounds non-negative.")
    current = solution.to(qubo.linear).round().clamp(0, 1)
    value = float(qubo.energy(current).item())
    moves = 0
    for _ in range(max_rounds):
        width = min(candidate_width, qubo.num_variables)
        promising = torch.topk(qubo.flip_delta(current), k=width, largest=False).indices
        left, right = torch.triu_indices(width, width, offset=1, device=current.device)
        pair_count = left.numel()
        candidates = current.repeat(width + pair_count, 1)
        candidates[torch.arange(width, device=current.device), promising] = (
            1 - candidates[torch.arange(width, device=current.device), promising]
        )
        if pair_count:
            rows = torch.arange(pair_count, device=current.device) + width
            first = promising[left]
            second = promising[right]
            candidates[rows, first] = 1 - candidates[rows, first]
            candidates[rows, second] = 1 - candidates[rows, second]
        energies = qubo.energy(candidates)
        best_index = int(torch.argmin(energies).item())
        best_value = float(energies[best_index].item())
        if best_value >= value - 1e-12:
            break
        moves += 1 if best_index < width else 2
        current.copy_(candidates[best_index])
        value = best_value
    return QUBOLocalSearchResult(current, value, "k-flip", moves)


@torch.no_grad()
def path_relink(
    qubo: SparseQUBO,
    start: torch.Tensor,
    target: torch.Tensor,
) -> QUBOLocalSearchResult:
    """Follow the best objective move among coordinates that differ from target."""
    current = start.to(qubo.linear).round().clamp(0, 1)
    destination = target.to(qubo.linear).round().clamp(0, 1)
    if current.shape != destination.shape or current.numel() != qubo.num_variables:
        raise ValueError("start and target must match the QUBO variable vector.")
    best = current.clone()
    best_value = float(qubo.energy(current).item())
    moves = 0
    while True:
        differing = torch.nonzero(current != destination, as_tuple=False).reshape(-1)
        if not len(differing):
            break
        delta = qubo.flip_delta(current)[differing]
        index = int(differing[torch.argmin(delta)].item())
        current[index] = destination[index]
        moves += 1
        value = float(qubo.energy(current).item())
        if value < best_value:
            best.copy_(current)
            best_value = value
    polished = sparse_qubo_descent(qubo, best)
    return QUBOLocalSearchResult(
        polished.solution,
        polished.objective,
        "path-relink",
        moves + polished.flips,
    )


@torch.no_grad()
def iterated_local_search(
    qubo: SparseQUBO,
    solution: torch.Tensor,
    *,
    restarts: int = 12,
    perturbation: int = 3,
    seed: int = 0,
) -> QUBOLocalSearchResult:
    """Repeated sparse descent after deterministic random perturbations."""
    if restarts < 0 or perturbation < 1:
        raise ValueError("restarts must be non-negative and perturbation positive.")
    initial = sparse_qubo_descent(qubo, solution)
    best = initial.solution
    best_value = initial.objective
    moves = initial.flips
    generator = torch.Generator(device=best.device).manual_seed(seed)
    for _ in range(restarts):
        candidate = best.clone()
        count = min(perturbation, qubo.num_variables)
        indices = torch.randperm(qubo.num_variables, generator=generator, device=best.device)[
            :count
        ]
        candidate[indices] = 1 - candidate[indices]
        polished = sparse_qubo_descent(qubo, candidate)
        moves += count + polished.flips
        if polished.objective < best_value:
            best = polished.solution
            best_value = polished.objective
    return QUBOLocalSearchResult(best, best_value, "iterated", moves)


__all__ = [
    "QUBOLocalSearchResult",
    "iterated_local_search",
    "k_flip_search",
    "path_relink",
    "tabu_search",
]
