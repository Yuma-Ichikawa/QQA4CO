"""Structure-specific discrete refinements kept outside problem classes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class StructuredSearchResult:
    solution: torch.Tensor
    objective: float
    moves: int
    method: str


def _tour_length(distance: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    return distance[tour, torch.roll(tour, -1)].sum()


@torch.no_grad()
def two_opt_tour(
    distance: torch.Tensor,
    tour: torch.Tensor,
    *,
    max_passes: int = 20,
) -> StructuredSearchResult:
    matrix = torch.as_tensor(distance)
    current = torch.as_tensor(tour, dtype=torch.long, device=matrix.device).clone()
    n = len(current)
    if matrix.shape != (n, n) or torch.unique(current).numel() != n:
        raise ValueError("distance must be square and tour a full permutation.")
    moves = 0
    for _ in range(max_passes):
        best_delta = 0.0
        best_pair: tuple[int, int] | None = None
        for left in range(n - 1):
            for right in range(left + 2, n if left else n - 1):
                a, b = current[left], current[(left + 1) % n]
                c, d = current[right], current[(right + 1) % n]
                delta = float((matrix[a, c] + matrix[b, d] - matrix[a, b] - matrix[c, d]).item())
                if delta < best_delta - 1e-12:
                    best_delta, best_pair = delta, (left + 1, right)
        if best_pair is None:
            break
        left, right = best_pair
        current[left : right + 1] = current[left : right + 1].flip(0)
        moves += 1
    return StructuredSearchResult(
        current, float(_tour_length(matrix, current).item()), moves, "2-opt"
    )


@torch.no_grad()
def three_opt_tour(
    distance: torch.Tensor,
    tour: torch.Tensor,
    *,
    max_passes: int = 5,
) -> StructuredSearchResult:
    """Bounded 3-opt using the four non-cyclic segment reconnections."""
    matrix = torch.as_tensor(distance)
    current = two_opt_tour(matrix, tour, max_passes=max_passes).solution
    n = len(current)
    moves = 0
    for _ in range(max_passes):
        incumbent = float(_tour_length(matrix, current).item())
        best_value = incumbent
        best = None
        for i in range(1, n - 3):
            for j in range(i + 1, n - 2):
                for k in range(j + 1, n):
                    a, b, c, d = current[:i], current[i:j], current[j:k], current[k:]
                    variants = (
                        torch.cat((a, b.flip(0), c, d)),
                        torch.cat((a, b, c.flip(0), d)),
                        torch.cat((a, c, b, d)),
                        torch.cat((a, c.flip(0), b.flip(0), d)),
                    )
                    values = torch.stack([_tour_length(matrix, item) for item in variants])
                    index = int(torch.argmin(values).item())
                    value = float(values[index].item())
                    if value < best_value - 1e-12:
                        best_value, best = value, variants[index]
        if best is None:
            break
        current = best
        moves += 1
    return StructuredSearchResult(
        current, float(_tour_length(matrix, current).item()), moves, "3-opt"
    )


@torch.no_grad()
def maxcut_fm_search(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    partition: torch.Tensor,
    *,
    max_passes: int = 20,
) -> StructuredSearchResult:
    edges = torch.as_tensor(edge_index, dtype=torch.long)
    weights = torch.as_tensor(edge_weight, device=edges.device, dtype=torch.float64)
    current = torch.as_tensor(partition, device=edges.device).round().clamp(0, 1)

    def objective(values: torch.Tensor) -> torch.Tensor:
        return -(weights * (values[edges[0]] != values[edges[1]])).sum()

    value = objective(current)
    moves = 0
    for _ in range(max_passes):
        candidates = current.repeat(len(current), 1)
        diagonal = torch.arange(len(current), device=current.device)
        candidates[diagonal, diagonal] = 1 - candidates[diagonal, diagonal]
        scores = torch.stack([objective(item) for item in candidates])
        index = int(torch.argmin(scores).item())
        if scores[index] >= value - 1e-12:
            break
        current.copy_(candidates[index])
        value = scores[index]
        moves += 1
    return StructuredSearchResult(current, float(value.item()), moves, "FM")


@torch.no_grad()
def mis_swap_search(
    edge_index: torch.Tensor,
    solution: torch.Tensor,
    *,
    max_passes: int = 20,
) -> StructuredSearchResult:
    edges = torch.as_tensor(edge_index, dtype=torch.long)
    proposed = torch.as_tensor(solution, device=edges.device).round().clamp(0, 1)
    adjacency = torch.zeros(
        (len(proposed), len(proposed)), dtype=torch.bool, device=proposed.device
    )
    adjacency[edges[0], edges[1]] = True
    adjacency[edges[1], edges[0]] = True
    current = torch.zeros_like(proposed)
    priority = torch.argsort(proposed, descending=True, stable=True)
    for vertex in priority.tolist():
        if not bool(adjacency[vertex, current > 0.5].any()):
            current[vertex] = 1
    moves = 0
    for _ in range(max_passes):
        selected = torch.nonzero(current > 0.5, as_tuple=False).reshape(-1)
        excluded = torch.nonzero(current < 0.5, as_tuple=False).reshape(-1)
        addable = (
            excluded[~adjacency[excluded][:, selected].any(dim=1)] if len(selected) else excluded
        )
        if len(addable):
            current[addable] = 1
            moves += len(addable)
            continue
        improved = False
        for removed in selected.tolist():
            remaining = selected[selected != removed]
            candidates = (
                excluded[~adjacency[excluded][:, remaining].any(dim=1)]
                if len(remaining)
                else excluded
            )
            if len(candidates) < 2:
                continue
            compatible = ~adjacency[candidates][:, candidates]
            compatible.fill_diagonal_(False)
            pair = torch.nonzero(torch.triu(compatible, diagonal=1), as_tuple=False)
            if len(pair):
                current[removed] = 0
                current[candidates[pair[0]]] = 1
                moves += 3
                improved = True
                break
        if not improved:
            break
    return StructuredSearchResult(current, -float(current.sum().item()), moves, "MIS-swap")


@torch.no_grad()
def kempe_coloring_search(
    edge_index: torch.Tensor,
    colors: torch.Tensor,
    *,
    num_colors: int | None = None,
    max_passes: int = 50,
) -> StructuredSearchResult:
    edges = torch.as_tensor(edge_index, dtype=torch.long)
    current = torch.as_tensor(colors, device=edges.device, dtype=torch.long).clone()
    if not current.numel():
        raise ValueError("colors must be non-empty.")
    k = int(current.max().item()) + 1 if num_colors is None else num_colors
    if k < 1 or torch.any(current < 0) or torch.any(current >= k):
        raise ValueError("colors must lie in [0, num_colors).")

    def conflicts(values: torch.Tensor) -> torch.Tensor:
        return (values[edges[0]] == values[edges[1]]).sum()

    neighbours: list[list[int]] = [[] for _ in range(len(current))]
    for left, right in edges.detach().cpu().T.tolist():
        neighbours[left].append(right)
        neighbours[right].append(left)

    moves = 0
    vertices = torch.arange(len(current), device=current.device).repeat_interleave(k)
    palette = torch.arange(k, device=current.device).repeat(len(current))
    for _ in range(max_passes):
        baseline = conflicts(current)
        if int(baseline.item()) == 0:
            break

        candidates = current.repeat(len(vertices), 1)
        candidates[torch.arange(len(vertices), device=current.device), vertices] = palette
        scores = torch.stack([conflicts(item) for item in candidates])
        best_index = int(torch.argmin(scores).item())
        if scores[best_index] < baseline:
            current.copy_(candidates[best_index])
            moves += 1
            continue

        best_score = baseline
        best_swap: torch.Tensor | None = None
        conflicting = torch.unique(edges[:, current[edges[0]] == current[edges[1]]])
        for start in conflicting.detach().cpu().tolist():
            first = int(current[start].item())
            for alternate in range(k):
                if alternate == first:
                    continue
                component = {start}
                frontier = [start]
                while frontier:
                    vertex = frontier.pop()
                    for neighbour in neighbours[vertex]:
                        colour = int(current[neighbour].item())
                        if colour in (first, alternate) and neighbour not in component:
                            component.add(neighbour)
                            frontier.append(neighbour)
                swapped = current.clone()
                indices = torch.tensor(sorted(component), device=current.device)
                original = current[indices]
                swapped[indices[original == first]] = alternate
                swapped[indices[original == alternate]] = first
                score = conflicts(swapped)
                if score < best_score:
                    best_score = score
                    best_swap = swapped
        if best_swap is None:
            break
        current = best_swap
        moves += 1
    return StructuredSearchResult(current, float(conflicts(current).item()), moves, "Kempe")


@torch.no_grad()
def walksat_search(
    indices: torch.Tensor,
    signs: torch.Tensor,
    assignment: torch.Tensor,
    *,
    max_flips: int = 1000,
    noise: float = 0.35,
    seed: int = 0,
) -> StructuredSearchResult:
    clauses = torch.as_tensor(indices, dtype=torch.long)
    polarity = torch.as_tensor(signs, device=clauses.device)
    current = torch.as_tensor(assignment, device=clauses.device).round().clamp(0, 1)
    generator = torch.Generator(device=current.device).manual_seed(seed)

    def unsatisfied(values: torch.Tensor) -> torch.Tensor:
        literals = torch.where(polarity > 0, values[clauses], 1 - values[clauses])
        return ~literals.bool().any(dim=-1)

    moves = 0
    for _ in range(max_flips):
        failed = torch.nonzero(unsatisfied(current), as_tuple=False).reshape(-1)
        if not len(failed):
            break
        clause = int(
            failed[
                torch.randint(len(failed), (), generator=generator, device=current.device)
            ].item()
        )
        variables = clauses[clause]
        if torch.rand((), generator=generator, device=current.device) < noise:
            index = int(
                variables[
                    torch.randint(len(variables), (), generator=generator, device=current.device)
                ].item()
            )
        else:
            candidates = current.repeat(len(variables), 1)
            rows = torch.arange(len(variables), device=current.device)
            candidates[rows, variables] = 1 - candidates[rows, variables]
            scores = torch.stack([unsatisfied(item).sum() for item in candidates])
            index = int(variables[torch.argmin(scores)].item())
        current[index] = 1 - current[index]
        moves += 1
    return StructuredSearchResult(
        current,
        float(unsatisfied(current).sum().item()),
        moves,
        "WalkSAT",
    )


__all__ = [
    "StructuredSearchResult",
    "kempe_coloring_search",
    "maxcut_fm_search",
    "mis_swap_search",
    "three_opt_tour",
    "two_opt_tour",
    "walksat_search",
]
