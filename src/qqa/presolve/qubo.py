"""Provably safe persistency, probing, dominance, and symmetry passes for QUBOs."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import torch

from qqa.compile import SparseQUBO


@dataclass(frozen=True, slots=True)
class PersistencyResult:
    fixings: dict[int, int]
    optimum_or_lower_bound: float
    exact: bool
    method: str


@torch.no_grad()
def dominance_fixings(qubo: SparseQUBO, *, tolerance: float = 1e-12) -> dict[int, int]:
    """Fix variables whose marginal has one strict sign over the full hypercube."""
    minimum = qubo.linear.detach().clone()
    maximum = qubo.linear.detach().clone()
    if qubo.num_edges:
        source, target = qubo.edge_index
        negative = qubo.edge_weight.clamp_max(0)
        positive = qubo.edge_weight.clamp_min(0)
        minimum.scatter_add_(0, source, negative)
        minimum.scatter_add_(0, target, negative)
        maximum.scatter_add_(0, source, positive)
        maximum.scatter_add_(0, target, positive)
    fixings: dict[int, int] = {}
    for index in range(qubo.num_variables):
        if float(minimum[index].item()) > tolerance:
            fixings[index] = 0
        elif float(maximum[index].item()) < -tolerance:
            fixings[index] = 1
    return fixings


@torch.no_grad()
def exact_probe_persistency(
    qubo: SparseQUBO,
    *,
    max_variables: int = 24,
    chunk_size: int = 1 << 16,
    tolerance: float = 1e-9,
) -> PersistencyResult:
    """Enumerate a bounded QUBO exactly and retain variables shared by all optima."""
    n = qubo.num_variables
    if n > max_variables:
        raise ValueError(f"Exact probing is limited to {max_variables} variables.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    working = qubo.to("cpu", torch.float64)
    shifts = torch.arange(n, dtype=torch.int64)
    best = torch.tensor(torch.inf, dtype=torch.float64)
    all_zero = torch.ones(n, dtype=torch.bool)
    all_one = torch.ones(n, dtype=torch.bool)
    found = False
    for start in range(0, 1 << n, chunk_size):
        identifiers = torch.arange(start, min(start + chunk_size, 1 << n), dtype=torch.int64)
        assignments = ((identifiers[:, None] >> shifts) & 1).to(torch.float64)
        energies = working.energy(assignments)
        chunk_best = energies.min()
        if chunk_best < best - tolerance:
            best = chunk_best
            mask = energies <= best + tolerance
            minima = assignments[mask].bool()
            all_zero = ~minima.any(dim=0)
            all_one = minima.all(dim=0)
            found = True
        elif chunk_best <= best + tolerance:
            minima = assignments[energies <= best + tolerance].bool()
            all_zero &= ~minima.any(dim=0)
            all_one &= minima.all(dim=0)
    if not found:
        raise RuntimeError("Exact probing failed to evaluate any assignments.")
    fixings = {
        index: int(all_one[index].item())
        for index in range(n)
        if bool(all_zero[index] | all_one[index])
    }
    return PersistencyResult(fixings, float(best.item()), True, "exact-probing")


def _submodular_cut(
    qubo: SparseQUBO,
    *,
    forced: tuple[int, int] | None = None,
) -> torch.Tensor:
    source_node = "__source__"
    sink_node = "__sink__"
    graph = nx.DiGraph()
    graph.add_nodes_from((source_node, sink_node, *range(qubo.num_variables)))
    unary = qubo.linear.detach().cpu().to(torch.float64).clone()
    source, target = qubo.edge_index.detach().cpu()
    weights = qubo.edge_weight.detach().cpu().to(torch.float64)
    unary.scatter_add_(0, source, weights / 2)
    unary.scatter_add_(0, target, weights / 2)
    for index, coefficient in enumerate(unary.tolist()):
        if coefficient > 0:
            graph.add_edge(source_node, index, capacity=coefficient)
        elif coefficient < 0:
            graph.add_edge(index, sink_node, capacity=-coefficient)
    for left, right, coefficient in zip(
        source.tolist(), target.tolist(), weights.tolist(), strict=True
    ):
        capacity = -coefficient / 2
        if capacity:
            graph.add_edge(left, right, capacity=capacity)
            graph.add_edge(right, left, capacity=capacity)
    if forced is not None:
        index, value = forced
        capacity = (
            float(qubo.linear.detach().abs().sum().item())
            + float(qubo.edge_weight.detach().abs().sum().item())
            + 1.0
        )
        if value == 0:
            graph.add_edge(source_node, index, capacity=capacity)
        else:
            graph.add_edge(index, sink_node, capacity=capacity)
    _, partition = nx.minimum_cut(graph, source_node, sink_node, capacity="capacity")
    reachable, _ = partition
    return torch.tensor(
        [0.0 if index in reachable else 1.0 for index in range(qubo.num_variables)],
        dtype=qubo.linear.dtype,
        device=qubo.linear.device,
    )


@torch.no_grad()
def submodular_roof_duality(
    qubo: SparseQUBO,
    *,
    tolerance: float = 1e-9,
) -> PersistencyResult:
    """Solve a submodular QUBO by graph cut and prove strong persistencies.

    Opposite-value probing certifies a variable only when every optimum agrees.
    General non-submodular instances can use exact probing on bounded components.
    """
    if bool((qubo.edge_weight > tolerance).any()):
        raise ValueError("Roof-duality graph cuts require non-positive pair coefficients.")
    optimum_solution = _submodular_cut(qubo)
    optimum = float(qubo.energy(optimum_solution).item())
    fixings: dict[int, int] = {}
    for index, value in enumerate(optimum_solution.to(torch.int64).tolist()):
        forced = _submodular_cut(qubo, forced=(index, 1 - value))
        forced_value = float(qubo.energy(forced).item())
        if forced_value > optimum + tolerance:
            fixings[index] = value
    return PersistencyResult(fixings, optimum, True, "submodular-roof-duality")


@torch.no_grad()
def general_qpbo_persistency(
    qubo: SparseQUBO,
    *,
    exact_component_limit: int = 24,
    tolerance: float = 1e-9,
) -> PersistencyResult:
    """Return safe persistencies and a bound for any binary quadratic energy.

    Submodular instances use graph-cut roof duality.  Small non-submodular
    instances use exact probing.  Larger instances return the valid termwise
    relaxation bound with no speculative fixings; this conservative fallback
    is intentionally labelled non-exact and never overstates QPBO persistency.
    """
    if not bool((qubo.edge_weight > tolerance).any()):
        return submodular_roof_duality(qubo, tolerance=tolerance)
    if qubo.num_variables <= exact_component_limit:
        result = exact_probe_persistency(
            qubo,
            max_variables=exact_component_limit,
            tolerance=tolerance,
        )
        return PersistencyResult(
            result.fixings,
            result.optimum_or_lower_bound,
            True,
            "general-qpbo-exact-probing",
        )
    lower_bound = (
        float(qubo.constant)
        + float(qubo.linear.clamp_max(0).sum().item())
        + float(qubo.edge_weight.clamp_max(0).sum().item())
    )
    return PersistencyResult({}, lower_bound, False, "general-qpbo-termwise-relaxation")


def detect_qubo_symmetries(
    qubo: SparseQUBO,
    *,
    tolerance: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    """Return variable groups whose pairwise swaps leave the QUBO unchanged."""
    adjacency: list[dict[int, float]] = [dict() for _ in range(qubo.num_variables)]
    for left, right, weight in zip(
        qubo.edge_index.detach().cpu()[0].tolist(),
        qubo.edge_index.detach().cpu()[1].tolist(),
        qubo.edge_weight.detach().cpu().tolist(),
        strict=True,
    ):
        adjacency[left][right] = float(weight)
        adjacency[right][left] = float(weight)
    linear = qubo.linear.detach().cpu().tolist()
    parent = list(range(qubo.num_variables))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(qubo.num_variables):
        for right in range(left + 1, qubo.num_variables):
            if abs(float(linear[left]) - float(linear[right])) > tolerance:
                continue
            if all(
                abs(adjacency[left].get(other, 0.0) - adjacency[right].get(other, 0.0)) <= tolerance
                for other in range(qubo.num_variables)
                if other not in {left, right}
            ):
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(qubo.num_variables):
        groups.setdefault(find(index), []).append(index)
    return tuple(tuple(group) for group in groups.values() if len(group) > 1)


__all__ = [
    "PersistencyResult",
    "detect_qubo_symmetries",
    "dominance_fixings",
    "exact_probe_persistency",
    "general_qpbo_persistency",
    "submodular_roof_duality",
]
