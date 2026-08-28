"""McCormick envelopes and a bounded spatial branch-and-bound runtime."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class BilinearTerm:
    left: int
    right: int
    coefficient: float


@dataclass(frozen=True, slots=True)
class McCormickEnvelope:
    left_lower: float
    left_upper: float
    right_lower: float
    right_upper: float

    def bounds(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower = torch.maximum(
            self.left_lower * right + self.right_lower * left - self.left_lower * self.right_lower,
            self.left_upper * right + self.right_upper * left - self.left_upper * self.right_upper,
        )
        upper = torch.minimum(
            self.left_upper * right + self.right_lower * left - self.left_upper * self.right_lower,
            self.left_lower * right + self.right_upper * left - self.left_lower * self.right_upper,
        )
        return lower, upper


@dataclass(frozen=True, slots=True)
class SpatialBranchAndBoundResult:
    solution: torch.Tensor
    objective: float
    lower_bound: float
    gap: float
    nodes: int
    proven: bool


def spatial_branch_and_bound(
    objective: Callable[[torch.Tensor], torch.Tensor],
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    relaxation_bound: Callable[[torch.Tensor, torch.Tensor], float],
    tolerance: float = 1e-4,
    maximum_nodes: int = 10_000,
) -> SpatialBranchAndBoundResult:
    """Globally search a bounded box using user-supplied valid lower bounds."""
    lower = torch.as_tensor(lower, dtype=torch.float64).reshape(-1)
    upper = torch.as_tensor(upper, dtype=torch.float64).reshape(-1)
    if (
        lower.shape != upper.shape
        or torch.any(lower > upper)
        or not torch.isfinite(lower).all()
        or not torch.isfinite(upper).all()
    ):
        raise ValueError("Spatial bounds must be aligned, finite, and ordered.")
    if tolerance <= 0 or maximum_nodes < 1:
        raise ValueError("tolerance and maximum_nodes must be positive.")
    incumbent = (lower + upper) / 2
    incumbent_value = float(objective(incumbent).reshape(()).item())
    root_bound = float(relaxation_bound(lower, upper))
    queue: list[tuple[float, int, torch.Tensor, torch.Tensor]] = [(root_bound, 0, lower, upper)]
    nodes = 0
    sequence = 1
    while queue and nodes < maximum_nodes:
        bound, _, node_lower, node_upper = heapq.heappop(queue)
        if bound >= incumbent_value - tolerance:
            continue
        midpoint = (node_lower + node_upper) / 2
        value = float(objective(midpoint).reshape(()).item())
        if value < incumbent_value:
            incumbent, incumbent_value = midpoint, value
        widths = node_upper - node_lower
        split = int(torch.argmax(widths).item())
        if float(widths[split].item()) <= tolerance:
            continue
        for is_upper in (False, True):
            child_lower = node_lower.clone()
            child_upper = node_upper.clone()
            if is_upper:
                child_lower[split] = midpoint[split]
            else:
                child_upper[split] = midpoint[split]
            child_bound = float(relaxation_bound(child_lower, child_upper))
            if not math.isfinite(child_bound):
                raise ValueError("relaxation_bound must return finite valid lower bounds.")
            if child_bound < incumbent_value - tolerance:
                heapq.heappush(queue, (child_bound, sequence, child_lower, child_upper))
                sequence += 1
        nodes += 1
    lower_bound = min((item[0] for item in queue), default=incumbent_value)
    gap = max(0.0, incumbent_value - lower_bound) / max(1.0, abs(incumbent_value), abs(lower_bound))
    return SpatialBranchAndBoundResult(
        incumbent,
        incumbent_value,
        lower_bound,
        gap,
        nodes,
        not queue or gap <= tolerance,
    )


__all__ = [
    "BilinearTerm",
    "McCormickEnvelope",
    "SpatialBranchAndBoundResult",
    "spatial_branch_and_bound",
]
