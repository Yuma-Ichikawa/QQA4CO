"""Native sparse factors for common scheduling, logical, and network constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from qqa.model.ir import ObjectiveIR, _tensor


def _indices(value, *, minimum: int = 1) -> torch.Tensor:
    result = _tensor(value, dtype=torch.long).reshape(-1)
    if result.numel() < minimum or torch.any(result < 0):
        raise ValueError(f"Expected at least {minimum} non-negative variable indices.")
    return result


def _positive_weight(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("factor weight must be finite and positive.")


@dataclass(frozen=True, slots=True)
class IndicatorFactor:
    indicator: int
    expression: ObjectiveIR
    sense: Literal["<=", ">=", "=="] = "<="
    rhs: float = 0.0
    active_value: int = 1
    weight: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.indicator, bool) or self.indicator < 0:
            raise ValueError("indicator must be a non-negative integer.")
        if self.sense not in {"<=", ">=", "=="} or self.active_value not in {0, 1}:
            raise ValueError("Invalid indicator sense or active_value.")
        if not math.isfinite(self.rhs) or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("rhs must be finite and weight must be finite and positive.")

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.expression.evaluate(values) - self.rhs
        violation = (
            residual.abs()
            if self.sense == "=="
            else residual.clamp_min(0.0)
            if self.sense == "<="
            else (-residual).clamp_min(0.0)
        )
        gate = values[..., self.indicator]
        if self.active_value == 0:
            gate = 1.0 - gate
        return self.weight * gate * violation.square()


@dataclass(frozen=True, slots=True)
class SOS1Factor:
    indices: torch.Tensor
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", _indices(self.indices, minimum=2))
        _positive_weight(self.weight)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        selected = values[..., self.indices].abs()
        return self.weight * torch.triu(
            selected.unsqueeze(-1) * selected.unsqueeze(-2), diagonal=1
        ).sum(dim=(-2, -1))


@dataclass(frozen=True, slots=True)
class SOS2Factor:
    indices: torch.Tensor
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", _indices(self.indices, minimum=3))
        _positive_weight(self.weight)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        selected = values[..., self.indices].abs()
        pair = selected.unsqueeze(-1) * selected.unsqueeze(-2)
        size = len(self.indices)
        non_adjacent = torch.triu(
            torch.ones((size, size), dtype=torch.bool, device=values.device), diagonal=2
        )
        return self.weight * pair[..., non_adjacent].sum(dim=-1)


@dataclass(frozen=True, slots=True)
class PiecewiseLinearFactor:
    index: int
    breakpoints: torch.Tensor
    outputs: torch.Tensor

    def __post_init__(self) -> None:
        points = _tensor(self.breakpoints, dtype=torch.float64).reshape(-1)
        outputs = _tensor(self.outputs, dtype=torch.float64).reshape(-1)
        if self.index < 0 or len(points) < 2 or points.shape != outputs.shape:
            raise ValueError("Piecewise-linear points/outputs must align and contain >=2 values.")
        if not torch.all(points[1:] > points[:-1]):
            raise ValueError("Piecewise-linear breakpoints must be strictly increasing.")
        object.__setattr__(self, "breakpoints", points)
        object.__setattr__(self, "outputs", outputs)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        x = values[..., self.index]
        points = self.breakpoints.to(values)
        outputs = self.outputs.to(values)
        segment = torch.searchsorted(points, x).clamp(1, len(points) - 1)
        left = segment - 1
        fraction = (x - points[left]) / (points[segment] - points[left])
        return outputs[left] + fraction * (outputs[segment] - outputs[left])


@dataclass(frozen=True, slots=True)
class LogicalFactor:
    """AND/OR/XOR relation where the final index is the output variable."""

    indices: torch.Tensor
    operation: Literal["and", "or", "xor"]
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", _indices(self.indices, minimum=3))
        if self.operation not in {"and", "or", "xor"}:
            raise ValueError("operation must be and, or, or xor.")
        _positive_weight(self.weight)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        inputs = values[..., self.indices[:-1]]
        output = values[..., self.indices[-1]]
        if self.operation == "and":
            expected = inputs.prod(dim=-1)
        elif self.operation == "or":
            expected = 1.0 - (1.0 - inputs).prod(dim=-1)
        else:
            expected = torch.remainder(inputs.sum(dim=-1), 2.0)
        return self.weight * (output - expected).square()


@dataclass(frozen=True, slots=True)
class PrecedenceFactor:
    before: torch.Tensor
    after: torch.Tensor
    durations: torch.Tensor
    weight: float = 1.0

    def __post_init__(self) -> None:
        before = _indices(self.before)
        after = _indices(self.after)
        durations = _tensor(self.durations, dtype=torch.float64).reshape(-1)
        if (
            before.shape != after.shape
            or before.shape != durations.shape
            or torch.any(durations < 0)
        ):
            raise ValueError("Precedence arrays must align and durations be non-negative.")
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)
        object.__setattr__(self, "durations", durations)
        _positive_weight(self.weight)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        violation = (
            values[..., self.before] + self.durations.to(values) - values[..., self.after]
        ).clamp_min(0.0)
        return self.weight * violation.square().sum(dim=-1)


@dataclass(frozen=True, slots=True)
class NoOverlapFactor:
    starts: torch.Tensor
    durations: torch.Tensor
    weight: float = 1.0

    def __post_init__(self) -> None:
        starts = _indices(self.starts, minimum=2)
        durations = _tensor(self.durations, dtype=torch.float64).reshape(-1)
        if starts.shape != durations.shape or torch.any(durations < 0):
            raise ValueError("No-overlap starts/durations must align and be non-negative.")
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "durations", durations)
        _positive_weight(self.weight)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        starts = values[..., self.starts]
        ends = starts + self.durations.to(values)
        left, right = torch.triu_indices(
            len(self.starts), len(self.starts), 1, device=values.device
        )
        overlap = (
            torch.minimum(ends[..., left], ends[..., right])
            - torch.maximum(starts[..., left], starts[..., right])
        ).clamp_min(0.0)
        return self.weight * overlap.square().sum(dim=-1)


@dataclass(frozen=True, slots=True)
class CumulativeResourceFactor:
    starts: torch.Tensor
    durations: torch.Tensor
    demands: torch.Tensor
    capacity: float
    time_points: torch.Tensor
    temperature: float = 0.1
    weight: float = 1.0

    def __post_init__(self) -> None:
        starts = _indices(self.starts)
        durations = _tensor(self.durations, dtype=torch.float64).reshape(-1)
        demands = _tensor(self.demands, dtype=torch.float64).reshape(-1)
        time_points = _tensor(self.time_points, dtype=torch.float64).reshape(-1)
        if starts.shape != durations.shape or starts.shape != demands.shape:
            raise ValueError("Cumulative resource arrays must align.")
        if (
            torch.any(durations < 0)
            or torch.any(demands < 0)
            or not math.isfinite(self.capacity)
            or self.capacity < 0
        ):
            raise ValueError("Durations, demands, and capacity must be non-negative.")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not len(time_points):
            raise ValueError("temperature and time_points must be positive/non-empty.")
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "demands", demands)
        object.__setattr__(self, "time_points", time_points)
        _positive_weight(self.weight)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        starts = values[..., self.starts].unsqueeze(-1)
        times = self.time_points.to(values)
        active = torch.sigmoid((times - starts) / self.temperature) * torch.sigmoid(
            (starts + self.durations.to(values).unsqueeze(-1) - times) / self.temperature
        )
        load = (active * self.demands.to(values).unsqueeze(-1)).sum(dim=-2)
        return self.weight * (load - self.capacity).clamp_min(0.0).square().sum(dim=-1)


@dataclass(frozen=True, slots=True)
class FlowConservationFactor:
    edge_indices: torch.Tensor
    incidence: torch.Tensor
    supplies: torch.Tensor
    weight: float = 1.0

    def __post_init__(self) -> None:
        edges = _indices(self.edge_indices)
        incidence = _tensor(self.incidence, dtype=torch.float64)
        supplies = _tensor(self.supplies, dtype=torch.float64).reshape(-1)
        if incidence.ndim != 2 or incidence.shape != (len(supplies), len(edges)):
            raise ValueError("incidence must have shape (nodes, selected edges).")
        object.__setattr__(self, "edge_indices", edges)
        object.__setattr__(self, "incidence", incidence)
        object.__setattr__(self, "supplies", supplies)
        _positive_weight(self.weight)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        residual = values[..., self.edge_indices] @ self.incidence.to(values).T - self.supplies.to(
            values
        )
        return self.weight * residual.square().sum(dim=-1)


@dataclass(frozen=True, slots=True)
class MatchingFactor:
    groups: torch.Tensor
    target: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        groups = _tensor(self.groups, dtype=torch.long)
        if groups.ndim != 2 or torch.any(groups < 0):
            raise ValueError("Matching groups must be a non-negative rank-two index tensor.")
        if not math.isfinite(self.target):
            raise ValueError("Matching target must be finite.")
        _positive_weight(self.weight)
        object.__setattr__(self, "groups", groups)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        residual = values[..., self.groups].sum(dim=-1) - self.target
        return self.weight * residual.square().sum(dim=-1)


@dataclass(frozen=True, slots=True)
class SubtourEliminationFactor:
    """Penalty for explicit subsets in a flattened directed edge matrix."""

    edge_indices: torch.Tensor
    subsets: tuple[torch.Tensor, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        edges = _tensor(self.edge_indices, dtype=torch.long)
        if edges.ndim != 2 or edges.shape[0] != edges.shape[1] or torch.any(edges < 0):
            raise ValueError("edge_indices must be a square variable-index matrix.")
        subsets = tuple(_indices(item, minimum=2) for item in self.subsets)
        _positive_weight(self.weight)
        object.__setattr__(self, "edge_indices", edges)
        object.__setattr__(self, "subsets", subsets)

    def evaluate(self, values: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(values.shape[:-1], device=values.device, dtype=values.dtype)
        edges = self.edge_indices.to(values.device)
        for subset in self.subsets:
            scope = subset.to(values.device)
            selected_edges = edges[scope[:, None], scope[None, :]].reshape(-1)
            violation = (values[..., selected_edges].sum(dim=-1) - (len(scope) - 1)).clamp_min(0)
            total = total + violation.square()
        return self.weight * total


__all__ = [
    "CumulativeResourceFactor",
    "FlowConservationFactor",
    "IndicatorFactor",
    "LogicalFactor",
    "MatchingFactor",
    "NoOverlapFactor",
    "PiecewiseLinearFactor",
    "PrecedenceFactor",
    "SOS1Factor",
    "SOS2Factor",
    "SubtourEliminationFactor",
]
