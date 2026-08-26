"""Dependency-free ModelIR to bipartite factor-graph conversion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qqa.model import ModelIR, VariableDomain


@dataclass(frozen=True, slots=True)
class FactorGraphData:
    variable_features: torch.Tensor
    factor_features: torch.Tensor
    incidence_edge_index: torch.Tensor
    factor_kinds: tuple[str, ...]

    def to(self, device: str | torch.device) -> FactorGraphData:
        return FactorGraphData(
            self.variable_features.to(device),
            self.factor_features.to(device),
            self.incidence_edge_index.to(device),
            self.factor_kinds,
        )


def _scope(factor, num_variables: int) -> torch.Tensor:
    edge_index = getattr(factor, "edge_index", None)
    if torch.is_tensor(edge_index):
        return torch.unique(edge_index.detach().reshape(-1).to(torch.long))
    indices = getattr(factor, "indices", None)
    if torch.is_tensor(indices):
        return torch.unique(indices.detach().reshape(-1).to(torch.long))
    for name in ("index", "variable_index", "start_index"):
        index = getattr(factor, name, None)
        if isinstance(index, int):
            return torch.tensor([index], dtype=torch.long)
    return torch.arange(num_variables, dtype=torch.long)


def _scale(factor) -> float:
    values: list[float] = []
    for name in ("weights", "table", "outputs"):
        tensor = getattr(factor, name, None)
        if torch.is_tensor(tensor) and tensor.numel():
            values.append(float(tensor.detach().abs().max().item()))
    for name in ("weight", "capacity", "target"):
        value = getattr(factor, name, None)
        if isinstance(value, (float, int)):
            values.append(abs(float(value)))
    return max(values, default=1.0)


def model_to_factor_graph(model: ModelIR) -> FactorGraphData:
    """Convert factors and constraints to a compact bipartite graph."""
    if not isinstance(model, ModelIR):
        raise TypeError("model must be a ModelIR.")
    domain_order = tuple(VariableDomain)
    features: list[list[float]] = []
    for block in model.variables:
        lower = (
            torch.full((block.size,), -1.0)
            if block.lower is None
            else torch.as_tensor(block.lower).reshape(-1).expand(block.size)
        )
        upper = (
            torch.full((block.size,), 1.0)
            if block.upper is None
            else torch.as_tensor(block.upper).reshape(-1).expand(block.size)
        )
        lower = torch.nan_to_num(lower.to(torch.float32), neginf=-1.0, posinf=1.0)
        upper = torch.nan_to_num(upper.to(torch.float32), neginf=-1.0, posinf=1.0)
        for index in range(block.size):
            one_hot = [float(block.domain is domain) for domain in domain_order]
            features.append([*one_hot, float(lower[index]), float(upper[index])])
    variable_features = torch.tensor(features, dtype=torch.float32)

    factor_records: list[tuple[object, bool, float, float]] = [
        (factor, False, 1.0, 1.0) for factor in model.objective.factors
    ]
    for row in model.constraints:
        factor_records.extend(
            (factor, True, float(row.scale), float(row.weight)) for factor in row.expression.factors
        )
    factor_features: list[list[float]] = []
    kinds: list[str] = []
    variable_nodes: list[torch.Tensor] = []
    factor_nodes: list[torch.Tensor] = []
    for factor_index, (factor, constrained, scale, weight) in enumerate(factor_records):
        scope = _scope(factor, model.num_variables)
        if bool((scope < 0).any()) or bool((scope >= model.num_variables).any()):
            raise ValueError(f"{type(factor).__name__} references an out-of-range variable.")
        factor_features.append(
            [
                float(constrained),
                float(len(scope)) / model.num_variables,
                float(torch.log1p(torch.tensor(_scale(factor))).item()),
                float(torch.log1p(torch.tensor(abs(scale * weight))).item()),
            ]
        )
        kinds.append(type(factor).__name__)
        variable_nodes.append(scope)
        factor_nodes.append(torch.full((len(scope),), factor_index, dtype=torch.long))
    if factor_features:
        factor_tensor = torch.tensor(factor_features, dtype=torch.float32)
        incidence = torch.stack((torch.cat(variable_nodes), torch.cat(factor_nodes)))
    else:
        factor_tensor = torch.empty((0, 4), dtype=torch.float32)
        incidence = torch.empty((2, 0), dtype=torch.long)
    return FactorGraphData(variable_features, factor_tensor, incidence, tuple(kinds))


__all__ = ["FactorGraphData", "model_to_factor_graph"]
