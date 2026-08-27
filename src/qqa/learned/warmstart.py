"""Small bipartite message-passing warm start trained on one ModelIR instance."""

from __future__ import annotations

import torch
from torch import nn

from qqa.learned.factor_graph import FactorGraphData, model_to_factor_graph
from qqa.model import ModelIR, VariableDomain


class FactorGraphWarmStart(nn.Module):
    """Portable factor-graph network with no torch-geometric dependency."""

    def __init__(
        self,
        variable_features: int = 8,
        factor_features: int = 4,
        hidden_size: int = 64,
        layers: int = 3,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or layers < 1:
            raise ValueError("hidden_size and layers must be positive.")
        self.variable_encoder = nn.Linear(variable_features, hidden_size)
        self.factor_encoder = nn.Linear(factor_features, hidden_size)
        self.factor_updates = nn.ModuleList(
            nn.Linear(2 * hidden_size, hidden_size) for _ in range(layers)
        )
        self.variable_updates = nn.ModuleList(
            nn.Linear(2 * hidden_size, hidden_size) for _ in range(layers)
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, graph: FactorGraphData) -> torch.Tensor:
        variable = torch.relu(self.variable_encoder(graph.variable_features))
        if not len(graph.factor_features):
            return self.output(variable).squeeze(-1)
        factor = torch.relu(self.factor_encoder(graph.factor_features))
        variable_index, factor_index = graph.incidence_edge_index
        for factor_update, variable_update in zip(
            self.factor_updates, self.variable_updates, strict=True
        ):
            factor_message = torch.zeros_like(factor)
            factor_message.index_add_(0, factor_index, variable[variable_index])
            factor_degree = torch.bincount(factor_index, minlength=len(factor)).clamp_min(1)
            factor_message /= factor_degree.unsqueeze(-1)
            factor = torch.relu(factor_update(torch.cat((factor, factor_message), dim=-1)))

            variable_message = torch.zeros_like(variable)
            variable_message.index_add_(0, variable_index, factor[factor_index])
            variable_degree = torch.bincount(variable_index, minlength=len(variable)).clamp_min(1)
            variable_message /= variable_degree.unsqueeze(-1)
            variable = torch.relu(variable_update(torch.cat((variable, variable_message), dim=-1)))
        return self.output(variable).squeeze(-1)


def _relaxed_values(model: ModelIR, probabilities: torch.Tensor) -> torch.Tensor:
    values: list[torch.Tensor] = []
    offset = 0
    for block in model.variables:
        selected = probabilities[offset : offset + block.size]
        if block.domain is VariableDomain.BINARY:
            values.append(selected)
        elif block.domain is VariableDomain.SPIN:
            values.append(2 * selected - 1)
        else:
            lower = (
                torch.full_like(selected, -1.0)
                if block.lower is None
                else torch.as_tensor(block.lower, device=selected.device)
                .reshape(-1)
                .expand(block.size)
            )
            upper = (
                torch.full_like(selected, 1.0)
                if block.upper is None
                else torch.as_tensor(block.upper, device=selected.device)
                .reshape(-1)
                .expand(block.size)
            )
            lower = torch.nan_to_num(lower.to(selected), neginf=-1.0, posinf=1.0)
            upper = torch.nan_to_num(upper.to(selected), neginf=-1.0, posinf=1.0)
            values.append(lower + (upper - lower) * selected)
        offset += block.size
    return torch.cat(values)


def _discretize(model: ModelIR, values: torch.Tensor) -> torch.Tensor:
    result = values.clone()
    offset = 0
    for block in model.variables:
        selected = result[offset : offset + block.size]
        if block.domain is VariableDomain.BINARY:
            selected.copy_((selected >= 0.5).to(selected))
        elif block.domain is VariableDomain.SPIN:
            selected.copy_(torch.where(selected >= 0, 1.0, -1.0))
        elif block.domain is VariableDomain.INTEGER:
            selected.round_()
            if block.lower is not None:
                selected.copy_(
                    torch.maximum(
                        selected,
                        torch.as_tensor(block.lower, device=selected.device).reshape(-1),
                    )
                )
            if block.upper is not None:
                selected.copy_(
                    torch.minimum(
                        selected,
                        torch.as_tensor(block.upper, device=selected.device).reshape(-1),
                    )
                )
        offset += block.size
    return result


def factor_graph_warm_start(
    model: ModelIR,
    *,
    steps: int = 100,
    learning_rate: float = 1e-2,
    penalty: float = 10.0,
    hidden_size: int = 64,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Train an instance-specific GNN and return a detached discrete warm start."""
    if model.structured_block is not None:
        raise ValueError("Structured categorical blocks should use their native repair warm start.")
    if steps < 0 or learning_rate <= 0 or penalty < 0:
        raise ValueError(
            "steps must be non-negative; learning_rate positive; penalty non-negative."
        )
    graph = model_to_factor_graph(model).to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = FactorGraphWarmStart(
            graph.variable_features.shape[1],
            graph.factor_features.shape[1],
            hidden_size,
        ).to(device)
    optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
    for _ in range(steps):
        probabilities = torch.sigmoid(network(graph))
        values = _relaxed_values(model, probabilities)
        loss = model.internal_energy(values).mean()
        for row in model.constraints:
            loss = loss + penalty * row.weight * (row.violation(values) / row.scale).square().mean()
        loss = loss + 1e-3 * (probabilities * (1 - probabilities)).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        values = _relaxed_values(model, torch.sigmoid(network(graph)))
        return _discretize(model, values).detach()


__all__ = ["FactorGraphWarmStart", "factor_graph_warm_start"]
