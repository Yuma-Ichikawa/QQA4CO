"""Structure-preserving decomposition detection for typed factor models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx
import torch

from qqa.model.ir import ModelIR


@dataclass(frozen=True, slots=True)
class DecompositionBlock:
    variables: tuple[int, ...]
    factor_locations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecompositionPlan:
    blocks: tuple[DecompositionBlock, ...]
    linking_variables: tuple[int, ...]
    method: str
    independently_solvable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scope(factor: Any) -> tuple[int, ...]:
    indices = []
    for name in (
        "indices",
        "edge_index",
        "starts",
        "before",
        "after",
        "edge_indices",
        "groups",
    ):
        value = getattr(factor, name, None)
        if torch.is_tensor(value):
            indices.extend(value.reshape(-1).tolist())
    return tuple(sorted({int(value) for value in indices}))


def detect_decomposition(
    model: ModelIR, *, maximum_linking_fraction: float = 0.05
) -> DecompositionPlan:
    """Find independent blocks or a small variable separator."""
    if not isinstance(model, ModelIR):
        raise TypeError("model must be a ModelIR.")
    if not 0 <= maximum_linking_fraction < 1:
        raise ValueError("maximum_linking_fraction must lie in [0, 1).")
    graph = nx.Graph()
    graph.add_nodes_from(range(model.num_variables))
    locations: dict[int, list[str]] = {index: [] for index in range(model.num_variables)}
    expressions = [("objective", model.objective)] + [
        (f"constraint:{row.name}", row.expression) for row in model.constraints
    ]
    for location, expression in expressions:
        for factor_index, factor in enumerate(expression.factors):
            scope = _scope(factor)
            label = f"{location}[{factor_index}]"
            for variable in scope:
                locations[variable].append(label)
            if len(scope) > 1:
                root = scope[0]
                graph.add_edges_from((root, variable) for variable in scope[1:])
    components = [tuple(sorted(component)) for component in nx.connected_components(graph)]
    if len(components) > 1:
        blocks = tuple(
            DecompositionBlock(
                component,
                tuple(sorted({item for variable in component for item in locations[variable]})),
            )
            for component in sorted(components, key=lambda item: (item[0], len(item)))
        )
        return DecompositionPlan(blocks, (), "connected-components", True)

    maximum_linking = max(1, int(model.num_variables * maximum_linking_fraction))
    centrality = nx.betweenness_centrality(graph, k=min(64, len(graph)), seed=0)
    order = sorted(graph, key=lambda node: (-centrality[node], node))
    for count in range(1, maximum_linking + 1):
        linking = tuple(order[:count])
        reduced = graph.copy()
        reduced.remove_nodes_from(linking)
        reduced_components = [
            tuple(sorted(component)) for component in nx.connected_components(reduced)
        ]
        if len(reduced_components) > 1:
            blocks = tuple(
                DecompositionBlock(
                    component,
                    tuple(sorted({item for variable in component for item in locations[variable]})),
                )
                for component in reduced_components
            )
            return DecompositionPlan(blocks, linking, "variable-separator", False)
    block = DecompositionBlock(
        tuple(range(model.num_variables)),
        tuple(sorted({item for values in locations.values() for item in values})),
    )
    return DecompositionPlan((block,), (), "monolithic", True)


__all__ = ["DecompositionBlock", "DecompositionPlan", "detect_decomposition"]
