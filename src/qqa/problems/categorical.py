"""Categorical (one-hot) problems: balanced graph partitioning and coloring."""

from __future__ import annotations

import networkx as nx
import torch

from qqa.problems.base import COProblem, normalize_graph
from qqa.relaxation import CategoricalRelaxation


class BalancedGraphPartition(COProblem):
    """Balanced K-partitioning of a graph.

    Minimises the edge cut plus a soft balance penalty (so each partition
    contains roughly ``N/K`` nodes).
    """

    def __init__(
        self,
        nx_graph: nx.Graph,
        num_category: int = 3,
        device: str | torch.device = "cpu",
        penalty: float = 5e-4,
    ):
        super().__init__()
        nx_graph = normalize_graph(nx_graph)
        self.nx_graph = nx_graph
        self.num_node = nx_graph.number_of_nodes()
        self.num_nodes = self.num_node
        self.num_edge = nx_graph.number_of_edges()
        self.num_category = num_category
        self.penalty = penalty
        self.device = device
        edges = torch.as_tensor(list(nx_graph.edges()), dtype=torch.long, device=device).reshape(
            -1, 2
        )
        self.edge_u = edges[:, 0]
        self.edge_v = edges[:, 1]
        self._adj: torch.Tensor | None = None
        self.relaxation = CategoricalRelaxation()

    @property
    def adj(self) -> torch.Tensor:
        if self._adj is None:
            self._adj = torch.zeros(
                (self.num_node, self.num_node), device=self.device, dtype=torch.float32
            )
            self._adj[self.edge_u, self.edge_v] = 1.0
            self._adj[self.edge_v, self.edge_u] = 1.0
        return self._adj

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        intra = (x[:, self.edge_u] * x[:, self.edge_v]).sum(dim=(1, 2))
        edge_cut = self.num_edge - intra
        bal = torch.sum((self.num_node / self.num_category - torch.sum(x, dim=1)) ** 2, dim=1)
        return edge_cut + bal * self.penalty

    def cut_ratio(self, x: torch.Tensor) -> torch.Tensor:
        """Edge-cut ratio ``(|E| - intra-class edges) / |E|``."""
        intra = (x[:, self.edge_u] * x[:, self.edge_v]).sum(dim=(1, 2))
        return (self.num_edge - intra) / self.num_edge

    def balanceness(self, x: torch.Tensor) -> torch.Tensor:
        """Balance score in ``[0, 1]`` (higher is better)."""
        return 1 - torch.mean(
            (1 - torch.sum(x, dim=1) / (self.num_node / self.num_category)) ** 2, dim=1
        )

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x = x_disc if x_disc.ndim == 3 else x_disc.unsqueeze(0)
        with torch.no_grad():
            xd = x.float()
            cut = self.num_edge - (xd[:, self.edge_u] * xd[:, self.edge_v]).sum(dim=(1, 2))
            sizes = xd.sum(dim=1)  # (B, K)
            target = self.num_node / self.num_category
            imbalance = (sizes - target).abs().max(dim=1).values
        idx = int(torch.argmin(cut).item())
        return {
            "label": "edge cut",
            "value": int(cut[idx].item()),
            "unit": f"/ {self.num_edge}",
            "feasible": bool(imbalance[idx].item() <= 1),
            "extra": {"max_imbalance": float(imbalance[idx].item())},
        }


class Coloring(COProblem):
    """K-coloring: counts same-colour adjacent pairs (``0`` iff proper)."""

    def __init__(
        self,
        nx_graph: nx.Graph,
        num_category: int = 3,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        nx_graph = normalize_graph(nx_graph)
        self.nx_graph = nx_graph
        self.num_node = nx_graph.number_of_nodes()
        self.num_nodes = self.num_node
        self.num_edge = nx_graph.number_of_edges()
        self.num_category = num_category
        self.device = device
        edges = torch.as_tensor(list(nx_graph.edges()), dtype=torch.long, device=device).reshape(
            -1, 2
        )
        self.edge_u = edges[:, 0]
        self.edge_v = edges[:, 1]
        self._adj: torch.Tensor | None = None
        self.relaxation = CategoricalRelaxation()

    @property
    def adj(self) -> torch.Tensor:
        if self._adj is None:
            self._adj = torch.zeros(
                (self.num_node, self.num_node), device=self.device, dtype=torch.float32
            )
            self._adj[self.edge_u, self.edge_v] = 1.0
            self._adj[self.edge_v, self.edge_u] = 1.0
        return self._adj

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, self.edge_u] * x[:, self.edge_v]).sum(dim=(1, 2))

    @torch.no_grad()
    def repair_solution(self, x_disc: torch.Tensor) -> torch.Tensor:
        """Min-conflicts recolouring with deterministic degree tie-breaking."""
        unbatched = x_disc.ndim == 2
        values = x_disc.unsqueeze(0) if unbatched else x_disc
        repaired = torch.zeros_like(values)
        adjacency = [set() for _ in range(self.num_node)]
        for left, right in zip(self.edge_u.tolist(), self.edge_v.tolist(), strict=True):
            adjacency[left].add(right)
            adjacency[right].add(left)
        for batch in range(values.shape[0]):
            colours = torch.argmax(values[batch], dim=1).tolist()
            for _ in range(max(1, 4 * self.num_node)):
                conflicts = [
                    sum(colours[vertex] == colours[other] for other in adjacency[vertex])
                    for vertex in range(self.num_node)
                ]
                worst = max(
                    range(self.num_node),
                    key=lambda vertex: (conflicts[vertex], len(adjacency[vertex])),
                )
                if conflicts[worst] == 0:
                    break
                counts = [
                    sum(colour == colours[other] for other in adjacency[worst])
                    for colour in range(self.num_category)
                ]
                colours[worst] = min(
                    range(self.num_category), key=lambda colour: (counts[colour], colour)
                )
            repaired[
                batch, torch.arange(self.num_node), torch.as_tensor(colours, device=values.device)
            ] = 1.0
        return repaired[0] if unbatched else repaired

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x = x_disc if x_disc.ndim == 3 else x_disc.unsqueeze(0)
        with torch.no_grad():
            conflicts = self.loss_fn(x.float())
            used_colors = (x.sum(dim=1) > 0).sum(dim=1)
        idx = int(torch.argmin(conflicts).item())
        conf = int(conflicts[idx].item())
        return {
            "label": "conflicts",
            "value": conf,
            "unit": "",
            "feasible": conf == 0,
            "extra": {"colors_used": int(used_colors[idx].item()), "K": self.num_category},
        }
