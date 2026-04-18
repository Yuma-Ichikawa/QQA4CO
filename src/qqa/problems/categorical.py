"""Categorical (one-hot) problems: balanced graph partitioning and coloring."""

from __future__ import annotations

import networkx as nx
import torch

from qqa.problems.base import COProblem
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
        self.nx_graph = nx_graph
        self.adj = torch.tensor(
            nx.adjacency_matrix(nx_graph).toarray(), device=device, dtype=torch.float32
        )
        self.num_node = nx_graph.number_of_nodes()
        self.num_nodes = self.num_node
        self.num_edge = nx_graph.number_of_edges()
        self.num_category = num_category
        self.penalty = penalty
        self.device = device
        self.relaxation = CategoricalRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        edge_cut = self.num_edge - torch.sum(
            torch.einsum("bis,ij,bjs->bs", x, self.adj, x) / 2, dim=1
        )
        bal = torch.sum((self.num_node / self.num_category - torch.sum(x, dim=1)) ** 2, dim=1)
        return edge_cut + bal * self.penalty

    def cut_ratio(self, x: torch.Tensor) -> torch.Tensor:
        """Edge-cut ratio ``(|E| - intra-class edges) / |E|``."""
        return (
            self.num_edge - torch.sum(torch.einsum("bis,ij,bjs->bs", x, self.adj, x) / 2, dim=1)
        ) / self.num_edge

    def balanceness(self, x: torch.Tensor) -> torch.Tensor:
        """Balance score in ``[0, 1]`` (higher is better)."""
        return 1 - torch.mean(
            (1 - torch.sum(x, dim=1) / (self.num_node / self.num_category)) ** 2, dim=1
        )

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x = x_disc if x_disc.ndim == 3 else x_disc.unsqueeze(0)
        with torch.no_grad():
            xd = x.float()
            cut = self.num_edge - torch.sum(
                torch.einsum("bis,ij,bjs->bs", xd, self.adj, xd) / 2, dim=1
            )
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
        self.nx_graph = nx_graph
        self.adj = torch.tensor(
            nx.adjacency_matrix(nx_graph).toarray(), device=device, dtype=torch.float32
        )
        self.num_node = nx_graph.number_of_nodes()
        self.num_nodes = self.num_node
        self.num_edge = nx_graph.number_of_edges()
        self.num_category = num_category
        self.device = device
        self.relaxation = CategoricalRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.einsum("bis,ij,bjs->bs", x, self.adj, x) / 2, dim=1)

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
