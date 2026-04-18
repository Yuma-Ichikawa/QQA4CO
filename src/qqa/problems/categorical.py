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
