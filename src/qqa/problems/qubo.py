"""Binary QUBO problems: MIS, MaxClique, MaxCut.

All classes compute ``loss = x^T Q x`` on the continuous relaxation
``x \\in [0, 1]^N`` supplied by :class:`~qqa.relaxation.BinaryRelaxation`
(or its batched variant). Minimising the loss is equivalent to solving the
corresponding combinatorial problem.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import torch

from qqa.problems.base import COProblem, QUBOProblem
from qqa.relaxation import BinaryInstanceRelaxation, BinaryRelaxation


class MaximumIndependentSet(QUBOProblem):
    """MIS as a QUBO: ``diag(-1)`` with ``penalty`` on each edge.

    The loss ``x^T Q x`` is ``-|S| + penalty * (#violated edges)``, so when
    all constraints are satisfied, ``-loss`` equals the independent-set size.
    """

    def __init__(
        self,
        nx_graph: nx.Graph,
        penalty: float = 3.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.nx_graph = nx_graph
        self.penalty = penalty
        self.device = device
        self.num_nodes = nx_graph.number_of_nodes()
        self.Q_mat = self.generate_qubo_matrix()
        self.relaxation = BinaryRelaxation()

    def generate_qubo_matrix(self) -> torch.Tensor:
        Q = torch.zeros((self.num_nodes, self.num_nodes))
        for u, v in self.nx_graph.edges:
            Q[u, v] = self.penalty
            Q[v, u] = self.penalty
        for u in self.nx_graph.nodes:
            Q[u, u] = -1.0
        return Q.to(self.device)

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,ij,bj->b", x, self.Q_mat, x)

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x = x_disc if x_disc.ndim == 2 else x_disc.unsqueeze(0)
        with torch.no_grad():
            xd = x.float()
            size = xd.sum(dim=-1)
            # Count violated edges directly from the graph to avoid double-counting.
            adj = self.Q_mat.clone()
            adj.fill_diagonal_(0.0)
            violations = 0.5 * torch.einsum("bi,ij,bj->b", xd, (adj > 0).float(), xd)
        feas = violations <= 0.5
        if feas.any():
            s = size.clone()
            s[~feas] = -float("inf")
            idx = int(torch.argmax(s).item())
            feasible = True
        else:
            idx = int(torch.argmax(size).item())
            feasible = False
        return {
            "label": "IS size",
            "value": int(size[idx].item()),
            "unit": f"/ {self.num_nodes}",
            "feasible": feasible,
            "extra": {"violated_edges": int(violations[idx].item())},
        }


class MaximumIndependentSetInstance(COProblem):
    """Batched-instance MIS. All graphs padded to ``max_node`` nodes."""

    def __init__(
        self,
        nx_graph_list: Sequence[nx.Graph],
        max_node: int,
        penalty: float = 3.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        Q_list = []
        for g in nx_graph_list:
            Q = torch.zeros((max_node, max_node))
            for u, v in g.edges:
                Q[u, v] = penalty
                Q[v, u] = penalty
            for u in g.nodes:
                Q[u, u] = -1.0
            Q_list.append(Q)
        self.Q_tensor = torch.stack(Q_list).to(device)
        self.num_instance = len(nx_graph_list)
        self.max_node = max_node
        self.num_nodes = max_node
        self.device = device
        self.relaxation = BinaryInstanceRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bci,cij,bcj->bc", x, self.Q_tensor, x)


class MaxClique(QUBOProblem):
    """Max clique as a QUBO: ``diag(-1)`` with ``penalty`` on non-edges."""

    def __init__(
        self,
        nx_graph: nx.Graph,
        penalty: float = 3.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.nx_graph = nx_graph
        self.penalty = penalty
        self.device = device
        self.num_nodes = nx_graph.number_of_nodes()
        self.Q_mat = self.generate_qubo_matrix()
        self.relaxation = BinaryRelaxation()

    def generate_qubo_matrix(self) -> torch.Tensor:
        Q = torch.full((self.num_nodes, self.num_nodes), float(self.penalty))
        for u, v in self.nx_graph.edges:
            Q[u, v] = 0.0
            Q[v, u] = 0.0
        for u in self.nx_graph.nodes:
            Q[u, u] = -1.0
        return Q.to(self.device)

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,ij,bj->b", x, self.Q_mat, x)

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x = x_disc if x_disc.ndim == 2 else x_disc.unsqueeze(0)
        with torch.no_grad():
            xd = x.float()
            size = xd.sum(dim=-1)
            # Missing edges inside the chosen set (i.e. non-clique pairs).
            non_edge = self.Q_mat.clone()
            non_edge.fill_diagonal_(0.0)
            violations = 0.5 * torch.einsum("bi,ij,bj->b", xd, (non_edge > 0).float(), xd)
        feas = violations <= 0.5
        if feas.any():
            s = size.clone()
            s[~feas] = -float("inf")
            idx = int(torch.argmax(s).item())
            feasible = True
        else:
            idx = int(torch.argmax(size).item())
            feasible = False
        return {
            "label": "clique size",
            "value": int(size[idx].item()),
            "unit": f"/ {self.num_nodes}",
            "feasible": feasible,
            "extra": {"missing_edges": int(violations[idx].item())},
        }


class MaxCliqueInstance(COProblem):
    """Batched-instance Max Clique."""

    def __init__(
        self,
        nx_graph_list: Sequence[nx.Graph],
        max_node: int,
        penalty: float = 3.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        Q_list = []
        for g in nx_graph_list:
            Q = torch.full((max_node, max_node), float(penalty))
            for u, v in g.edges:
                Q[u, v] = 0.0
                Q[v, u] = 0.0
            for u in g.nodes:
                Q[u, u] = -1.0
            Q_list.append(Q)
        self.Q_tensor = torch.stack(Q_list).to(device)
        self.num_instance = len(nx_graph_list)
        self.max_node = max_node
        self.num_nodes = max_node
        self.device = device
        self.relaxation = BinaryInstanceRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bci,cij,bcj->bc", x, self.Q_tensor, x)


class MaxCut(QUBOProblem):
    """Weighted Max-Cut QUBO (minimising ``x^T Q x``)."""

    def __init__(self, nx_graph: nx.Graph, device: str | torch.device = "cpu"):
        super().__init__()
        self.nx_graph = nx_graph
        self.device = device
        self.num_nodes = nx_graph.number_of_nodes()
        self.Q_mat = self.generate_qubo_matrix()
        self.relaxation = BinaryRelaxation()

    def generate_qubo_matrix(self) -> torch.Tensor:
        Q = torch.zeros((self.num_nodes, self.num_nodes))
        for u, v, data in self.nx_graph.edges(data=True):
            w = float(data.get("weight", 1.0))
            Q[u, v] = w
            Q[v, u] = w
        wsum = Q.sum(dim=1)
        for u in self.nx_graph.nodes:
            Q[u, u] = -wsum[u].item()
        return Q.to(self.device)

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,ij,bj->b", x, self.Q_mat, x)

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x = x_disc if x_disc.ndim == 2 else x_disc.unsqueeze(0)
        with torch.no_grad():
            xd = x.float()
            # Cut size = sum of edge weights w_{uv} [x_u != x_v].
            W = self.Q_mat.clone()
            W.fill_diagonal_(0.0)
            cut = 0.5 * torch.einsum("bi,ij,bj->b", xd, W, 1 - xd) + 0.5 * torch.einsum(
                "bi,ij,bj->b", 1 - xd, W, xd
            )
            # (two terms equal; using the average for symmetry)
        idx = int(torch.argmax(cut).item())
        return {
            "label": "cut size",
            "value": float(cut[idx].item()),
            "unit": "",
            "feasible": True,
            "extra": {},
        }


class MaxCutInstance(COProblem):
    """Batched-instance Max-Cut."""

    def __init__(
        self,
        nx_graph_list: Sequence[nx.Graph],
        max_node: int,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        Q_list = []
        for g in nx_graph_list:
            Q = torch.zeros((max_node, max_node))
            for u, v, data in g.edges(data=True):
                w = float(data.get("weight", 1.0))
                Q[u, v] = w
                Q[v, u] = w
            wsum = Q.sum(dim=1)
            for u in g.nodes:
                Q[u, u] = -wsum[u].item()
            Q_list.append(Q)
        self.Q_tensor = torch.stack(Q_list).to(device)
        self.num_instance = len(nx_graph_list)
        self.max_node = max_node
        self.num_nodes = max_node
        self.device = device
        self.relaxation = BinaryInstanceRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bci,cij,bcj->bc", x, self.Q_tensor, x)
