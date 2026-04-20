"""Binary QUBO problems: MIS, MaxClique, MaxCut.

All classes compute ``loss = x^T Q x`` on the continuous relaxation
``x \\in [0, 1]^N`` supplied by :class:`~qqa.relaxation.BinaryRelaxation`
(or its batched variant). Minimising the loss is equivalent to solving the
corresponding combinatorial problem.

The ``*Instance`` variants pack a *list* of graphs of (possibly different)
sizes into a single ``(num_instance, max_node, max_node)`` Q tensor so the
solver can attack all of them in **one** ``qqa.anneal`` call. Each instance
carries a ``pad_mask`` that the loss / score_summary multiplies in to keep
padded positions semantically inert — the optimiser may put anything in
``x[i, n_i:]`` because the mask zeroes its contribution to both the loss
and the reported objective.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import torch

from qqa.problems.base import COProblem, QUBOProblem, normalize_graph
from qqa.relaxation import BinaryInstanceRelaxation, BinaryRelaxation


def _build_pad_mask(
    n_per_instance: Sequence[int], max_node: int, *, device, dtype=torch.float32
) -> torch.Tensor:
    """``(I, max_node)`` mask: ``1`` for real vars, ``0`` for padding.

    A floating mask is more cache-friendly for downstream einsums than a
    boolean mask; callers can ``.bool()`` it if they need indexing.
    """
    mask = torch.zeros((len(n_per_instance), max_node), dtype=dtype, device=device)
    for i, n in enumerate(n_per_instance):
        mask[i, :n] = 1.0
    return mask


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
        self.nx_graph = normalize_graph(nx_graph)
        self.penalty = penalty
        self.device = device
        self.num_nodes = self.nx_graph.number_of_nodes()
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
    """Batched-instance MIS, padded to ``max_node`` and *masked* in the loss.

    Parameters
    ----------
    nx_graph_list
        Heterogeneous list of NetworkX graphs (any sizes ``n_i <= max_node``).
    max_node
        Padding width. If ``None`` (recommended), uses
        ``max(g.number_of_nodes() for g in nx_graph_list)``.
    penalty
        Edge-violation penalty. Either a scalar (broadcast to all instances)
        or a per-instance sequence of length ``I``.
    device
        Torch device for the dense ``Q_tensor`` and ``pad_mask``.

    Loss
    ----
    ``loss[b, i] = (m_i ⊙ x_{b,i})^T Q_i (m_i ⊙ x_{b,i})``
    where ``m_i`` is the per-instance pad mask. The mask makes padded
    positions strictly inert: anything the optimiser writes into
    ``x[:, i, n_i:]`` is squashed to zero before the einsum.
    """

    def __init__(
        self,
        nx_graph_list: Sequence[nx.Graph],
        max_node: int | None = None,
        penalty: float | Sequence[float] = 3.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        nx_graph_list = [normalize_graph(g) for g in nx_graph_list]
        n_per = [g.number_of_nodes() for g in nx_graph_list]
        if max_node is None:
            max_node = max(n_per)
        if max_node < max(n_per):
            raise ValueError(
                f"max_node={max_node} smaller than the largest instance ({max(n_per)} nodes)."
            )
        if isinstance(penalty, (int, float)):
            penalties = [float(penalty)] * len(nx_graph_list)
        else:
            penalties = [float(p) for p in penalty]
            if len(penalties) != len(nx_graph_list):
                raise ValueError("len(penalty) must equal len(nx_graph_list).")
        Q_list = []
        for g, p in zip(nx_graph_list, penalties, strict=True):
            Q = torch.zeros((max_node, max_node))
            for u, v in g.edges:
                Q[u, v] = p
                Q[v, u] = p
            for u in g.nodes:
                Q[u, u] = -1.0
            Q_list.append(Q)
        self.nx_graph_list = nx_graph_list
        self.penalties = penalties
        self.Q_tensor = torch.stack(Q_list).to(device)
        self.num_instance = len(nx_graph_list)
        self.max_node = max_node
        self.num_nodes = max_node
        self.n_per_instance = torch.as_tensor(n_per, dtype=torch.long, device=device)
        self.pad_mask = _build_pad_mask(n_per, max_node, device=device)
        self.device = device
        self.relaxation = BinaryInstanceRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        # Zero out padded positions before the einsum so only real vars
        # contribute. The mask broadcasts (1, I, N) over the (B, I, N) batch.
        x_masked = x * self.pad_mask
        return torch.einsum("bci,cij,bcj->bc", x_masked, self.Q_tensor, x_masked)

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        """Per-instance IS sizes & feasibility for ``best_sol`` of shape ``(I, N)``.

        ``value`` and ``feasible`` are 1-D ``np.ndarray`` of length
        ``num_instance``; ``extra`` carries plain-Python lists / ints so
        the whole dict is ``json.dumps``-able (the bench runner relies on this).
        """
        x = x_disc if x_disc.ndim == 2 else x_disc
        with torch.no_grad():
            xd = (x.float() * self.pad_mask).round()  # (I, N)
            sizes = xd.sum(dim=-1).long()  # (I,)
            adj = (self.Q_tensor > 0).float()  # (I, N, N) — penalty marks edges
            violations = 0.5 * torch.einsum("ci,cij,cj->c", xd, adj, xd).long()
        feasible = violations == 0
        sizes_np = sizes.cpu().numpy()
        feas_np = feasible.cpu().numpy()
        viol_np = violations.cpu().numpy()
        return {
            "label": "IS size",
            "value": sizes_np,
            "unit": "per instance",
            "feasible": feas_np,
            "extra": {
                "violated_edges": viol_np.tolist(),
                "n_per_instance": self.n_per_instance.cpu().numpy().tolist(),
                "feasible_count": int(feas_np.sum()),
                "num_instance": int(self.num_instance),
            },
        }


class MaxClique(QUBOProblem):
    """Max clique as a QUBO: ``diag(-1)`` with ``penalty`` on non-edges."""

    def __init__(
        self,
        nx_graph: nx.Graph,
        penalty: float = 3.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.nx_graph = normalize_graph(nx_graph)
        self.penalty = penalty
        self.device = device
        self.num_nodes = self.nx_graph.number_of_nodes()
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
    """Batched-instance Max Clique with per-instance pad mask."""

    def __init__(
        self,
        nx_graph_list: Sequence[nx.Graph],
        max_node: int | None = None,
        penalty: float | Sequence[float] = 3.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        nx_graph_list = [normalize_graph(g) for g in nx_graph_list]
        n_per = [g.number_of_nodes() for g in nx_graph_list]
        if max_node is None:
            max_node = max(n_per)
        if max_node < max(n_per):
            raise ValueError(
                f"max_node={max_node} smaller than the largest instance ({max(n_per)} nodes)."
            )
        if isinstance(penalty, (int, float)):
            penalties = [float(penalty)] * len(nx_graph_list)
        else:
            penalties = [float(p) for p in penalty]
            if len(penalties) != len(nx_graph_list):
                raise ValueError("len(penalty) must equal len(nx_graph_list).")
        Q_list = []
        non_edge_list = []
        for g, p in zip(nx_graph_list, penalties, strict=True):
            Q = torch.full((max_node, max_node), float(p))
            for u, v in g.edges:
                Q[u, v] = 0.0
                Q[v, u] = 0.0
            for u in g.nodes:
                Q[u, u] = -1.0
            non_edge = (Q > 0).float()
            non_edge.fill_diagonal_(0.0)
            Q_list.append(Q)
            non_edge_list.append(non_edge)
        self.nx_graph_list = nx_graph_list
        self.penalties = penalties
        self.Q_tensor = torch.stack(Q_list).to(device)
        self._non_edge_tensor = torch.stack(non_edge_list).to(device)
        self.num_instance = len(nx_graph_list)
        self.max_node = max_node
        self.num_nodes = max_node
        self.n_per_instance = torch.as_tensor(n_per, dtype=torch.long, device=device)
        self.pad_mask = _build_pad_mask(n_per, max_node, device=device)
        self.device = device
        self.relaxation = BinaryInstanceRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        x_masked = x * self.pad_mask
        return torch.einsum("bci,cij,bcj->bc", x_masked, self.Q_tensor, x_masked)

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x = x_disc if x_disc.ndim == 2 else x_disc
        with torch.no_grad():
            xd = (x.float() * self.pad_mask).round()  # (I, N)
            sizes = xd.sum(dim=-1).long()
            # Missing edges inside the chosen set (clique violations).
            violations = 0.5 * torch.einsum("ci,cij,cj->c", xd, self._non_edge_tensor, xd).long()
        feasible = violations == 0
        return {
            "label": "clique size",
            "value": sizes.cpu().numpy(),
            "unit": "per instance",
            "feasible": feasible.cpu().numpy(),
            "extra": {
                "missing_edges": violations.cpu().numpy().tolist(),
                "n_per_instance": self.n_per_instance.cpu().numpy().tolist(),
                "feasible_count": int(feasible.sum().item()),
                "num_instance": int(self.num_instance),
            },
        }


class MaxCut(QUBOProblem):
    """Weighted Max-Cut QUBO (minimising ``x^T Q x``)."""

    def __init__(self, nx_graph: nx.Graph, device: str | torch.device = "cpu"):
        super().__init__()
        self.nx_graph = normalize_graph(nx_graph)
        self.device = device
        self.num_nodes = self.nx_graph.number_of_nodes()
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
    """Batched-instance Max-Cut with per-instance pad mask."""

    def __init__(
        self,
        nx_graph_list: Sequence[nx.Graph],
        max_node: int | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        nx_graph_list = [normalize_graph(g) for g in nx_graph_list]
        n_per = [g.number_of_nodes() for g in nx_graph_list]
        if max_node is None:
            max_node = max(n_per)
        if max_node < max(n_per):
            raise ValueError(
                f"max_node={max_node} smaller than the largest instance ({max(n_per)} nodes)."
            )
        Q_list = []
        W_list = []
        for g in nx_graph_list:
            Q = torch.zeros((max_node, max_node))
            for u, v, data in g.edges(data=True):
                w = float(data.get("weight", 1.0))
                Q[u, v] = w
                Q[v, u] = w
            W = Q.clone()
            wsum = Q.sum(dim=1)
            for u in g.nodes:
                Q[u, u] = -wsum[u].item()
            Q_list.append(Q)
            W_list.append(W)
        self.nx_graph_list = nx_graph_list
        self.Q_tensor = torch.stack(Q_list).to(device)
        self._W_tensor = torch.stack(W_list).to(device)
        self.num_instance = len(nx_graph_list)
        self.max_node = max_node
        self.num_nodes = max_node
        self.n_per_instance = torch.as_tensor(n_per, dtype=torch.long, device=device)
        self.pad_mask = _build_pad_mask(n_per, max_node, device=device)
        self.device = device
        self.relaxation = BinaryInstanceRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        x_masked = x * self.pad_mask
        return torch.einsum("bci,cij,bcj->bc", x_masked, self.Q_tensor, x_masked)

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x = x_disc if x_disc.ndim == 2 else x_disc
        with torch.no_grad():
            xd = (x.float() * self.pad_mask).round()  # (I, N)
            one_minus = (1.0 - xd) * self.pad_mask
            cut = torch.einsum("ci,cij,cj->c", xd, self._W_tensor, one_minus)
        return {
            "label": "cut size",
            "value": cut.cpu().numpy(),
            "unit": "per instance",
            "feasible": np.ones(self.num_instance, dtype=bool),
            "extra": {
                "n_per_instance": self.n_per_instance.cpu().numpy().tolist(),
                "feasible_count": int(self.num_instance),
                "num_instance": int(self.num_instance),
            },
        }
