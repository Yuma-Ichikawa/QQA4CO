"""Normalized Cut (Ncut) — DISCS-compatible K-way graph partitioning.

The objective follows the standard formulation popularised by Shi & Malik
(2000) and used as a benchmark target in the DISCS NeurIPS-2023 paper
(``discs/models/normcut.py``)::

    Ncut(x) = Σ_k  cut(V_k, V \\ V_k) / vol(V_k)

where ``cut(V_k, V \\ V_k) = Σ_{(u,v) ∈ E, x_u^k ≠ x_v^k} 1`` and
``vol(V_k) = Σ_{u ∈ V_k} deg(u)``.

We minimise ``Ncut`` directly. Lower is better; **an Ncut of exactly 0 is
a degenerate solution** that simply assigns each connected component to
one partition (cut = 0 by construction). Several DISCS ``nets/`` graphs
(BABELFISH, TTS, ALEXNET, VGG, …) have a giant component plus dozens of
2-4 node fragments, so a vanilla solver will always find Ncut = 0. If
you want non-trivial bisections of those graphs, restrict to the largest
connected component first::

    g = max(nx.connected_components(g_raw), key=len)
    g = g_raw.subgraph(g).copy()
    p = qqa.NormalizedCut(g, num_category=2)

A balanced bisection of a connected Erdős-Rényi graph typically lands
around ``Ncut ≈ K / 2``.
"""

from __future__ import annotations

import networkx as nx
import torch

from qqa.problems.base import COProblem, normalize_graph
from qqa.relaxation import CategoricalRelaxation


class NormalizedCut(COProblem):
    """K-way Normalized Cut on an undirected graph.

    Parameters
    ----------
    nx_graph
        Undirected ``networkx`` graph. Edge weights are ignored (Ncut is
        defined on unweighted edges; weighted Ncut is identical up to a
        re-scaling of ``deg``).
    num_category
        Number of partitions ``K`` (default 2).
    eps
        Numerical guard added to the volume denominator. The default of
        ``1e-3`` matches DISCS' choice well; very small ``eps`` lets the
        optimiser shrink one partition to a single high-degree node and
        win a meaningless ``0/eps`` for that partition.
    device
        Torch device for the dense adjacency / degree tensors.
    """

    def __init__(
        self,
        nx_graph: nx.Graph,
        num_category: int = 2,
        device: str | torch.device = "cpu",
        eps: float = 1e-3,
    ):
        super().__init__()
        if num_category < 2:
            raise ValueError(f"num_category must be >= 2, got {num_category}")

        nx_graph = normalize_graph(nx_graph)
        n_nodes = nx_graph.number_of_nodes()
        n_edges = nx_graph.number_of_edges()
        if n_nodes < num_category:
            raise ValueError(
                f"NormalizedCut needs >= {num_category} nodes (got {n_nodes}); "
                "an empty or near-empty graph cannot be partitioned."
            )
        if n_edges == 0:
            raise ValueError(
                "NormalizedCut requires a graph with at least one edge "
                "(got 0). Verify the source pickle is not corrupted."
            )

        self.nx_graph = nx_graph
        self.num_node = n_nodes
        self.num_nodes = n_nodes
        self.num_edge = n_edges
        self.num_category = num_category
        self.eps = float(eps)
        self.device = device

        adj = nx.adjacency_matrix(nx_graph).toarray()
        self.adj = torch.tensor(adj, device=device, dtype=torch.float32)
        self.degrees = self.adj.sum(dim=1)
        self.relaxation = CategoricalRelaxation()

    def _cut_per_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-partition cut sizes ``(B, K)``.

        Vectorised version of ``Σ_{(u,v) ∈ E} (x_u^k − x_v^k)^2 / 2`` which
        equals the integer cut ``|{(u,v): x_u^k ≠ x_v^k}|`` on one-hot
        inputs and is a smooth surrogate on the simplex.
        """
        # x: (B, N, K), adj: (N, N)
        # diff[b, u, v, k] = x[b, u, k] - x[b, v, k]
        # => cut[b, k] = 1/2 * Σ_{u, v} adj[u, v] * (x[b, u, k] - x[b, v, k])^2
        # Using the identity (a - b)^2 = a^2 + b^2 - 2 a b and Σ adj[u,v]=2|E|:
        x2 = x * x  # (B, N, K)
        deg_x2 = torch.einsum("u,buk->bk", self.degrees, x2)
        xax = torch.einsum("buk,uv,bvk->bk", x, self.adj, x)
        return deg_x2 - xax

    def _volume_per_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Per-partition volume ``Σ_{u} deg(u) * x[u,k]`` -> ``(B, K)``."""
        return torch.einsum("u,buk->bk", self.degrees, x)

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth Ncut on simplex-valued ``x`` of shape ``(B, N, K)``."""
        cut = self._cut_per_partition(x)
        vol = self._volume_per_partition(x)
        return torch.sum(cut / (vol + self.eps), dim=1)

    @torch.no_grad()
    def discrete_ncut(self, x_disc: torch.Tensor) -> torch.Tensor:
        """Exact integer Ncut on a one-hot ``x`` (no eps in denominator)."""
        cut = self._cut_per_partition(x_disc.float())
        vol = self._volume_per_partition(x_disc.float())
        # When a partition is empty (vol=0) Ncut is conventionally infinite;
        # we replace by a large finite value so torch.argmin still works.
        vol = torch.where(vol > 0, vol, torch.full_like(vol, 1.0))
        cut = torch.where(vol > 0, cut, torch.full_like(cut, float(self.num_edge)))
        return torch.sum(cut / vol, dim=1)

    @torch.no_grad()
    def cut_size(self, x_disc: torch.Tensor) -> torch.Tensor:
        """Total integer edge cut ``Σ_k cut_k / 2`` (each edge counted once)."""
        cut = self._cut_per_partition(x_disc.float())
        return cut.sum(dim=1) / 2

    @torch.no_grad()
    def balanceness(self, x_disc: torch.Tensor) -> torch.Tensor:
        """Imbalance score in ``[0, 1]``: ``min_k |V_k| / (N/K)``."""
        sizes = x_disc.sum(dim=1).float()  # (B, K)
        target = self.num_node / self.num_category
        return sizes.min(dim=1).values / target

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x = x_disc if x_disc.ndim == 3 else x_disc.unsqueeze(0)
        with torch.no_grad():
            ncut = self.discrete_ncut(x)
            edge_cut = self.cut_size(x)
            balance = self.balanceness(x)
        idx = int(torch.argmin(ncut).item())
        return {
            "label": "Ncut",
            "value": float(ncut[idx].item()),
            "unit": f"(K={self.num_category})",
            "feasible": bool(balance[idx].item() > 0.0),
            "extra": {
                "edge_cut": int(edge_cut[idx].item()),
                "num_edges": self.num_edge,
                "min_partition_ratio": float(balance[idx].item()),
            },
        }


# Convenient alias matching DISCS' class name verbatim — useful when porting
# scripts that say ``from qqa import NormCut``.
NormCut = NormalizedCut


__all__ = ["NormalizedCut", "NormCut"]
