"""GNN architectures for :mod:`qqa.pignn`.

Mirrors the reference ``GCN_dev`` from CRA4CO_:

* a learnable per-node :class:`torch.nn.Embedding` provides the input
  features (no node attributes are assumed),
* two stacked :class:`torch_geometric.nn.GCNConv` layers with a ReLU
  in-between and dropout,
* a final :func:`torch.sigmoid` so the output ``p \\in (0, 1)^N`` is
  immediately compatible with the QUBO loss
  ``problem.loss_fn(p) = p^T Q p``.

.. _CRA4CO: https://github.com/Yuma-Ichikawa/CRA4CO
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qqa.pignn._import import require_pyg


def default_in_feats(num_nodes: int) -> int:
    """Heuristic used by the original CRA paper: ``floor(sqrt(N))``."""
    return max(2, int(math.floor(math.sqrt(num_nodes))))


class GCNNet(nn.Module):
    """Two-layer GCN with a learnable node embedding (PI-GNN style).

    Parameters
    ----------
    num_nodes:
        Number of nodes in the graph (also the embedding table size).
    in_feats:
        Width of the input embedding. Defaults to ``floor(sqrt(N))`` to
        match the reference implementation.
    hidden_dim:
        Hidden width between the two GCN layers. Defaults to ``in_feats``.
    dropout:
        Dropout probability applied after the first GCN layer. Defaults
        to 0 — the original paper used 0 for the headline MIS results.
    num_replicas:
        Number of independent output channels. Defaults to ``1``,
        which preserves the single-head CRA-PI-GNN behaviour and keeps
        the forward output shape ``(N,)``. With ``num_replicas >= 2``
        the network becomes the **CPRA** multi-head backbone of
        Ichikawa & Iwashita (TMLR 2025) — a single shared embedding +
        GCN backbone produces ``R`` parallel continuous solutions in
        one forward pass and the output shape is ``(N, R)``.

    Notes
    -----
    The forward pass takes only ``edge_index`` because the node
    "features" are the learned embedding rows; they evolve through the
    same backward pass as the convolution weights. This is the standard
    PI-GNN trick from Schuetz et al. (Nature MI 2022). For
    ``num_replicas >= 2`` only the second convolution's output channels
    grow — the embedding and first convolution are shared across
    replicas, matching the CPRA shared-representation design.
    """

    def __init__(
        self,
        num_nodes: int,
        in_feats: int | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        num_replicas: int = 1,
    ):
        super().__init__()
        require_pyg()
        from torch_geometric.nn import GCNConv

        if int(num_replicas) < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")

        self.num_nodes = int(num_nodes)
        self.in_feats = default_in_feats(num_nodes) if in_feats is None else int(in_feats)
        self.hidden_dim = self.in_feats if hidden_dim is None else int(hidden_dim)
        self.dropout = float(dropout)
        self.num_replicas = int(num_replicas)

        self.embedding = nn.Embedding(self.num_nodes, self.in_feats)
        self.conv1 = GCNConv(self.in_feats, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.num_replicas)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute soft node assignments ``p \\in (0, 1)``.

        Parameters
        ----------
        edge_index:
            ``(2, 2|E|)`` ``long`` tensor produced by
            :func:`qqa.pignn.graph.nx_to_edge_index`.

        Returns
        -------
        torch.Tensor
            ``(N,)`` tensor of probabilities when ``num_replicas == 1``
            (CRA-PI-GNN compatibility), or ``(N, num_replicas)`` when
            ``num_replicas >= 2`` (CPRA layout).
        """
        x = self.embedding.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.sigmoid(x)
        if self.num_replicas == 1:
            return x.squeeze(-1)
        return x
