"""Pure-PyTorch sparse factor QQA problem adapter."""

from __future__ import annotations

import torch

from qqa.compile import SparseQUBO
from qqa.problems.base import QUBOProblem
from qqa.relaxation import BinaryRelaxation


class SparseQUBOProblem(QUBOProblem):
    """Binary problem whose hot objective path scales with nonzero edges."""

    def __init__(self, qubo: SparseQUBO, *, name: str = "sparse-qubo") -> None:
        self.sparse_qubo = qubo
        self.name = name
        self.num_nodes = qubo.num_variables
        self.device = qubo.linear.device
        self.relaxation = BinaryRelaxation()

    @property
    def Q_mat(self) -> torch.Tensor:
        """Deprecated dense compatibility view, materialised only on access."""
        return self.sparse_qubo.to_dense()

    def generate_qubo_matrix(self) -> torch.Tensor:
        return self.sparse_qubo.to_dense()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparse_qubo.energy(x)

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        values = x_disc if x_disc.ndim == 2 else x_disc.unsqueeze(0)
        energy = self.loss_fn(values)
        index = int(torch.argmin(energy).item())
        return {
            "label": "QUBO objective",
            "value": float(energy[index].item()),
            "unit": "",
            "feasible": True,
            "extra": {
                "variables": self.num_nodes,
                "quadratic_edges": self.sparse_qubo.num_edges,
            },
        }


__all__ = ["SparseQUBOProblem"]
