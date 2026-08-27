"""Canonical edge representation for sparse binary quadratic objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

from qqa.model import LinearFactor, ObjectiveIR, QuadraticEdgeFactor


@dataclass(frozen=True, slots=True)
class SparseQUBO:
    """Energy ``linear @ x + sum(w_e x_u x_v) + constant``.

    Every edge is canonicalised to ``u < v`` and duplicate edges are summed.
    On binary points this representation preserves arbitrary dense
    ``x.T @ Q @ x`` matrices by treating the diagonal as a linear term and
    combining ``Q[u,v] + Q[v,u]`` into one edge coefficient.  Away from the
    binary corners, :meth:`energy` uses the multilinear extension; this is the
    relaxation used by the sparse QQA engine.
    """

    linear: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    constant: float = 0.0

    def __post_init__(self) -> None:
        linear = torch.as_tensor(self.linear).detach().clone().reshape(-1)
        edges = (
            torch.as_tensor(self.edge_index, dtype=torch.long, device=linear.device)
            .detach()
            .clone()
        )
        weights = (
            torch.as_tensor(
                self.edge_weight,
                dtype=linear.dtype,
                device=linear.device,
            )
            .detach()
            .clone()
            .reshape(-1)
        )
        if linear.numel() == 0 or not torch.isfinite(linear).all():
            raise ValueError("linear must be a non-empty finite vector.")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] != weights.numel():
            raise ValueError("edge_index must have shape (2, E) aligned with edge_weight.")
        if (
            not torch.isfinite(weights).all()
            or torch.any(edges < 0)
            or torch.any(edges >= linear.numel())
        ):
            raise ValueError("Sparse QUBO edges and weights must be finite and in range.")
        if not math.isfinite(float(self.constant)):
            raise ValueError("constant must be finite.")
        if edges.shape[1]:
            source = torch.minimum(edges[0], edges[1])
            target = torch.maximum(edges[0], edges[1])
            diagonal = source == target
            if diagonal.any():
                linear.scatter_add_(0, source[diagonal], weights[diagonal])
                source = source[~diagonal]
                target = target[~diagonal]
                weights = weights[~diagonal]
            keys = source * linear.numel() + target
            unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
            combined = torch.zeros(unique.numel(), dtype=weights.dtype, device=weights.device)
            combined.scatter_add_(0, inverse, weights)
            nonzero = combined != 0
            unique = unique[nonzero]
            weights = combined[nonzero]
            edges = torch.stack((unique // linear.numel(), unique % linear.numel()))
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "edge_weight", weights)
        object.__setattr__(self, "constant", float(self.constant))

    @property
    def num_variables(self) -> int:
        return self.linear.numel()

    @property
    def num_edges(self) -> int:
        return self.edge_weight.numel()

    @property
    def density(self) -> float:
        possible = self.num_variables * (self.num_variables - 1) // 2
        return self.num_edges / possible if possible else 0.0

    @classmethod
    def from_dense(cls, matrix: torch.Tensor, *, constant: float = 0.0) -> SparseQUBO:
        matrix = torch.as_tensor(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("matrix must be a non-empty square tensor.")
        if not torch.isfinite(matrix).all():
            raise ValueError("matrix must contain only finite values.")
        n = matrix.shape[0]
        rows, columns = torch.triu_indices(n, n, offset=1, device=matrix.device)
        weights = matrix[rows, columns] + matrix[columns, rows]
        keep = weights != 0
        return cls(
            linear=torch.diagonal(matrix),
            edge_index=torch.stack((rows[keep], columns[keep])),
            edge_weight=weights[keep],
            constant=constant,
        )

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> SparseQUBO:
        dtype = self.linear.dtype if dtype is None else dtype
        return SparseQUBO(
            self.linear.to(device=device, dtype=dtype),
            self.edge_index.to(device=device),
            self.edge_weight.to(device=device, dtype=dtype),
            self.constant,
        )

    def energy(self, values: torch.Tensor) -> torch.Tensor:
        """Evaluate any leading batch dimensions in ``O(batch * edges)``."""
        if values.shape[-1] != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} variables, got {values.shape[-1]}.")
        linear = torch.matmul(values, self.linear.to(values))
        if self.num_edges == 0:
            return linear + self.constant
        # One vectorised advanced-index operation is materially faster than
        # gathering the two endpoints separately on current CPU and GPU Torch
        # kernels.  The final matmul also avoids an intermediate weighted edge
        # tensor while preserving autograd and arbitrary leading dimensions.
        endpoints = values[..., self.edge_index]
        pairwise = endpoints[..., 0, :] * endpoints[..., 1, :]
        return linear + torch.matmul(pairwise, self.edge_weight.to(values)) + self.constant

    def accelerated_energy(
        self,
        values: torch.Tensor,
        *,
        implementation: Literal["auto", "torch", "triton"] = "auto",
    ) -> torch.Tensor:
        """Evaluate with an analytic-gradient custom op or optional Triton kernels."""
        from qqa.gpu.ops import sparse_qubo_energy

        return sparse_qubo_energy(
            values,
            self.linear,
            self.edge_index,
            self.edge_weight,
            self.constant,
            implementation=implementation,
        )

    def energy_gradient(
        self,
        values: torch.Tensor,
        *,
        implementation: Literal["auto", "torch", "triton"] = "auto",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return energy and analytic gradient without constructing a dense matrix."""
        from qqa.gpu.ops import sparse_qubo_energy_gradient

        return sparse_qubo_energy_gradient(
            values,
            self.linear,
            self.edge_index,
            self.edge_weight,
            self.constant,
            implementation=implementation,
        )

    def compile_aot(self, example_values: torch.Tensor, **kwargs):
        """Create or load a persistent :mod:`torch.export`/AOTInductor artifact."""
        from qqa.compile.aot import compile_sparse_qubo_aot

        return compile_sparse_qubo_aot(self, example_values, **kwargs)

    def gradient(self, values: torch.Tensor) -> torch.Tensor:
        """Analytic continuous gradient without constructing a dense matrix."""
        if values.shape[-1] != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} variables, got {values.shape[-1]}.")
        gradient = self.linear.to(values).expand_as(values).clone()
        if self.num_edges:
            source, target = self.edge_index
            weights = self.edge_weight.to(values)
            source_shape = (1,) * (values.ndim - 1) + (self.num_edges,)
            source_index = source.reshape(source_shape).expand(*values.shape[:-1], -1)
            target_index = target.reshape(source_shape).expand(*values.shape[:-1], -1)
            gradient.scatter_add_(-1, source_index, values[..., target] * weights)
            gradient.scatter_add_(-1, target_index, values[..., source] * weights)
        return gradient

    def flip_delta(self, values: torch.Tensor) -> torch.Tensor:
        """Exact energy change for flipping each binary variable."""
        return (1.0 - 2.0 * values) * self.gradient(values)

    def to_dense(self) -> torch.Tensor:
        """Return a symmetric dense QUBO matrix with binary-corner parity.

        ``x.T @ Q @ x`` equals :meth:`energy` (minus ``constant``) for binary
        ``x``.  For continuous ``x``, use :meth:`energy` directly because its
        diagonal terms intentionally remain linear.
        """
        matrix = torch.diag(self.linear.clone())
        if self.num_edges:
            source, target = self.edge_index
            half = self.edge_weight / 2.0
            matrix[source, target] = half
            matrix[target, source] = half
        return matrix

    def objective_ir(self) -> ObjectiveIR:
        indices = torch.arange(self.num_variables, device=self.linear.device)
        return ObjectiveIR(
            (
                LinearFactor(indices, self.linear),
                QuadraticEdgeFactor(self.edge_index, self.edge_weight),
            ),
            constant=self.constant,
        )

    def connected_components(self) -> tuple[torch.Tensor, ...]:
        """Return deterministic variable components of the factor graph."""
        parent = list(range(self.num_variables))

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for left, right in self.edge_index.detach().cpu().T.tolist():
            union(left, right)
        groups: dict[int, list[int]] = {}
        for index in range(self.num_variables):
            groups.setdefault(find(index), []).append(index)
        return tuple(
            torch.tensor(group, device=self.linear.device, dtype=torch.long)
            for _, group in sorted(groups.items())
        )

    def subqubo(self, variables: torch.Tensor, *, include_constant: bool = False) -> SparseQUBO:
        """Extract an induced variable subproblem with local zero-based indices."""
        variables = torch.as_tensor(variables, device=self.linear.device, dtype=torch.long).reshape(
            -1
        )
        if (
            variables.numel() == 0
            or torch.any(variables < 0)
            or torch.any(variables >= self.num_variables)
        ):
            raise ValueError("variables must be a non-empty in-range index vector.")
        if torch.unique(variables).numel() != variables.numel():
            raise ValueError("variables must not contain duplicates.")
        inverse = torch.full(
            (self.num_variables,),
            -1,
            dtype=torch.long,
            device=self.linear.device,
        )
        inverse[variables] = torch.arange(variables.numel(), device=self.linear.device)
        source, target = self.edge_index
        keep = (inverse[source] >= 0) & (inverse[target] >= 0)
        return SparseQUBO(
            self.linear[variables],
            torch.stack((inverse[source[keep]], inverse[target[keep]])),
            self.edge_weight[keep],
            self.constant if include_constant else 0.0,
        )


def compile_sparse_qubo(problem: Any) -> SparseQUBO:
    """Compile a QUBO-like problem without exposing a solver-specific type."""
    sparse_qubo = getattr(problem, "sparse_qubo", None)
    if isinstance(sparse_qubo, SparseQUBO):
        return sparse_qubo
    matrix = getattr(problem, "Q_mat", None)
    if matrix is None and hasattr(problem, "generate_qubo_matrix"):
        matrix = problem.generate_qubo_matrix()
    if not torch.is_tensor(matrix):
        raise TypeError("problem must expose SparseQUBO, Q_mat, or generate_qubo_matrix().")
    return SparseQUBO.from_dense(matrix)


__all__ = ["SparseQUBO", "compile_sparse_qubo"]
