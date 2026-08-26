"""Compile-friendly custom operators used by sparse GPU engines."""

from __future__ import annotations

from typing import Literal

import torch

from qqa.gpu.triton_ops import triton_available, triton_sparse_qubo_energy_gradient

SparseImplementation = Literal["auto", "torch", "triton"]


def _torch_energy_gradient(
    values: torch.Tensor,
    linear: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    constant: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[-1] != linear.numel():
        raise ValueError("values and linear dimensions do not align.")
    coefficients = linear.to(values)
    weights = edge_weight.to(values)
    edges = edge_index.to(values.device)
    energy = (values * coefficients).sum(dim=-1) + float(constant)
    gradient = coefficients.expand_as(values).clone()
    if weights.numel():
        source, target = edges
        pair = values[..., source] * values[..., target] * weights
        energy = energy + pair.sum(dim=-1)
        index_shape = (1,) * (values.ndim - 1) + (weights.numel(),)
        source_index = source.reshape(index_shape).expand(*values.shape[:-1], -1)
        target_index = target.reshape(index_shape).expand(*values.shape[:-1], -1)
        gradient.scatter_add_(-1, source_index, values[..., target] * weights)
        gradient.scatter_add_(-1, target_index, values[..., source] * weights)
    return energy, gradient


# A CompositeImplicitAutograd registration gives torch.compile/export a stable
# operator boundary while preserving a portable implementation on every device.
# Keep the Library object alive for the lifetime of the module.
_SPARSE_LIBRARY = torch.library.Library("qqa4co", "FRAGMENT")
_SPARSE_LIBRARY.define(
    "sparse_qubo_energy_gradient(Tensor values, Tensor linear, Tensor edge_index, "
    "Tensor edge_weight, float constant=0.) -> (Tensor, Tensor)"
)
_SPARSE_LIBRARY.impl(
    "sparse_qubo_energy_gradient", _torch_energy_gradient, "CompositeImplicitAutograd"
)


def sparse_qubo_energy_gradient(
    values: torch.Tensor,
    linear: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    constant: float = 0.0,
    *,
    implementation: SparseImplementation = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse energy and gradient through Triton or portable PyTorch.

    ``auto`` selects Triton only for CUDA tensors when Triton is importable;
    every other environment uses the deterministic gather/scatter fallback.
    """
    if implementation not in {"auto", "torch", "triton"}:
        raise ValueError("implementation must be auto, torch, or triton.")
    use_triton = implementation == "triton" or (
        implementation == "auto" and values.is_cuda and triton_available()
    )
    if use_triton:
        return triton_sparse_qubo_energy_gradient(values, linear, edge_index, edge_weight, constant)
    return torch.ops.qqa4co.sparse_qubo_energy_gradient(
        values, linear, edge_index, edge_weight, float(constant)
    )


class _SparseQUBOEnergy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        linear: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        constant: float,
        implementation: str,
    ) -> torch.Tensor:
        energy, gradient = sparse_qubo_energy_gradient(
            values,
            linear,
            edge_index,
            edge_weight,
            constant,
            implementation=implementation,
        )
        ctx.save_for_backward(gradient)
        return energy

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (gradient,) = ctx.saved_tensors
        return grad_output.unsqueeze(-1) * gradient, None, None, None, None, None


def sparse_qubo_energy(
    values: torch.Tensor,
    linear: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    constant: float = 0.0,
    *,
    implementation: SparseImplementation = "auto",
) -> torch.Tensor:
    """Autograd-enabled sparse energy backed by the analytic fused gradient."""
    return _SparseQUBOEnergy.apply(
        values, linear, edge_index, edge_weight, float(constant), implementation
    )


__all__ = ["SparseImplementation", "sparse_qubo_energy", "sparse_qubo_energy_gradient"]
