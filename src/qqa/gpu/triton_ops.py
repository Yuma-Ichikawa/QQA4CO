"""Optional Triton kernels for sparse QUBO energy and analytic gradients."""

from __future__ import annotations

import torch

try:  # pragma: no cover - exercised only by the opt-in CUDA nightly job
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - ordinary CPU installations
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:  # pragma: no branch - definitions are import-time only

    @triton.jit
    def _linear_kernel(
        values, linear, energy, gradient, n_elements: tl.constexpr, BLOCK: tl.constexpr
    ):
        replica = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(values + replica * n_elements + offsets, mask=mask, other=0.0).to(tl.float32)
        coefficient = tl.load(linear + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(gradient + replica * n_elements + offsets, coefficient, mask=mask)
        tl.atomic_add(energy + replica, tl.sum(x * coefficient, axis=0))

    @triton.jit
    def _edge_kernel(
        values,
        source,
        target,
        weights,
        energy,
        gradient,
        n_variables: tl.constexpr,
        n_edges: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        replica = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_edges
        left = tl.load(source + offsets, mask=mask, other=0)
        right = tl.load(target + offsets, mask=mask, other=0)
        weight = tl.load(weights + offsets, mask=mask, other=0.0).to(tl.float32)
        x_left = tl.load(values + replica * n_variables + left, mask=mask, other=0.0).to(tl.float32)
        x_right = tl.load(values + replica * n_variables + right, mask=mask, other=0.0).to(
            tl.float32
        )
        tl.atomic_add(energy + replica, tl.sum(weight * x_left * x_right, axis=0))
        tl.atomic_add(gradient + replica * n_variables + left, weight * x_right, mask=mask)
        tl.atomic_add(gradient + replica * n_variables + right, weight * x_left, mask=mask)


def triton_sparse_qubo_energy_gradient(
    values: torch.Tensor,
    linear: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    constant: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused edge kernels, falling back is handled by the caller."""
    if not triton_available() or not values.is_cuda:
        raise RuntimeError("Triton sparse QUBO kernels require an available CUDA device.")
    if values.shape[-1] != linear.numel():
        raise ValueError("values and linear dimensions do not align.")
    original_shape = values.shape
    flat = values.contiguous().reshape(-1, original_shape[-1])
    coefficients = linear.to(device=values.device, dtype=values.dtype).contiguous()
    edges = edge_index.to(device=values.device, dtype=torch.int64).contiguous()
    weights = edge_weight.to(device=values.device, dtype=values.dtype).contiguous()
    energy = torch.full(
        (flat.shape[0],), float(constant), dtype=torch.float32, device=values.device
    )
    gradient = torch.empty_like(flat, dtype=torch.float32)
    block = 256
    _linear_kernel[(flat.shape[0], triton.cdiv(flat.shape[1], block))](
        flat, coefficients, energy, gradient, flat.shape[1], BLOCK=block
    )
    if weights.numel():
        _edge_kernel[(flat.shape[0], triton.cdiv(weights.numel(), block))](
            flat,
            edges[0],
            edges[1],
            weights,
            energy,
            gradient,
            flat.shape[1],
            weights.numel(),
            BLOCK=block,
        )
    return energy.reshape(original_shape[:-1]).to(values.dtype), gradient.reshape(
        original_shape
    ).to(values.dtype)


__all__ = ["triton_available", "triton_sparse_qubo_energy_gradient"]
