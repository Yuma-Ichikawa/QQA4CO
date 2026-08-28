"""Device-resident projections and repairs for common discrete structures."""

from __future__ import annotations

import torch


@torch.no_grad()
def exact_k_repair(values: torch.Tensor, indices: torch.Tensor, target: int) -> torch.Tensor:
    result = values.clone()
    scope = torch.as_tensor(indices, device=values.device, dtype=torch.long).reshape(-1)
    if isinstance(target, bool) or not 0 <= target <= len(scope):
        raise ValueError("target must lie in [0, number of selected variables].")
    selected = values[..., scope]
    repaired = torch.zeros_like(selected)
    if target:
        top = torch.topk(selected, target, dim=-1).indices
        repaired.scatter_(-1, top, 1)
    result[..., scope] = repaired
    return result


@torch.no_grad()
def one_hot_repair(values: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("one_hot_repair requires (..., items, categories).")
    result = torch.zeros_like(values)
    result.scatter_(-1, values.argmax(dim=-1, keepdim=True), 1)
    return result


@torch.no_grad()
def assignment_repair(values: torch.Tensor) -> torch.Tensor:
    """Greedy one-to-one projection that remains on the input device."""
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise ValueError("assignment_repair requires square (..., n, n) scores.")
    size = values.shape[-1]
    batch = values.reshape(-1, size, size)
    result = torch.zeros_like(batch)
    available = torch.ones((len(batch), size), dtype=torch.bool, device=values.device)
    certainty = batch.topk(min(2, size), dim=-1).values
    margin = certainty[..., 0] - (certainty[..., 1] if size > 1 else 0)
    row_order = margin.argsort(dim=-1, descending=True)
    batch_index = torch.arange(len(batch), device=values.device)
    for step in range(size):
        row = row_order[:, step]
        scores = batch[batch_index, row].masked_fill(~available, -torch.inf)
        column = scores.argmax(dim=-1)
        result[batch_index, row, column] = 1
        available[batch_index, column] = False
    return result.reshape_as(values)


__all__ = ["assignment_repair", "exact_k_repair", "one_hot_repair"]
