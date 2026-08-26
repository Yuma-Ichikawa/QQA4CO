"""Reusable projection/repair plugins for common discrete constraints."""

from __future__ import annotations

from collections.abc import Callable

import torch

RepairFunction = Callable[..., torch.Tensor]


class RepairRegistry:
    def __init__(self) -> None:
        self._functions: dict[str, RepairFunction] = {}

    def register(self, name: str, function: RepairFunction) -> None:
        if not name or not callable(function):
            raise ValueError("Repair name must be non-empty and function callable.")
        if name in self._functions:
            raise ValueError(f"Repair {name!r} is already registered.")
        self._functions[name] = function

    def get(self, name: str) -> RepairFunction:
        try:
            return self._functions[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown repair {name!r}; available: {sorted(self._functions)}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._functions))


@torch.no_grad()
def one_hot_projection(values: torch.Tensor) -> torch.Tensor:
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("one_hot_projection requires at least two categories.")
    result = torch.zeros_like(values)
    result.scatter_(-1, torch.argmax(values, dim=-1, keepdim=True), 1.0)
    return result


@torch.no_grad()
def exact_k_projection(values: torch.Tensor, k: int) -> torch.Tensor:
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= values.shape[-1]:
        raise ValueError("k must be an integer in [0, number of variables].")
    result = torch.zeros_like(values)
    if k:
        result.scatter_(-1, torch.topk(values, k=k, dim=-1).indices, 1.0)
    return result


@torch.no_grad()
def assignment_projection(values: torch.Tensor) -> torch.Tensor:
    if values.ndim not in {2, 3}:
        raise ValueError("assignment_projection expects (rows, cols) or (B, rows, cols).")
    unbatched = values.ndim == 2
    matrices = values.unsqueeze(0) if unbatched else values
    if matrices.shape[-2] != matrices.shape[-1]:
        raise ValueError("assignment_projection currently requires square matrices.")
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    result = torch.zeros_like(matrices)
    for batch, matrix in enumerate(matrices.detach().cpu().numpy()):
        rows, columns = linear_sum_assignment(-matrix)
        result[
            batch,
            torch.as_tensor(rows, device=values.device),
            torch.as_tensor(columns, device=values.device),
        ] = 1.0
    return result[0] if unbatched else result


@torch.no_grad()
def independent_set_repair(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    priorities: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("independent_set_repair expects one flat candidate.")
    edges = torch.as_tensor(edge_index, device=values.device, dtype=torch.long)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E).")
    result = values.round().clamp(0, 1).clone()
    degree = torch.bincount(edges.reshape(-1), minlength=values.numel()).to(values)
    priority = -degree if priorities is None else priorities.to(values)
    for left, right in edges.T:
        if result[left] > 0.5 and result[right] > 0.5:
            drop = right if priority[left] >= priority[right] else left
            result[drop] = 0.0
    adjacency = [set() for _ in range(values.numel())]
    for left, right in edges.detach().cpu().T.tolist():
        adjacency[left].add(right)
        adjacency[right].add(left)
    order = torch.argsort(priority, descending=True).tolist()
    selected = set(torch.nonzero(result > 0.5, as_tuple=False).reshape(-1).tolist())
    for vertex in order:
        if vertex not in selected and not (adjacency[vertex] & selected):
            result[vertex] = 1.0
            selected.add(vertex)
    return result


@torch.no_grad()
def knapsack_repair(
    values: torch.Tensor,
    weights: torch.Tensor,
    profits: torch.Tensor,
    capacity: float,
) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("knapsack_repair expects one flat candidate.")
    weights = torch.as_tensor(weights, device=values.device, dtype=torch.float64)
    profits = torch.as_tensor(profits, device=values.device, dtype=torch.float64)
    if weights.shape != values.shape or profits.shape != values.shape or torch.any(weights <= 0):
        raise ValueError("weights/profits must align with values and weights must be positive.")
    result = values.round().clamp(0, 1).clone()
    ratio = profits / weights
    selected_weight = float((result.to(weights) * weights).sum().item())
    for index in torch.argsort(ratio).tolist():
        if selected_weight <= capacity + 1e-12:
            break
        if result[index] > 0.5:
            result[index] = 0.0
            selected_weight -= float(weights[index].item())
    for index in torch.argsort(ratio, descending=True).tolist():
        item_weight = float(weights[index].item())
        if result[index] < 0.5 and selected_weight + item_weight <= capacity + 1e-12:
            result[index] = 1.0
            selected_weight += item_weight
    return result


@torch.no_grad()
def repair_model_ir(model, values: torch.Tensor, *, passes: int = 4) -> torch.Tensor:
    """Apply conservative native-factor repairs to one ModelIR candidate.

    Only mathematically recognisable projections are applied. Unknown or
    coupled constraints are left untouched instead of being guessed.
    """
    from qqa.model import (  # noqa: PLC0415
        AssignmentFactor,
        CardinalityFactor,
        ClauseFactor,
        LinearFactor,
        VariableDomain,
    )

    structured = model.structured_block
    if structured is not None:
        expected = (structured.size, int(structured.categories or 0))
        if values.shape != expected:
            raise ValueError(f"structured values must have shape {expected}.")
        return (
            assignment_projection(values)
            if structured.domain is VariableDomain.PERMUTATION
            else one_hot_projection(values)
        )
    if values.ndim != 1 or values.numel() != model.num_variables:
        raise ValueError("values must be one flattened ModelIR candidate.")
    result = values.detach().clone()
    binary = torch.zeros(model.num_variables, dtype=torch.bool, device=result.device)
    offset = 0
    for block in model.variables:
        if block.domain is VariableDomain.BINARY:
            binary[offset : offset + block.size] = True
        offset += block.size

    def cardinality(indices: torch.Tensor, target: int) -> None:
        scoped = indices.to(result.device)
        if not binary[scoped].all() or not 0 <= target <= len(scoped):
            return
        for _ in range(abs(int(result[scoped].round().sum().item()) - target)):
            current = result.round().clamp(0, 1)
            count = int(current[scoped].sum().item())
            eligible = (
                scoped[current[scoped] > 0.5] if count > target else scoped[current[scoped] < 0.5]
            )
            if not len(eligible):
                break
            candidates = current.repeat(len(eligible), 1)
            candidates[torch.arange(len(eligible), device=result.device), eligible] = (
                0.0 if count > target else 1.0
            )
            score = model.internal_energy(candidates)
            result.copy_(candidates[torch.argmin(score)])

    for _ in range(max(1, passes)):
        before = result.clone()
        for row in model.constraints:
            factors = row.expression.factors
            if len(factors) != 1:
                continue
            factor = factors[0]
            if isinstance(factor, CardinalityFactor) and row.sense == "<=" and row.rhs <= 0:
                target = round(factor.target)
                if abs(target - factor.target) <= 1e-9:
                    cardinality(factor.indices, target)
            elif isinstance(factor, AssignmentFactor) and row.sense == "<=" and row.rhs <= 0:
                indices = factor.indices.to(result.device)
                repaired = assignment_projection(result[indices])
                result[indices] = repaired
            elif isinstance(factor, LinearFactor) and row.sense == "==":
                indices = factor.indices.to(result.device)
                weights = factor.weights.to(result)
                if len(weights) and torch.allclose(weights, weights[0].expand_as(weights)):
                    weight = float(weights[0].item())
                    if abs(weight) > 1e-12:
                        target_value = (row.rhs - row.expression.constant) / weight
                        target = round(target_value)
                        if abs(target - target_value) <= 1e-9:
                            cardinality(indices, target)
            elif isinstance(factor, ClauseFactor) and row.sense == "<=" and row.rhs <= 0:
                clause_indices = factor.indices.to(result.device)
                signs = factor.signs.to(result.device)
                for indices, clause_signs in zip(clause_indices, signs, strict=True):
                    literals = torch.where(clause_signs > 0, result[indices], 1 - result[indices])
                    if bool((literals > 0.5).any()):
                        continue
                    candidates = result.repeat(len(indices), 1)
                    candidates[torch.arange(len(indices), device=result.device), indices] = (
                        clause_signs > 0
                    ).to(result)
                    result.copy_(candidates[torch.argmin(model.internal_energy(candidates))])
        if torch.equal(before, result):
            break
    return result


registry = RepairRegistry()
for _name, _function in (
    ("one-hot", one_hot_projection),
    ("exact-k", exact_k_projection),
    ("assignment", assignment_projection),
    ("independent-set", independent_set_repair),
    ("knapsack", knapsack_repair),
    ("model-ir", repair_model_ir),
):
    registry.register(_name, _function)


__all__ = [
    "RepairRegistry",
    "assignment_projection",
    "exact_k_projection",
    "independent_set_repair",
    "knapsack_repair",
    "one_hot_projection",
    "repair_model_ir",
    "registry",
]
