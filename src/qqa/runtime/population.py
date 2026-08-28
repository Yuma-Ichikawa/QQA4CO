"""Heterogeneous replica roles, warm states, exchange, and preconditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch


class ReplicaRole(IntEnum):
    CONVEXIFY = 0
    EXPLORE = 1
    DISCRETIZE = 2
    NOISY = 3
    INCUMBENT = 4
    LP_CENTRED = 5
    CONFLICT_AVOIDING = 6
    GLOBAL = 7


@dataclass(frozen=True, slots=True)
class WarmStateBundle:
    incumbent: torch.Tensor | None = None
    lp_primal: torch.Tensor | None = None
    conflict_avoiding: torch.Tensor | None = None

    def candidates(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            value.detach().clone()
            for value in (self.incumbent, self.lp_primal, self.conflict_avoiding)
            if value is not None
        )


def compose_warm_population(
    bundle: WarmStateBundle,
    *,
    replicas: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Broadcast several semantically distinct warm states across replicas."""
    candidates = bundle.candidates()
    if not candidates:
        return None
    shape = candidates[0].shape
    if any(candidate.shape != shape for candidate in candidates):
        raise ValueError("Every warm state must have the same shape.")
    rows = [candidate.to(device=device, dtype=dtype) for candidate in candidates]
    repeats = (replicas + len(rows) - 1) // len(rows)
    return torch.stack((rows * repeats)[:replicas])


def estimate_convexification_beta(problem: Any, *, margin: float = 1.05) -> float:
    """Return a safe c=2 beta from a Gershgorin Hessian lower bound.

    The estimate is conservative and does not claim convexity for unknown
    nonlinear callables.  A negative beta contributes ``8*abs(beta) I``.
    """
    qubo = getattr(problem, "sparse_qubo", None)
    if qubo is None:
        return 0.0
    diagonal = torch.zeros(qubo.num_variables, dtype=torch.float64, device=qubo.linear.device)
    radius = torch.zeros_like(diagonal)
    if qubo.num_edges:
        source, target = qubo.edge_index
        absolute = qubo.edge_weight.to(torch.float64).abs()
        radius.scatter_add_(0, source, absolute)
        radius.scatter_add_(0, target, absolute)
    lower_bound = float((diagonal - radius).amin().item())
    return -margin * max(0.0, -lower_bound) / 8.0


def factor_preconditioner(problem: Any, reference: torch.Tensor) -> torch.Tensor:
    """Build a diagonal factor-aware inverse-curvature approximation."""
    qubo = getattr(problem, "sparse_qubo", None)
    if qubo is None or reference.shape[-1] != qubo.num_variables:
        return torch.ones(reference.shape[1:], device=reference.device, dtype=reference.dtype)
    diagonal = qubo.linear.to(reference).abs().clamp_min(1e-6)
    if qubo.num_edges:
        source, target = qubo.edge_index.to(reference.device)
        absolute = qubo.edge_weight.to(reference).abs()
        diagonal.scatter_add_(0, source, absolute)
        diagonal.scatter_add_(0, target, absolute)
    inverse = diagonal.rsqrt()
    return inverse / inverse.mean().clamp_min(1e-12)


class ReplicaPortfolio:
    """Assign roles and perform device-resident parallel-tempering exchange."""

    def __init__(self, replicas: int, *, convexification_beta: float = 0.0) -> None:
        if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
            raise ValueError("replicas must be a positive integer.")
        if not math.isfinite(convexification_beta) or convexification_beta > 0:
            raise ValueError("convexification_beta must be finite and non-positive.")
        self.replicas = replicas
        self.convexification_beta = float(convexification_beta)

    def roles(self, device: torch.device | str) -> torch.Tensor:
        return torch.arange(self.replicas, device=device, dtype=torch.long) % len(ReplicaRole)

    def beta(
        self,
        base: float,
        progress: float,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        roles = self.roles(device)
        result = torch.full((self.replicas,), float(base), device=device, dtype=dtype)
        convex = roles == int(ReplicaRole.CONVEXIFY)
        result = torch.where(
            convex,
            torch.minimum(result, result.new_tensor(self.convexification_beta) * (1.0 - progress)),
            result,
        )
        result = torch.where(roles == int(ReplicaRole.EXPLORE), result * 0.25, result)
        result = torch.where(
            roles == int(ReplicaRole.DISCRETIZE),
            torch.maximum(result, result.new_tensor(0.1 * progress)),
            result,
        )
        result = torch.where(roles == int(ReplicaRole.NOISY), result * 0.5, result)
        return result

    def learning_rate_scale(self, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        roles = self.roles(device)
        scales = torch.ones(self.replicas, device=device, dtype=dtype)
        scales = torch.where(roles == int(ReplicaRole.CONVEXIFY), scales * 0.6, scales)
        scales = torch.where(roles == int(ReplicaRole.EXPLORE), scales * 1.25, scales)
        scales = torch.where(roles == int(ReplicaRole.NOISY), scales * 0.8, scales)
        scales = torch.where(roles == int(ReplicaRole.INCUMBENT), scales * 0.5, scales)
        return scales

    @torch.no_grad()
    def exchange_(
        self,
        state: torch.Tensor,
        merit: torch.Tensor,
        optimizer: Any,
        *,
        epoch: int,
    ) -> torch.Tensor:
        """Swap adjacent role states with a Metropolis acceptance rule."""
        if self.replicas < 2:
            return torch.zeros((), dtype=torch.int64, device=state.device)
        start = epoch % 2
        left = torch.arange(start, self.replicas - 1, 2, device=state.device)
        right = left + 1
        scores = merit.reshape(self.replicas, -1).mean(dim=1)
        temperatures = 0.25 + self.roles(state.device).to(state.dtype) / len(ReplicaRole)
        log_acceptance = (temperatures[left].reciprocal() - temperatures[right].reciprocal()) * (
            scores[left] - scores[right]
        )
        accepted = torch.log(torch.rand_like(log_acceptance).clamp_min(1e-12)) < torch.minimum(
            log_acceptance, torch.zeros_like(log_acceptance)
        )
        old_left = state[left].clone()
        old_right = state[right].clone()
        view = accepted.reshape(-1, *((1,) * (state.ndim - 1)))
        state[left] = torch.where(view, old_right, old_left)
        state[right] = torch.where(view, old_left, old_right)
        optimizer_state = getattr(optimizer, "state", {}).get(state, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = optimizer_state.get(key)
            if torch.is_tensor(value) and value.shape == state.shape:
                old_left = value[left].clone()
                old_right = value[right].clone()
                value[left] = torch.where(view, old_right, old_left)
                value[right] = torch.where(view, old_left, old_right)
        return accepted.sum()


__all__ = [
    "ReplicaPortfolio",
    "ReplicaRole",
    "WarmStateBundle",
    "compose_warm_population",
    "estimate_convexification_beta",
    "factor_preconditioner",
]
