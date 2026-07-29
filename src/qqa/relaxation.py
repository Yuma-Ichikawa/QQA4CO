"""Relaxation strategies for QQA.

A ``Relaxation`` defines how a combinatorial variable is represented as a
continuous tensor during annealing. It encapsulates:

* initialization of the relaxed variable,
* the transformation fed into ``problem.loss_fn`` (``forward``),
* the discrete projection used to evaluate the true objective (``project``),
* the quasi-quantum penalty function,
* the diversity term across the parallel batch,
* an in-place Langevin-style perturbation.

All relaxations operate on a leading batch dimension of size ``sol_size``.
"""

from __future__ import annotations

from typing import Protocol

import torch


class Relaxation(Protocol):
    """Protocol that any relaxation strategy must satisfy."""

    def init(self, sol_size: int, problem, device) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def encode(self, values: torch.Tensor) -> torch.Tensor: ...

    def project(self, x: torch.Tensor) -> torch.Tensor: ...

    def penalty(self, x: torch.Tensor, curve_rate: int) -> torch.Tensor: ...

    def diversity(self, x: torch.Tensor) -> torch.Tensor: ...

    def perturb_(self, x: torch.Tensor, learning_rate: float, temp: float) -> None: ...

    def num_variables(self, problem) -> int: ...


def _default_penalty_from_forward(relax, x, x_fwd, curve_rate):  # noqa: ARG001 - x_fwd unused in default
    """Default fallback: ignore the cached forward and recompute via ``penalty``.

    Relaxations whose ``penalty`` does not internally re-run ``forward`` (e.g.
    :class:`BinaryRelaxation`, :class:`SpinRelaxation`) get the same numerical
    behaviour. Relaxations that *do* re-run ``forward`` (notably
    :class:`CategoricalRelaxation`) should override ``penalty_from_forward``
    so the annealing loop can reuse the cached normalised tensor and skip the
    second forward pass.
    """
    return relax.penalty(x, curve_rate)


class BinaryRelaxation:
    """Relaxation for binary variables x in [0, 1].

    Used for QUBO problems (MIS, MaxClique, MaxCut) on either a single graph
    (shape ``(sol_size, N)``) or a batch of graphs via an instance problem
    (shape ``(sol_size, I, N)``).
    """

    def __init__(self, shape_fn=None):
        # shape_fn lets specialised problems override the tensor shape.
        self._shape_fn = shape_fn

    def init(self, sol_size, problem, device):
        shape = (
            self._shape_fn(sol_size, problem) if self._shape_fn else (sol_size, problem.num_nodes)
        )
        return torch.rand(shape, device=device, requires_grad=True)

    def forward(self, x):
        # Clamp to [0, 1] so the loss / penalty Φ(p) = 1 - (1 - 2p)^α stay
        # within the regime CRA Theorem 3.1 assumes ("ˆl(p; C) bounded on
        # [0, 1]^N"). Without this clamp, AdamW can push ``x`` outside the
        # cube and Φ becomes *negative*, which gives the optimiser a perverse
        # incentive to drift further out — empirically this freezes PQQA's
        # best loss / DIV value within the first few thousand epochs at
        # ``temp=0`` (the default). See ``tasks/test/verify_freeze_bug.py``.
        return x.clamp(0.0, 1.0)

    def encode(self, values):
        """Map physical binary values back to latent coordinates."""
        return values.clamp(0.0, 1.0)

    def project(self, x):
        # AdamW can push ``x`` far outside ``[0, 1]`` during early epochs, and
        # plain ``round()`` preserves that drift (round(-5) = -5). Clamping
        # first guarantees the discrete projection lives in ``{0, 1}`` so
        # problem losses evaluated on it remain meaningful.
        return x.clamp(0.0, 1.0).round()

    def penalty(self, x, curve_rate):
        # Sum across variable axes (keep leading batch axes intact).
        # For shape (B, N) -> (B,); for shape (B, I, N) -> (B, I).
        # Clamp to [0, 1] for the same reason ``forward`` does — the penalty
        # is otherwise unbounded below for x outside the cube and the CRA
        # annealing schedule's discrete attractor at γ > 0 collapses.
        x_clip = x.clamp(0.0, 1.0)
        return torch.sum(1 - (1 - 2 * x_clip) ** curve_rate, dim=-1)

    def diversity(self, x):
        # Standard deviation across the batch axis (dim=0), summed over the rest.
        std = x.std(dim=0)
        return std.sum()

    def perturb_(self, x, learning_rate, temp):
        # Always clamp x back into [0, 1] in-place after the AdamW step, even
        # when ``temp == 0`` (the PQQA default). Without the clamp, AdamW
        # drifts ``x`` outside the cube within ~1 k epochs and the relaxation
        # loses its semantic meaning (the discrete project clips, so the
        # *integer* solution stays sensible, but the gradient signal that
        # drives further improvement vanishes).
        with torch.no_grad():
            if temp > 0:
                noise = torch.randn_like(x) * ((2 * learning_rate * temp) ** 0.5)
                x.add_(noise)
            x.clamp_(0.0, 1.0)

    def num_variables(self, problem):
        return problem.num_nodes


def _instance_shape(sol_size, problem):
    return (sol_size, problem.num_instance, problem.max_node)


class SpinRelaxation(BinaryRelaxation):
    """Relaxation for ising-style spin variables ``s \\in \\{-1, +1\\}``.

    Internally the latent representation ``x`` lives in ``[0, 1]`` (same as
    :class:`BinaryRelaxation`), but :meth:`forward` maps it to the spin
    ``s = 2 \\, \\text{clip}(x, 0, 1) - 1`` so that ``problem.loss_fn`` can
    safely work on real-valued spins in ``[-1, +1]``. The discrete projection
    thresholds at ``0.5``.

    Because spin problems typically couple variables quadratically without a
    convex QUBO structure, AdamW steps can push the latent ``x`` outside
    ``[0, 1]``; we clip before the forward so the effective spin stays in
    ``[-1, +1]``, and :meth:`perturb_` always clamps ``x`` back even when
    ``temp == 0``.
    """

    def forward(self, x):
        return 2 * x.clamp(0.0, 1.0) - 1

    def encode(self, values):
        """Map physical spins in ``[-1, 1]`` back to ``[0, 1]``."""
        return ((values + 1.0) * 0.5).clamp(0.0, 1.0)

    def project(self, x):
        # Clamp first for the same defensive reason ``BinaryRelaxation.project``
        # does: AdamW can drift the latent ``x`` outside ``[0, 1]`` between
        # the final perturb_ and a downstream caller's manual ``project``,
        # and a stray ``x = -3`` would otherwise still threshold cleanly to
        # ``-1`` here but mask the issue. Broadcasted scalar ``where``
        # avoids allocating two ``ones_like`` intermediates per call (hot
        # on large batches).
        return torch.where(x.clamp(0.0, 1.0) >= 0.5, 1.0, -1.0).to(x.dtype)

    # ``perturb_`` is inherited from :class:`BinaryRelaxation` — the
    # latent ``x`` lives in ``[0, 1]`` for both relaxations so the noise
    # schedule and ``clamp_(0, 1)`` rule are identical.

    def num_variables(self, problem):
        n = getattr(problem, "num_spins", getattr(problem, "num_nodes", None))
        if n is None:
            raise TypeError(
                f"SpinRelaxation requires the problem to expose 'num_spins' "
                f"or 'num_nodes'; got {type(problem).__name__}."
            )
        return n

    def init(self, sol_size, problem, device):
        if self._shape_fn is not None:
            shape = self._shape_fn(sol_size, problem)
        else:
            shape = (sol_size, self.num_variables(problem))
        return torch.rand(shape, device=device, requires_grad=True)


class BinaryInstanceRelaxation(BinaryRelaxation):
    """Binary relaxation for batched instance problems.

    Expects the problem to expose ``num_instance`` and ``max_node``.
    """

    def __init__(self):
        super().__init__(shape_fn=_instance_shape)

    def num_variables(self, problem):
        return problem.max_node


class CategoricalRelaxation:
    """Relaxation for one-hot categorical variables.

    Variable tensor shape: ``(sol_size, N, K)``. The forward pass normalises
    across the category axis and ``project`` returns one-hot tensors.
    """

    def init(self, sol_size, problem, device):
        K = getattr(problem, "num_category", None)
        if K is None or K < 2:
            raise ValueError(
                "CategoricalRelaxation requires the problem to expose "
                f"num_category >= 2; got {K!r}."
            )
        return torch.rand(
            (sol_size, problem.num_node, K),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

    def forward(self, x):
        # AdamW is unconstrained and can push entries below zero even when
        # Langevin noise is disabled. Normalising those raw values can produce
        # negative "probabilities" or an almost-zero denominator. Clamp first
        # so the tensor remains on the non-negative simplex by construction.
        x_pos = x.clamp(min=1e-8)
        return x_pos / x_pos.sum(dim=2, keepdim=True)

    def encode(self, values):
        """Use simplex/one-hot values directly as latent coordinates."""
        values = values.clamp_min(0.0)
        return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def project(self, x):
        idx = torch.argmax(self.forward(x), dim=2)
        out = torch.zeros_like(x)
        out.scatter_(2, idx.unsqueeze(2), 1)
        return out

    def penalty(self, x, curve_rate):
        x_norm = self.forward(x)
        return self._penalty_from_norm(x_norm, curve_rate)

    def penalty_from_forward(self, x, x_fwd, curve_rate):  # noqa: ARG002 - x unused
        """Optimised path: reuse the already-normalised tensor from ``forward``.

        Used by :func:`qqa.anneal` to avoid the simplex normalisation
        re-running every epoch (the legacy ``penalty`` calls ``forward``
        internally, which doubled the cost on the hot path).
        """
        return self._penalty_from_norm(x_fwd, curve_rate)

    @staticmethod
    def _penalty_from_norm(x_norm, curve_rate):
        K = x_norm.shape[2]
        if K < 2:
            raise ValueError(
                f"CategoricalRelaxation.penalty is undefined for K={K}; use num_category >= 2."
            )
        num = torch.sum((K * x_norm - 1) ** curve_rate, dim=2)
        denom = (K - 1) ** curve_rate + (K - 1)
        return torch.sum(1 - num / denom, dim=1)

    def diversity(self, x):
        # Measure diversity in probability space. Raw logits have arbitrary
        # scale, so using them directly lets the optimiser manufacture
        # "diversity" without changing any categorical distribution.
        return self.forward(x).std(dim=0).mean(dim=1).sum()

    def perturb_(self, x, learning_rate, temp):
        with torch.no_grad():
            if temp > 0:
                noise = torch.randn_like(x) * ((2 * learning_rate * temp) ** 0.5)
                x.add_(noise)
            # Always restore the domain after AdamW, including the common
            # ``temp == 0`` path.
            x.clamp_(1e-5, 1.0)

    def num_variables(self, problem):
        return problem.num_node
