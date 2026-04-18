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

    def project(self, x: torch.Tensor) -> torch.Tensor: ...

    def penalty(self, x: torch.Tensor, curve_rate: int) -> torch.Tensor: ...

    def diversity(self, x: torch.Tensor) -> torch.Tensor: ...

    def perturb_(self, x: torch.Tensor, learning_rate: float, temp: float) -> None: ...

    def num_variables(self, problem) -> int: ...


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
        return x

    def project(self, x):
        # AdamW can push ``x`` far outside ``[0, 1]`` during early epochs, and
        # plain ``round()`` preserves that drift (round(-5) = -5). Clamping
        # first guarantees the discrete projection lives in ``{0, 1}`` so
        # problem losses evaluated on it remain meaningful.
        return x.clamp(0.0, 1.0).round()

    def penalty(self, x, curve_rate):
        # Sum across variable axes (keep leading batch axes intact).
        # For shape (B, N) -> (B,); for shape (B, I, N) -> (B, I).
        return torch.sum(1 - (1 - 2 * x) ** curve_rate, dim=-1)

    def diversity(self, x):
        # Standard deviation across the batch axis (dim=0), summed over the rest.
        std = x.std(dim=0)
        return std.sum()

    def perturb_(self, x, learning_rate, temp):
        if temp <= 0:
            return
        with torch.no_grad():
            noise = torch.randn_like(x) * ((2 * learning_rate * temp) ** 0.5)
            x.add_(noise).clamp_(0.0, 1.0)

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

    def __init__(self, shape_fn=None):
        super().__init__(shape_fn=shape_fn)

    def forward(self, x):
        return 2 * x.clamp(0.0, 1.0) - 1

    def project(self, x):
        return torch.where(
            x >= 0.5,
            torch.ones_like(x),
            -torch.ones_like(x),
        )

    def perturb_(self, x, learning_rate, temp):
        with torch.no_grad():
            if temp > 0:
                noise = torch.randn_like(x) * ((2 * learning_rate * temp) ** 0.5)
                x.add_(noise)
            x.clamp_(0.0, 1.0)

    def num_variables(self, problem):
        return getattr(problem, "num_spins", getattr(problem, "num_nodes", None))

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
        return torch.rand(
            (sol_size, problem.num_node, problem.num_category),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

    def forward(self, x):
        return x / x.sum(dim=2, keepdim=True)

    def project(self, x):
        idx = torch.argmax(x, dim=2)
        out = torch.zeros_like(x)
        out.scatter_(2, idx.unsqueeze(2), 1)
        return out

    def penalty(self, x, curve_rate):
        x_norm = self.forward(x)
        K = x.shape[2]
        num = torch.sum((K * x_norm - 1) ** curve_rate, dim=2)
        denom = (K - 1) ** curve_rate + (K - 1)
        return torch.sum(1 - num / denom, dim=1)

    def diversity(self, x):
        # Mean across categories -> sum across nodes (matches legacy formulation).
        return x.std(dim=0).mean(dim=1).sum()

    def perturb_(self, x, learning_rate, temp):
        if temp <= 0:
            return
        with torch.no_grad():
            noise = torch.randn_like(x) * ((2 * learning_rate * temp) ** 0.5)
            x.add_(noise).clamp_(1e-5, 1.0)

    def num_variables(self, problem):
        return problem.num_node
