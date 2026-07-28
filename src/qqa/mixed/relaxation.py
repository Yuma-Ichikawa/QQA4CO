"""Continuous relaxation for heterogeneous bounded variable spaces."""

from __future__ import annotations

import math

import torch

from qqa.mixed.variables import VariableSpace


class MixedRelaxation:
    """Relax binary, bounded-integer, and bounded-real variables together.

    Every coordinate is represented internally in ``[0, 1]``. Binary
    coordinates use the standard QQA penalty, integer coordinates use a
    periodic grid penalty, and real coordinates remain continuous.
    """

    def __init__(self, space: VariableSpace):
        self.space = space
        self._binary_indices = [index for index, kind in enumerate(space.kinds) if kind == "binary"]
        self._integer_indices = [
            index for index, kind in enumerate(space.kinds) if kind == "integer"
        ]

    def init(self, sol_size, problem, device):
        return torch.rand(
            (sol_size, self.space.dimension),
            device=device,
            dtype=problem.dtype,
            requires_grad=True,
        )

    def forward(self, x):
        return self.space.decode(x)

    def project(self, x):
        return self.space.project(x)

    def encode(self, values):
        return self.space.encode(values)

    def penalty(self, x, curve_rate):
        latent = x.clamp(0.0, 1.0)

        penalty = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        if self._binary_indices:
            xb = latent[..., self._binary_indices]
            penalty = penalty + (1 - (1 - 2 * xb) ** curve_rate).sum(dim=-1)
        if self._integer_indices:
            # Zero exactly at each integer grid point, positive between them.
            # Use offset-free grid coordinates to retain accuracy when bounds
            # themselves are large (sin(pi * 1e12) is numerically fragile).
            lower, upper = self.space._bounds_like(latent)
            steps = (upper - lower)[self._integer_indices]
            phase = latent[..., self._integer_indices] * steps
            penalty = penalty + torch.sin(math.pi * phase).abs().pow(curve_rate).sum(dim=-1)
        return penalty

    def diversity(self, x):
        # Latent coordinates put all variable types on the same [0, 1] scale.
        return x.clamp(0.0, 1.0).std(dim=0).sum()

    def perturb_(self, x, learning_rate, temp):
        with torch.no_grad():
            if temp > 0:
                x.add_(torch.randn_like(x) * ((2 * learning_rate * temp) ** 0.5))
            x.clamp_(0.0, 1.0)

    def num_variables(self, problem):  # noqa: ARG002 - protocol signature
        return self.space.dimension
