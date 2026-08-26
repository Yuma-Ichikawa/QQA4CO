"""Energy-guided discrete diffusion generator for sparse binary QUBOs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qqa.compile import SparseQUBO
from qqa.local import sparse_qubo_descent


@dataclass(frozen=True, slots=True)
class DiscreteDiffusionResult:
    population: torch.Tensor
    objectives: torch.Tensor
    steps: int


class DiscreteDiffusionGenerator:
    """Generate diverse candidates by noisy parallel denoising and local polish."""

    def __init__(self, qubo: SparseQUBO) -> None:
        if not isinstance(qubo, SparseQUBO):
            raise TypeError("qubo must be a SparseQUBO.")
        self.qubo = qubo

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        *,
        steps: int = 64,
        temperature: float = 1.0,
        seed: int = 0,
        warm_start: torch.Tensor | None = None,
        polish: bool = True,
    ) -> DiscreteDiffusionResult:
        if num_samples < 1 or steps < 0 or temperature <= 0:
            raise ValueError("num_samples/temperature must be positive and steps non-negative.")
        generator = torch.Generator(device=self.qubo.linear.device).manual_seed(seed)
        population = torch.randint(
            0,
            2,
            (num_samples, self.qubo.num_variables),
            generator=generator,
            device=self.qubo.linear.device,
            dtype=torch.int64,
        ).to(self.qubo.linear)
        if warm_start is not None:
            seed_value = torch.as_tensor(warm_start, device=population.device).reshape(-1)
            if seed_value.numel() != self.qubo.num_variables:
                raise ValueError("warm_start does not align with the QUBO.")
            population[0] = seed_value.round().clamp(0, 1).to(population)
        for step in range(steps):
            progress = (step + 1) / max(steps, 1)
            delta = self.qubo.flip_delta(population)
            acceptance = torch.sigmoid(-delta / max(temperature * (1.0 - 0.95 * progress), 1e-6))
            proposal = (
                torch.rand(
                    population.shape,
                    generator=generator,
                    device=population.device,
                    dtype=population.dtype,
                )
                < acceptance
            )
            candidate = torch.where(proposal, 1 - population, population)
            candidate_values = self.qubo.energy(candidate)
            current_values = self.qubo.energy(population)
            noise = temperature * (1.0 - progress)
            accept = (candidate_values <= current_values) | (
                torch.rand(
                    num_samples,
                    generator=generator,
                    device=population.device,
                    dtype=population.dtype,
                )
                < noise / (1.0 + noise)
            )
            population[accept] = candidate[accept]
        if polish:
            population = torch.stack(
                [sparse_qubo_descent(self.qubo, sample).solution for sample in population]
            )
        objectives = self.qubo.energy(population)
        order = torch.argsort(objectives)
        return DiscreteDiffusionResult(population[order], objectives[order], steps)


__all__ = ["DiscreteDiffusionGenerator", "DiscreteDiffusionResult"]
