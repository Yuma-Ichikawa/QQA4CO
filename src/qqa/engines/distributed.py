"""One-process-per-device replica islands with Gloo/NCCL elite exchange."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class DistributedExchange:
    local_elites: torch.Tensor
    gathered_elites: torch.Tensor
    backend: str
    world_size: int


def exchange_elites(
    elites: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
) -> DistributedExchange:
    """All-gather a fixed-size elite tensor at a coarse migration boundary.

    CUDA tensors with an NCCL process group remain device-to-device. CPU test
    and development environments can use Gloo with identical semantics.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialised before elite exchange.")
    local = torch.as_tensor(elites).contiguous()
    if local.ndim < 2 or local.shape[0] < 1:
        raise ValueError("elites must have shape (migration_size, ...).")
    world_size = dist.get_world_size(group)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local, group=group)
    return DistributedExchange(
        local,
        torch.cat(gathered, dim=0),
        dist.get_backend(group),
        world_size,
    )


def select_diverse_migrants(
    candidates: torch.Tensor,
    objectives: torch.Tensor,
    *,
    count: int,
) -> torch.Tensor:
    """Retain the best candidate, then maximise minimum Hamming distance."""
    values = torch.as_tensor(candidates)
    scores = torch.as_tensor(objectives, device=values.device).reshape(-1)
    if values.ndim < 2 or values.shape[0] != scores.numel():
        raise ValueError("candidates and objectives must align on their first dimension.")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(values):
        raise ValueError("count must be in [1, number of candidates].")
    selected = [int(torch.argmin(scores).item())]
    flattened = values.reshape(len(values), -1)
    while len(selected) < count:
        distances = torch.stack(
            [(flattened != flattened[index]).to(torch.float32).mean(dim=-1) for index in selected]
        ).amin(dim=0)
        distances[selected] = -1
        quality = (scores.max() - scores) / (scores.max() - scores.min()).clamp_min(1e-12)
        selected.append(int(torch.argmax(distances + 0.05 * quality).item()))
    return values[torch.as_tensor(selected, device=values.device)]


def distributed_island_ready(device: str | torch.device) -> bool:
    """Return whether the active process group matches ``device``."""
    if not dist.is_available() or not dist.is_initialized():
        return False
    resolved = torch.device(device)
    backend = dist.get_backend()
    return (resolved.type == "cuda" and backend == "nccl") or (
        resolved.type == "cpu" and backend == "gloo"
    )


def anneal_distributed_island(
    problem,
    *,
    rounds: int = 4,
    migration_size: int = 8,
    group: dist.ProcessGroup | None = None,
    seed: int | None = None,
    **anneal_kwargs,
):
    """Run coarse QQA rounds and exchange diverse elites across process ranks.

    Launch one process per GPU with torchrun and initialise NCCL before calling
    this function. Gloo follows the same contract for CPU development tests.
    """
    from qqa.annealing import anneal

    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer.")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        from qqa.utils import fix_seed

        fix_seed(seed)
    sol_size = int(anneal_kwargs.pop("sol_size", 100))
    if not 1 <= migration_size <= sol_size:
        raise ValueError("migration_size must be in [1, sol_size].")
    device = anneal_kwargs.get("device", "cpu")
    if not distributed_island_ready(device):
        raise RuntimeError("The active process group backend does not match the solver device.")

    total_epochs = int(anneal_kwargs.pop("num_epochs", 10_000))
    total_time = anneal_kwargs.pop("time_limit", None)
    if total_time is not None and (
        isinstance(total_time, bool) or not isinstance(total_time, (int, float)) or total_time <= 0
    ):
        raise ValueError("time_limit must be positive or None.")
    deadline = None if total_time is None else perf_counter() + float(total_time)
    effective_rounds = min(rounds, max(1, total_epochs))
    final_polish = bool(anneal_kwargs.pop("polish", True))
    initial_state = anneal_kwargs.pop("initial_state", None)
    best_solution = None
    best_objective = torch.inf
    result = None
    exchanges = 0
    completed_rounds = 0
    for round_index in range(effective_rounds):
        remaining = None if deadline is None else deadline - perf_counter()
        if remaining is not None and remaining <= 0 and result is not None:
            break
        round_epochs = total_epochs // effective_rounds + int(
            round_index < total_epochs % effective_rounds
        )
        result = anneal(
            problem,
            **anneal_kwargs,
            sol_size=sol_size,
            num_epochs=round_epochs,
            time_limit=(
                None
                if remaining is None
                else max(1e-6, remaining / max(1, effective_rounds - round_index))
            ),
            initial_state=initial_state,
            polish=final_polish and round_index == rounds - 1,
            return_population=True,
        )
        if not isinstance(result.best_obj, float):
            raise TypeError("Distributed islands currently require a single-instance problem.")
        if result.best_obj < best_objective:
            best_objective = result.best_obj
            best_solution = result.best_sol.detach().clone()
        population = result.final_population
        if population is None:
            raise RuntimeError("QQA did not return its final population.")
        with torch.no_grad():
            objectives = problem.loss_fn(population).reshape(-1)
        completed_rounds += 1
        if round_index == effective_rounds - 1 or (
            deadline is not None and perf_counter() >= deadline
        ):
            break
        migrants = select_diverse_migrants(
            population,
            objectives,
            count=migration_size,
        )
        gathered = exchange_elites(migrants, group=group).gathered_elites
        gathered_objectives = problem.loss_fn(gathered).reshape(-1)
        retained_count = sol_size - migration_size
        retained = (
            population[torch.argsort(objectives)[:retained_count]]
            if retained_count
            else population[:0]
        )
        retained_objectives = (
            problem.loss_fn(retained).reshape(-1) if retained_count else objectives[:0]
        )
        pool = torch.cat((retained, gathered), dim=0)
        pool_objectives = torch.cat((retained_objectives, gathered_objectives), dim=0)
        initial_state = select_diverse_migrants(pool, pool_objectives, count=sol_size)
        exchanges += 1

    assert result is not None and best_solution is not None
    objective_tensor = torch.tensor(
        [best_objective], device=best_solution.device, dtype=torch.float64
    )
    gathered_objectives = [
        torch.empty_like(objective_tensor) for _ in range(dist.get_world_size(group))
    ]
    gathered_solutions = [
        torch.empty_like(best_solution) for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather(gathered_objectives, objective_tensor, group=group)
    dist.all_gather(gathered_solutions, best_solution.contiguous(), group=group)
    winner = int(torch.argmin(torch.cat(gathered_objectives)).item())
    result.best_sol = gathered_solutions[winner].detach().clone()
    result.best_obj = float(gathered_objectives[winner].item())
    # The final population is the population actually evaluated in the final
    # round, never an unevaluated post-migration pool.
    result.final_population = population
    result.diagnostics.update(
        {
            "distributed_backend": dist.get_backend(group),
            "distributed_world_size": dist.get_world_size(group),
            "distributed_rounds": rounds,
            "distributed_effective_rounds": effective_rounds,
            "distributed_completed_rounds": completed_rounds,
            "distributed_exchanges": exchanges,
            "migration_size": migration_size,
            "deadline_reached": deadline is not None and perf_counter() >= deadline,
        }
    )
    return result


__all__ = [
    "DistributedExchange",
    "anneal_distributed_island",
    "distributed_island_ready",
    "exchange_elites",
    "select_diverse_migrants",
]
