"""One-shot parallel Pareto-front optimisation.

Every replica follows a distinct Sobol/Dirichlet reference direction.  An
augmented Tchebycheff scalarisation is used instead of a weighted sum so
non-convex portions of the Pareto front remain reachable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter

import torch

from qqa.multiobjective.problem import MultiObjectiveProblem
from qqa.schedule import LinearBGSchedule
from qqa.utils import require_cuda_if_requested


@dataclass(slots=True)
class ParetoResult:
    """Nondominated feasible solutions collected during one parallel run."""

    solutions: torch.Tensor
    objectives: torch.Tensor
    weights: torch.Tensor
    runtime: float
    objective_names: tuple[str, ...]
    directions: tuple[str, ...]
    history: dict[str, list] = field(default_factory=dict)

    def named_solutions(self, problem: MultiObjectiveProblem) -> dict[str, torch.Tensor]:
        """Return variable-name views for every Pareto solution."""
        return problem.unpack(self.solutions)

    def to_frame(self):
        """Return a tidy pandas DataFrame for export or dashboards."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - core currently includes pandas
            raise ImportError("Install pandas or `qqa[plotly]` to create a DataFrame.") from exc
        data = {
            name: self.objectives[:, index].detach().cpu().numpy()
            for index, name in enumerate(self.objective_names)
        }
        return pd.DataFrame(data)


def _reference_directions(count: int, objectives: int, seed: int, *, device, dtype):
    engine = torch.quasirandom.SobolEngine(objectives, scramble=True, seed=seed)
    uniforms = engine.draw(count).clamp_(1e-7, 1 - 1e-7)
    weights = -uniforms.log()
    weights /= weights.sum(dim=1, keepdim=True)
    for index in range(min(count, objectives)):
        weights[index].zero_()
        weights[index, index] = 1.0
    return weights.to(device=device, dtype=dtype).clamp_min_(1e-6)


def _feasible_mask(problem: MultiObjectiveProblem, values: torch.Tensor) -> torch.Tensor:
    feasible = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    violations = problem.constraint_violations(values)
    for constraint in problem.constraints:
        feasible &= violations[constraint.name] <= constraint.tolerance
    return feasible


def nondominated_mask(values: torch.Tensor, *, tolerance: float = 1e-8) -> torch.Tensor:
    """Return the Pareto-efficient rows of an all-minimisation matrix."""
    if values.ndim != 2:
        raise ValueError("values must have shape (points, objectives).")
    if values.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=values.device)
    left = values[:, None, :]
    right = values[None, :, :]
    # right[j] dominates left[i]
    dominates = (right <= left + tolerance).all(dim=2) & (right < left - tolerance).any(dim=2)
    return ~dominates.any(dim=1)


def _crowding_distance(values: torch.Tensor) -> torch.Tensor:
    count, objectives = values.shape
    distance = torch.zeros(count, dtype=values.dtype, device=values.device)
    if count <= 2:
        return torch.full_like(distance, float("inf"))
    for index in range(objectives):
        order = torch.argsort(values[:, index])
        distance[order[0]] = distance[order[-1]] = float("inf")
        span = (values[order[-1], index] - values[order[0], index]).clamp_min(1e-12)
        distance[order[1:-1]] += (values[order[2:], index] - values[order[:-2], index]) / span
    return distance


def _update_archive(
    problem: MultiObjectiveProblem,
    solutions: torch.Tensor,
    archive_solutions: torch.Tensor | None,
    *,
    max_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    with torch.no_grad():
        feasible = _feasible_mask(problem, solutions)
        solutions = solutions[feasible]
        if solutions.shape[0] == 0:
            return (
                archive_solutions,
                None if archive_solutions is None else problem.objective_matrix(archive_solutions),
            )
        if archive_solutions is not None:
            solutions = torch.cat([archive_solutions.to(solutions.device), solutions])
        solutions = torch.unique(solutions, dim=0)
        objective_min = problem.objective_matrix(solutions, minimize=True)
        keep = nondominated_mask(objective_min)
        solutions = solutions[keep]
        objective_min = objective_min[keep]
        if solutions.shape[0] > max_size:
            distance = _crowding_distance(objective_min)
            selected = torch.topk(distance, k=max_size).indices
            solutions = solutions[selected]
        return solutions.detach(), problem.objective_matrix(solutions).detach()


def pareto_anneal(
    problem: MultiObjectiveProblem,
    *,
    sol_size: int = 256,
    num_epochs: int = 1500,
    learning_rate: float = 0.05,
    temp: float = 0.0,
    min_bg: float = -0.5,
    max_bg: float = 1.0,
    curve_rate: int = 2,
    augmentation: float = 0.05,
    div_param: float = 0.01,
    archive_interval: int = 25,
    archive_size: int = 2048,
    history_stride: int = 10,
    seed: int = 0,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> ParetoResult:
    """Find a diverse Pareto front in one GPU-parallel optimisation run."""
    if not isinstance(problem, MultiObjectiveProblem):
        raise TypeError("problem must be a MultiObjectiveProblem.")
    if not isinstance(sol_size, int) or sol_size < problem.num_objectives:
        raise ValueError("sol_size must be an integer >= the number of objectives.")
    if not isinstance(num_epochs, int) or num_epochs < 0:
        raise ValueError("num_epochs must be a non-negative integer.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and > 0.")
    if not math.isfinite(temp) or temp < 0:
        raise ValueError("temp must be finite and >= 0.")
    if not isinstance(curve_rate, int) or curve_rate < 2 or curve_rate % 2:
        raise ValueError("curve_rate must be a positive even integer.")
    if not math.isfinite(augmentation) or augmentation < 0:
        raise ValueError("augmentation must be finite and >= 0.")
    if not math.isfinite(div_param) or div_param < 0:
        raise ValueError("div_param must be finite and >= 0.")
    for name, value in (
        ("archive_interval", archive_interval),
        ("archive_size", archive_size),
        ("history_stride", history_stride),
    ):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    require_cuda_if_requested(device)
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)

    started = perf_counter()
    relaxation = problem.relaxation
    latent = relaxation.init(sol_size, problem, device)
    optimizer = torch.optim.AdamW([latent], lr=learning_rate)
    schedule = LinearBGSchedule(min_bg, max_bg)
    weights = _reference_directions(
        sol_size,
        problem.num_objectives,
        seed,
        device=latent.device,
        dtype=latent.dtype,
    )

    archive_solutions: torch.Tensor | None = None
    archive_objectives: torch.Tensor | None = None
    history: dict[str, list] = {
        "epoch": [],
        "pareto_size": [],
        "ideal": [],
        "nadir": [],
        "loss": [],
    }

    projected = relaxation.project(latent)
    archive_solutions, archive_objectives = _update_archive(
        problem, projected, archive_solutions, max_size=archive_size
    )

    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        values = relaxation.forward(latent)
        objective_min = problem.objective_matrix(values, minimize=True)
        detached = objective_min.detach()
        ideal = detached.amin(dim=0)
        nadir = detached.amax(dim=0)
        normalised = (objective_min - ideal) / (nadir - ideal).clamp_min(1e-8)
        weighted = weights * normalised
        scalar = weighted.amax(dim=1) + augmentation * weighted.sum(dim=1)
        constraint_loss = problem.constraint_penalty(values)
        discrete_penalty = relaxation.penalty(latent, curve_rate)
        diversity = relaxation.diversity(latent)
        bg = float(schedule(epoch, num_epochs))
        loss = (scalar + constraint_loss + bg * discrete_penalty).sum()
        loss = loss - div_param * sol_size * diversity
        loss.backward()
        optimizer.step()
        relaxation.perturb_(latent, learning_rate, temp)

        if (epoch + 1) % archive_interval == 0 or epoch == num_epochs - 1:
            projected = relaxation.project(latent)
            archive_solutions, archive_objectives = _update_archive(
                problem, projected, archive_solutions, max_size=archive_size
            )
        if epoch % history_stride == 0 or epoch == num_epochs - 1:
            history["epoch"].append(epoch)
            history["pareto_size"].append(
                0 if archive_solutions is None else len(archive_solutions)
            )
            history["ideal"].append(ideal.detach().cpu().tolist())
            history["nadir"].append(nadir.detach().cpu().tolist())
            history["loss"].append(float(loss.detach().item()))
        if verbose and (epoch % max(1, num_epochs // 10) == 0 or epoch == num_epochs - 1):
            size = 0 if archive_solutions is None else len(archive_solutions)
            print(f"[Pareto] epoch={epoch:5d} front={size:4d} loss={loss.item():.5g}")

    if archive_solutions is None or archive_objectives is None:
        # Constraints may be impossible. Surface the failure explicitly
        # instead of returning an apparently valid infeasible front.
        raise RuntimeError("No feasible solution was found for the Pareto archive.")

    objective_min = problem.objective_matrix(archive_solutions, minimize=True)
    order = torch.argsort(objective_min[:, 0])
    archive_solutions = archive_solutions[order]
    archive_objectives = archive_objectives[order]
    return ParetoResult(
        solutions=archive_solutions,
        objectives=archive_objectives,
        weights=weights.detach(),
        runtime=perf_counter() - started,
        objective_names=tuple(objective.name for objective in problem.objectives),
        directions=tuple(objective.direction for objective in problem.objectives),
        history=history,
    )


__all__ = ["ParetoResult", "nondominated_mask", "pareto_anneal"]
