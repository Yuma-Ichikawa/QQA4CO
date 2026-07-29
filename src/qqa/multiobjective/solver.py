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
from qqa.utils import require_cuda_if_requested, resolve_device


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

    @property
    def minimize_objectives(self) -> torch.Tensor:
        """Objectives transformed to a common minimisation convention."""
        signs = self.objectives.new_tensor(
            [1.0 if direction == "min" else -1.0 for direction in self.directions]
        )
        return self.objectives * signs

    def knee_index(self) -> int:
        """Return a scale-invariant compromise point nearest the ideal vector."""
        values = self.minimize_objectives
        lower = values.amin(dim=0)
        span = (values.amax(dim=0) - lower).clamp_min(1e-12)
        distance = ((values - lower) / span).square().sum(dim=1).sqrt()
        return int(torch.argmin(distance).item())

    def select(self, weights: list[float] | tuple[float, ...] | torch.Tensor | None = None) -> int:
        """Select a compromise by normalised weighted achievement.

        Passing no weights returns the geometric knee.  Positive weights are
        normalised automatically and work for mixed min/max directions.
        """
        if weights is None:
            return self.knee_index()
        weights = torch.as_tensor(
            weights, device=self.objectives.device, dtype=self.objectives.dtype
        )
        if weights.ndim != 1 or weights.numel() != self.objectives.shape[1]:
            raise ValueError(f"weights must contain {self.objectives.shape[1]} values.")
        if not torch.isfinite(weights).all() or torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("weights must be finite, non-negative, and not all zero.")
        weights = weights / weights.sum()
        values = self.minimize_objectives
        lower = values.amin(dim=0)
        normalised = (values - lower) / (values.amax(dim=0) - lower).clamp_min(1e-12)
        return int(torch.argmin((normalised * weights).sum(dim=1)).item())

    def hypervolume(
        self, reference_point: list[float] | tuple[float, float] | torch.Tensor
    ) -> float:
        """Return exact dominated hypervolume for a two-objective front.

        ``reference_point`` uses the original reported objective directions.
        It must be weakly worse than every point in the front.
        """
        if self.objectives.shape[1] != 2:
            raise ValueError("Exact hypervolume currently requires exactly two objectives.")
        reference = torch.as_tensor(
            reference_point,
            device=self.objectives.device,
            dtype=self.objectives.dtype,
        )
        if reference.shape != (2,) or not torch.isfinite(reference).all():
            raise ValueError("reference_point must contain two finite values.")
        signs = reference.new_tensor(
            [1.0 if direction == "min" else -1.0 for direction in self.directions]
        )
        reference = reference * signs
        values = self.minimize_objectives
        if torch.any(values > reference + 1e-8):
            raise ValueError("reference_point must be no better than every Pareto point.")
        order = torch.argsort(values[:, 0])
        ordered = values[order]
        area = ordered.new_zeros(())
        previous_y = reference[1]
        for point in ordered:
            height = (previous_y - point[1]).clamp_min(0)
            area = area + (reference[0] - point[0]).clamp_min(0) * height
            previous_y = torch.minimum(previous_y, point[1])
        return float(area.item())

    def to_frame(self, problem: MultiObjectiveProblem | None = None):
        """Return objectives and, optionally, named decision variables."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - core currently includes pandas
            raise ImportError("Install pandas or `qqa[plotly]` to create a DataFrame.") from exc
        data = {
            name: self.objectives[:, index].detach().cpu().numpy()
            for index, name in enumerate(self.objective_names)
        }
        if problem is not None:
            if not isinstance(problem, MultiObjectiveProblem):
                raise TypeError("problem must be a MultiObjectiveProblem.")
            named = self.named_solutions(problem)
            for variable in problem.variables:
                values = named[variable.name].detach().cpu()
                if variable.size == 1:
                    data[variable.name] = values.numpy()
                else:
                    for index in range(variable.size):
                        data[f"{variable.name}[{index}]"] = values[:, index].numpy()
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


def _constraint_matrices(
    problem: MultiObjectiveProblem,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed residuals, non-negative violations, and equality flags.

    Every inequality is converted to ``g(x) <= 0`` and normalised by its
    declared scale. Equalities keep their signed residual so an augmented
    Lagrange multiplier can move in either direction.
    """
    lhs = problem.constraint_values(values)
    residuals = []
    violations = []
    equality = []
    for constraint in problem.constraints:
        residual = (lhs[constraint.name] - constraint.rhs) / constraint.scale
        if constraint.sense == ">=":
            residual = -residual
        is_equality = constraint.sense == "=="
        residuals.append(residual)
        violations.append(residual.abs() if is_equality else residual.clamp_min(0.0))
        equality.append(is_equality)
    return (
        torch.stack(residuals, dim=1),
        torch.stack(violations, dim=1),
        torch.tensor(equality, device=values.device, dtype=torch.bool),
    )


def _augmented_constraint_loss(
    residuals: torch.Tensor,
    equality: torch.Tensor,
    multipliers: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """Powell–Hestenes–Rockafellar loss for mixed equality/inequality rows."""
    equality_loss = multipliers * residuals + 0.5 * rho * residuals.square()
    shifted = (residuals + multipliers / rho).clamp_min(0.0)
    inequality_loss = 0.5 * rho * shifted.square() - 0.5 * multipliers.square() / rho
    return torch.where(equality, equality_loss, inequality_loss).sum(dim=1)


def nondominated_mask(
    values: torch.Tensor,
    *,
    tolerance: float = 1e-8,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Return the Pareto-efficient rows of an all-minimisation matrix.

    Comparisons are chunked along the candidate axis.  This preserves exact
    dominance while avoiding the ``O(points²*objectives)`` temporary tensor
    that previously exhausted GPU memory on large archives.
    """
    if values.ndim != 2:
        raise ValueError("values must have shape (points, objectives).")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if values.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=values.device)
    efficient = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    right = values[None, :, :]
    for start in range(0, values.shape[0], chunk_size):
        stop = min(start + chunk_size, values.shape[0])
        left = values[start:stop, None, :]
        # right[j] dominates left[i]
        dominates = (right <= left + tolerance).all(dim=2) & (right < left - tolerance).any(dim=2)
        efficient[start:stop] = ~dominates.any(dim=1)
    return efficient


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
    dominance_chunk_size: int,
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
        keep = nondominated_mask(objective_min, chunk_size=dominance_chunk_size)
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
    dominance_chunk_size: int = 1024,
    history_stride: int = 10,
    constraint_strategy: str = "adaptive",
    penalty_growth: float = 2.0,
    penalty_progress_ratio: float = 0.8,
    restart_patience: int = 8,
    restart_fraction: float = 0.15,
    restart_jitter: float = 0.08,
    gradient_clip_norm: float | None = 100.0,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> ParetoResult:
    """Find a diverse Pareto front in one GPU-parallel optimisation run.

    A Powell–Hestenes–Rockafellar augmented Lagrangian treats inequalities
    with projected non-negative multipliers and equalities with signed
    multipliers. When the archive stagnates, weak non-anchor replicas are
    split between archive-centred and global restarts while the nondominated
    archive and objective-axis reference directions are preserved.
    """
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
        ("dominance_chunk_size", dominance_chunk_size),
        ("history_stride", history_stride),
        ("restart_patience", restart_patience),
    ):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if constraint_strategy not in {"adaptive", "fixed"}:
        raise ValueError("constraint_strategy must be 'adaptive' or 'fixed'.")
    if not math.isfinite(penalty_growth) or penalty_growth < 1:
        raise ValueError("penalty_growth must be finite and >= 1.")
    if not math.isfinite(penalty_progress_ratio) or not 0 < penalty_progress_ratio <= 1:
        raise ValueError("penalty_progress_ratio must be in (0, 1].")
    if not math.isfinite(restart_fraction) or not 0 <= restart_fraction < 1:
        raise ValueError("restart_fraction must be in [0, 1).")
    if not math.isfinite(restart_jitter) or not 0 <= restart_jitter <= 1:
        raise ValueError("restart_jitter must be in [0, 1].")
    if gradient_clip_norm is not None and (
        not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0
    ):
        raise ValueError("gradient_clip_norm must be finite and > 0 or None.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and >= 0.")

    device = resolve_device(device)
    require_cuda_if_requested(device)
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)

    started = perf_counter()
    relaxation = problem.relaxation
    latent = relaxation.init(sol_size, problem, device)
    optimizer = torch.optim.AdamW([latent], lr=learning_rate, weight_decay=weight_decay)
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
    # Every reference direction defines a different scalar subproblem, so its
    # KKT multipliers are distinct. A shared multiplier can cancel opposite
    # equality residuals across replicas and then compensate by exploding ρ.
    multipliers = latent.new_zeros((sol_size, len(problem.constraints)))
    penalty_rho = 1.0
    previous_violation_norm: float | None = None
    best_archive_size = 0
    best_archive_ideal: torch.Tensor | None = None
    stagnation_intervals = 0
    restart_count = 0
    history: dict[str, list] = {
        "epoch": [],
        "pareto_size": [],
        "ideal": [],
        "nadir": [],
        "loss": [],
        "feasible_ratio": [],
        "mean_violation": [],
        "penalty_rho": [],
        "restarts": [],
    }

    projected = relaxation.project(latent)
    archive_solutions, archive_objectives = _update_archive(
        problem,
        projected,
        archive_solutions,
        max_size=archive_size,
        dominance_chunk_size=dominance_chunk_size,
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
        if problem.constraints:
            residual_matrix, violation_matrix, equality = _constraint_matrices(problem, values)
            tolerances = values.new_tensor(
                [constraint.tolerance / constraint.scale for constraint in problem.constraints]
            )
            feasible_ratio = float(
                (violation_matrix <= tolerances).all(dim=1).float().mean().item()
            )
            mean_violation_value = float(violation_matrix.mean().detach().item())
            if constraint_strategy == "adaptive":
                constraint_loss = _augmented_constraint_loss(
                    residual_matrix,
                    equality,
                    multipliers,
                    penalty_rho,
                )
            else:
                constraint_loss = problem.constraint_penalty(values)
        else:
            residual_matrix = values.new_zeros((len(values), 0))
            violation_matrix = values.new_zeros((len(values), 0))
            equality = torch.zeros(0, device=values.device, dtype=torch.bool)
            feasible_ratio = 1.0
            mean_violation_value = 0.0
            constraint_loss = values.new_zeros(len(values))
        discrete_penalty = relaxation.penalty(latent, curve_rate)
        diversity = relaxation.diversity(latent)
        bg = float(schedule(epoch, num_epochs))
        loss = (scalar + constraint_loss + bg * discrete_penalty).sum()
        loss = loss - div_param * sol_size * diversity
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([latent], gradient_clip_norm)
        optimizer.step()
        relaxation.perturb_(latent, learning_rate, temp)

        if (epoch + 1) % archive_interval == 0 or epoch == num_epochs - 1:
            projected = relaxation.project(latent)
            archive_solutions, archive_objectives = _update_archive(
                problem,
                projected,
                archive_solutions,
                max_size=archive_size,
                dominance_chunk_size=dominance_chunk_size,
            )
            if problem.constraints and constraint_strategy == "adaptive":
                with torch.no_grad():
                    updated_values = relaxation.forward(latent)
                    updated_residuals, updated_violations, equality = _constraint_matrices(
                        problem,
                        updated_values,
                    )
                    candidate_multipliers = multipliers + penalty_rho * updated_residuals
                    multipliers = torch.where(
                        equality,
                        candidate_multipliers.clamp(-1e6, 1e6),
                        candidate_multipliers.clamp(0.0, 1e6),
                    )
                    # Penalty growth should target the declared feasible set,
                    # not chase residuals that are already within engineering
                    # tolerances. Otherwise noisy equality rows can double ρ
                    # indefinitely even when almost the entire population is
                    # reportably feasible.
                    excess = (updated_violations - tolerances).clamp_min(0.0)
                    violation_norm = float(excess.mean(dim=0).amax().item())
                if (
                    previous_violation_norm is not None
                    and violation_norm > 1e-6
                    and violation_norm > penalty_progress_ratio * previous_violation_norm + 1e-10
                ):
                    penalty_rho = min(1e6, penalty_rho * penalty_growth)
                previous_violation_norm = violation_norm

            current_archive_size = 0 if archive_solutions is None else len(archive_solutions)
            quality_improved = current_archive_size > best_archive_size
            if archive_solutions is not None:
                current_archive_ideal = problem.objective_matrix(
                    archive_solutions,
                    minimize=True,
                ).amin(dim=0)
                if best_archive_ideal is None or bool(
                    torch.any(current_archive_ideal < best_archive_ideal - 1e-7)
                ):
                    quality_improved = True
                best_archive_ideal = (
                    current_archive_ideal
                    if best_archive_ideal is None
                    else torch.minimum(best_archive_ideal, current_archive_ideal)
                )
            best_archive_size = max(best_archive_size, current_archive_size)
            if quality_improved:
                stagnation_intervals = 0
            else:
                stagnation_intervals += 1
            if (
                restart_fraction > 0
                and stagnation_intervals >= restart_patience
                and sol_size > problem.num_objectives
            ):
                candidates = torch.arange(
                    problem.num_objectives,
                    sol_size,
                    device=latent.device,
                )
                count = min(
                    len(candidates),
                    max(1, math.ceil(sol_size * restart_fraction)),
                )
                merit = (
                    scalar.detach()
                    + constraint_loss.detach()
                    + max(bg, 0.0) * discrete_penalty.detach()
                )
                worst = candidates[torch.topk(merit[candidates], k=count, largest=True).indices]
                with torch.no_grad():
                    replacements = torch.rand_like(latent[worst])
                    if archive_solutions is not None:
                        elite_count = min((count + 1) // 2, len(archive_solutions))
                        selected = torch.randperm(
                            len(archive_solutions),
                            device=latent.device,
                        )[:elite_count]
                        elite = relaxation.encode(archive_solutions[selected])
                        elite = elite + restart_jitter * torch.randn_like(elite)
                        replacements[:elite_count] = elite
                    latent[worst] = replacements
                    relaxation.perturb_(latent, learning_rate, 0.0)
                    state = optimizer.state.get(latent, {})
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        value = state.get(key)
                        if torch.is_tensor(value) and value.shape == latent.shape:
                            value[worst].zero_()
                    # A new primal point must not inherit the KKT estimate of
                    # the discarded reference-direction trajectory.
                    multipliers[worst].zero_()
                restart_count += count
                stagnation_intervals = 0
        if epoch % history_stride == 0 or epoch == num_epochs - 1:
            history["epoch"].append(epoch)
            history["pareto_size"].append(
                0 if archive_solutions is None else len(archive_solutions)
            )
            history["ideal"].append(ideal.detach().cpu().tolist())
            history["nadir"].append(nadir.detach().cpu().tolist())
            history["loss"].append(float(loss.detach().item()))
            history["feasible_ratio"].append(feasible_ratio)
            history["mean_violation"].append(mean_violation_value)
            history["penalty_rho"].append(penalty_rho)
            history["restarts"].append(restart_count)
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
