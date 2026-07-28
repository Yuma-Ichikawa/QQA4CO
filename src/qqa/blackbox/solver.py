"""Batch Bayesian-style optimisation with an adaptive RBF trust region."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter

import torch

from qqa.blackbox.problem import BlackBoxProblem
from qqa.utils import require_cuda_if_requested


@dataclass(slots=True)
class BlackBoxResult:
    """Observations and incumbent from mixed-variable black-box optimisation."""

    best_point: dict
    best_value: float
    best_packed: torch.Tensor
    feasible: bool
    total_violation: float
    evaluations: int
    runtime: float
    points: torch.Tensor
    values: torch.Tensor
    violations: torch.Tensor
    history: dict[str, list] = field(default_factory=dict)


class _RBFSurrogate:
    def __init__(self, *, ridge: float, noise: float):
        self.ridge = ridge
        self.noise = noise

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.mean = y.mean()
        self.std = y.std(correction=0).clamp_min(1e-8)
        target = (y - self.mean) / self.std
        distances = torch.pdist(x)
        positive = distances[distances > 1e-10]
        self.lengthscale = (
            positive.median().clamp(0.05, 1.0) if positive.numel() else x.new_tensor(0.2)
        )
        squared = torch.cdist(x, x).square()
        kernel = torch.exp(-0.5 * squared / self.lengthscale.square())
        eye = torch.eye(len(x), dtype=x.dtype, device=x.device)
        jitter = self.ridge + self.noise**2
        for _ in range(6):
            factor, info = torch.linalg.cholesky_ex(kernel + jitter * eye)
            if int(info.max().item()) == 0:
                break
            jitter *= 10
        else:
            raise RuntimeError("RBF surrogate kernel is numerically singular.")
        self.factor = factor
        self.alpha = torch.cholesky_solve(target[:, None], factor).squeeze(1)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squared = torch.cdist(x, self.x).square()
        cross = torch.exp(-0.5 * squared / self.lengthscale.square())
        mean = (cross @ self.alpha) * self.std + self.mean
        solved = torch.linalg.solve_triangular(self.factor, cross.T, upper=False)
        variance = (1.0 - solved.square().sum(dim=0)).clamp_min(1e-10)
        return mean, variance.sqrt() * self.std


def _total_violation(violations: torch.Tensor) -> torch.Tensor:
    return violations.sum(dim=1) if violations.shape[1] else violations.new_zeros(len(violations))


def _best_index(values: torch.Tensor, violations: torch.Tensor, direction: str) -> tuple[int, bool]:
    total = _total_violation(violations)
    feasible = total <= 1e-10
    sign = 1.0 if direction == "min" else -1.0
    if feasible.any():
        candidate_indices = torch.where(feasible)[0]
        index = candidate_indices[torch.argmin(sign * values[candidate_indices])]
        return int(index.item()), True
    return int(torch.argmin(total).item()), False


def _keys(values: torch.Tensor) -> list[bytes]:
    return [bytes(row.contiguous().numpy().tobytes()) for row in values.to(torch.float64).cpu()]


def _candidate_set(
    *,
    dimension: int,
    count: int,
    incumbent: torch.Tensor,
    radius: float,
    iteration: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(
        dimension,
        scramble=True,
        seed=seed + 7919 * (iteration + 1),
    )
    global_count = count // 2
    global_points = engine.draw(global_count).to(device=device, dtype=dtype)
    local_count = count - global_count
    local_base = engine.draw(local_count).to(device=device, dtype=dtype)
    local_noise = (local_base - 0.5) * (2 * radius)
    local_points = (incumbent + local_noise).clamp(0.0, 1.0)
    return torch.cat([global_points, local_points], dim=0)


def _project_latent(
    problem: BlackBoxProblem, latent: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    physical = problem.space.project(latent)
    return problem.space.encode(physical), physical


def blackbox_optimize(
    problem: BlackBoxProblem,
    *,
    budget: int = 100,
    batch_size: int = 4,
    initial_points: int | None = None,
    workers: int = 1,
    candidate_pool: int = 4096,
    exploration: float = 2.0,
    constraint_weight: float = 10.0,
    initial_radius: float = 0.35,
    min_radius: float = 0.02,
    ridge: float = 1e-6,
    noise: float = 0.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> BlackBoxResult:
    """Optimise a costly mixed-variable function within an evaluation budget.

    An exact RBF surrogate supplies mean and uncertainty. Candidate batches
    combine global Sobol coverage with a success-adaptive local trust region;
    lower-confidence-bound acquisition and greedy distance penalisation keep
    parallel evaluations diverse.
    """
    if not isinstance(problem, BlackBoxProblem):
        raise TypeError("problem must be a BlackBoxProblem.")
    for name, value in (("budget", budget), ("batch_size", batch_size), ("workers", workers)):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if budget < 2:
        raise ValueError("budget must be >= 2.")
    if not isinstance(candidate_pool, int) or candidate_pool < max(32, batch_size):
        raise ValueError("candidate_pool must be an integer >= max(32, batch_size).")
    for name, value, lower in (
        ("exploration", exploration, 0.0),
        ("constraint_weight", constraint_weight, 0.0),
        ("initial_radius", initial_radius, 0.0),
        ("min_radius", min_radius, 0.0),
        ("ridge", ridge, 0.0),
        ("noise", noise, 0.0),
    ):
        if not math.isfinite(value) or value < lower:
            raise ValueError(f"{name} must be finite and >= {lower}.")
    if initial_radius <= 0 or initial_radius > 1:
        raise ValueError("initial_radius must be in (0, 1].")
    if min_radius <= 0 or min_radius > initial_radius:
        raise ValueError("min_radius must be in (0, initial_radius].")

    require_cuda_if_requested(device)
    compute_device = torch.device(device)
    dtype = torch.float64
    dimension = problem.space.dimension
    if initial_points is None:
        initial_points = min(budget, max(2 * (dimension + 1), batch_size))
    if not isinstance(initial_points, int) or not 2 <= initial_points <= budget:
        raise ValueError("initial_points must be an integer in [2, budget].")

    started = perf_counter()
    initial_engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    latent, physical = _project_latent(
        problem,
        initial_engine.draw(initial_points).to(device=compute_device, dtype=dtype),
    )
    latent = torch.unique(latent, dim=0)
    physical = problem.space.project(latent)
    y, violations, _ = problem.evaluate_batch(physical, workers=workers)
    x_obs = latent.cpu()
    p_obs = physical.cpu()
    y_obs = y
    v_obs = violations
    seen = set(_keys(p_obs))

    history: dict[str, list] = {
        "evaluations": [],
        "best_value": [],
        "best_violation": [],
        "trust_radius": [],
        "feasible_count": [],
    }
    radius = initial_radius
    success_streak = 0
    failure_streak = 0
    iteration = 0
    initial_best, _ = _best_index(y_obs, v_obs, problem.direction)
    initial_total_v = _total_violation(v_obs)
    history["evaluations"].append(len(y_obs))
    history["best_value"].append(float(y_obs[initial_best].item()))
    history["best_violation"].append(float(initial_total_v[initial_best].item()))
    history["trust_radius"].append(radius)
    history["feasible_count"].append(int((initial_total_v <= 1e-10).sum().item()))

    while len(y_obs) < budget:
        best_before, feasible_before = _best_index(y_obs, v_obs, problem.direction)
        model_x = x_obs.to(device=compute_device, dtype=dtype)
        sign = 1.0 if problem.direction == "min" else -1.0
        objective_model = _RBFSurrogate(ridge=ridge, noise=noise)
        objective_model.fit(model_x, sign * y_obs.to(compute_device))
        violation_model = None
        total_v = _total_violation(v_obs)
        if problem.constraints:
            violation_model = _RBFSurrogate(ridge=ridge, noise=noise)
            violation_model.fit(model_x, torch.log1p(total_v).to(compute_device))

        pool_latent = _candidate_set(
            dimension=dimension,
            count=candidate_pool,
            incumbent=model_x[best_before],
            radius=radius,
            iteration=iteration,
            seed=seed,
            device=compute_device,
            dtype=dtype,
        )
        pool_latent, pool_physical = _project_latent(problem, pool_latent)
        pool_keys = _keys(pool_physical)
        unseen_indices: list[int] = []
        local_seen: set[bytes] = set()
        for index, key in enumerate(pool_keys):
            if key not in seen and key not in local_seen:
                unseen_indices.append(index)
                local_seen.add(key)
        if not unseen_indices:
            break
        pool_latent = pool_latent[unseen_indices]
        pool_physical = pool_physical[unseen_indices]
        mean, std = objective_model.predict(pool_latent)
        acquisition = mean - exploration * std
        if violation_model is not None:
            violation_mean, violation_std = violation_model.predict(pool_latent)
            violation_ucb = (violation_mean + exploration * violation_std).clamp_min(0)
            adaptive_weight = constraint_weight * (10.0 if not feasible_before else 1.0)
            acquisition = acquisition + adaptive_weight * violation_ucb

        take = min(batch_size, budget - len(y_obs), len(pool_latent))
        selected: list[int] = []
        working = acquisition.clone()
        for _ in range(take):
            index = int(torch.argmin(working).item())
            selected.append(index)
            distance = torch.cdist(pool_latent, pool_latent[index : index + 1]).squeeze(1)
            # Local penalisation prevents a parallel batch from collapsing
            # onto one surrogate optimum.
            working = working + torch.exp(-distance.square() / max(radius**2, 1e-8))
            working[selected] = float("inf")

        new_latent = pool_latent[selected]
        new_physical = pool_physical[selected]
        new_y, new_v, _ = problem.evaluate_batch(new_physical, workers=workers)
        x_obs = torch.cat([x_obs, new_latent.cpu()])
        p_obs = torch.cat([p_obs, new_physical.cpu()])
        y_obs = torch.cat([y_obs, new_y])
        v_obs = torch.cat([v_obs, new_v])
        seen.update(_keys(new_physical))

        best_after, feasible_after = _best_index(y_obs, v_obs, problem.direction)
        improved = False
        if feasible_after and not feasible_before:
            improved = True
        elif feasible_after and feasible_before:
            before_value = sign * y_obs[best_before]
            after_value = sign * y_obs[best_after]
            improved = bool(after_value < before_value - 1e-12)
        elif not feasible_after:
            improved = bool(
                _total_violation(v_obs)[best_after] < _total_violation(v_obs)[best_before] - 1e-12
            )
        if improved:
            success_streak += 1
            failure_streak = 0
            if success_streak >= 2:
                radius = min(1.0, radius * 1.5)
                success_streak = 0
        else:
            failure_streak += 1
            success_streak = 0
            if failure_streak >= 3:
                radius = max(min_radius, radius * 0.5)
                failure_streak = 0

        total_v_now = _total_violation(v_obs)
        history["evaluations"].append(len(y_obs))
        history["best_value"].append(float(y_obs[best_after].item()))
        history["best_violation"].append(float(total_v_now[best_after].item()))
        history["trust_radius"].append(radius)
        history["feasible_count"].append(int((total_v_now <= 1e-10).sum().item()))
        if verbose:
            print(
                f"[BlackBox] evals={len(y_obs):4d}/{budget} "
                f"best={y_obs[best_after].item():.7g} "
                f"violation={total_v_now[best_after].item():.3g} radius={radius:.3g}"
            )
        iteration += 1

    best, feasible = _best_index(y_obs, v_obs, problem.direction)
    total_v = _total_violation(v_obs)
    best_packed = p_obs[best]
    return BlackBoxResult(
        best_point=problem._named_point(best_packed),
        best_value=float(y_obs[best].item()),
        best_packed=best_packed,
        feasible=feasible,
        total_violation=float(total_v[best].item()),
        evaluations=len(y_obs),
        runtime=perf_counter() - started,
        points=p_obs,
        values=y_obs,
        violations=v_obs,
        history=history,
    )


__all__ = ["BlackBoxResult", "blackbox_optimize"]
