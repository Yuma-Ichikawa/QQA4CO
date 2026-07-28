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
    metadata: dict[str, object] = field(default_factory=dict)

    def to_frame(self, problem: BlackBoxProblem):
        """Return every evaluated point as an analysis-ready DataFrame."""
        if not isinstance(problem, BlackBoxProblem):
            raise TypeError("problem must be a BlackBoxProblem.")
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - core currently includes pandas
            raise ImportError("Install pandas or `qqa[plotly]` to create a DataFrame.") from exc
        named = problem.space.unpack(self.points)
        data: dict[str, object] = {}
        for variable in problem.variables:
            values = named[variable.name].detach().cpu()
            if variable.size == 1:
                data[variable.name] = values.numpy()
            else:
                for index in range(variable.size):
                    data[f"{variable.name}[{index}]"] = values[:, index].numpy()
        data["objective"] = self.values.detach().cpu().numpy()
        for index, constraint in enumerate(problem.constraints):
            data[f"violation:{constraint.name}"] = self.violations[:, index].numpy()
        data["feasible"] = (_total_violation(self.violations) <= 1e-10).numpy()
        return pd.DataFrame(data)


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


def _problem_signature(problem: BlackBoxProblem) -> dict[str, object]:
    """Return stable metadata used to reject incompatible campaign resumes."""
    return {
        "name": problem.name,
        "direction": problem.direction,
        "variables": problem.space.describe(),
        "constraints": [
            {
                "name": constraint.name,
                "sense": constraint.sense,
                "rhs": constraint.rhs,
                "tolerance": constraint.tolerance,
                "scale": constraint.scale,
            }
            for constraint in problem.constraints
        ],
    }


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


def _normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def _expected_improvement(
    mean: torch.Tensor,
    std: torch.Tensor,
    incumbent: torch.Tensor,
) -> torch.Tensor:
    safe_std = std.clamp_min(1e-12)
    improvement = incumbent - mean
    z_score = improvement / safe_std
    density = torch.exp(-0.5 * z_score.square()) / math.sqrt(2.0 * math.pi)
    return improvement * _normal_cdf(z_score) + safe_std * density


def _model_subset(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    incumbent: torch.Tensor,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bound cubic RBF cost while retaining local and global information."""
    if len(x) <= max_points:
        return x, y, torch.arange(len(x), device=x.device)
    local_count = max_points * 3 // 4
    distances = torch.linalg.vector_norm(x - incumbent, dim=1)
    local = torch.topk(distances, k=local_count, largest=False).indices
    anchors = (
        torch.linspace(0, len(x) - 1, steps=max_points - local_count, device=x.device)
        .round()
        .to(torch.int64)
    )
    selected = torch.unique(torch.cat([local, anchors]))
    if len(selected) < max_points:
        remaining = torch.ones(len(x), dtype=torch.bool, device=x.device)
        remaining[selected] = False
        fill = torch.where(remaining)[0][: max_points - len(selected)]
        selected = torch.cat([selected, fill])
    return x[selected], y[selected], selected


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
    acquisition: str = "expected_improvement",
    constraint_weight: float = 10.0,
    initial_radius: float = 0.35,
    min_radius: float = 0.02,
    ridge: float = 1e-6,
    noise: float = 0.0,
    max_model_points: int = 512,
    resume_from: BlackBoxResult | None = None,
    surrogate_dtype: str = "auto",
    seed: int = 0,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> BlackBoxResult:
    """Optimise a costly mixed-variable function within an evaluation budget.

    An exact RBF surrogate supplies mean and uncertainty. Candidate batches
    combine global Sobol coverage with a success-adaptive local trust region;
    expected improvement (or a lower confidence bound), probability of
    feasibility, and greedy distance penalisation keep parallel evaluations
    diverse. ``resume_from`` continues an expensive campaign without repeating
    observations, while ``max_model_points`` bounds cubic surrogate cost.
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
    if acquisition not in {"expected_improvement", "lcb"}:
        raise ValueError("acquisition must be 'expected_improvement' or 'lcb'.")
    if not isinstance(max_model_points, int) or max_model_points < 16:
        raise ValueError("max_model_points must be an integer >= 16.")
    if surrogate_dtype not in {"auto", "float32", "float64"}:
        raise ValueError("surrogate_dtype must be 'auto', 'float32', or 'float64'.")
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
    if surrogate_dtype == "auto":
        dtype = torch.float32 if compute_device.type == "cuda" else torch.float64
    else:
        dtype = torch.float32 if surrogate_dtype == "float32" else torch.float64
    dimension = problem.space.dimension
    problem_signature = _problem_signature(problem)
    if initial_points is None:
        initial_points = min(budget, max(4 * (dimension + 1), 2 * batch_size))
    if not isinstance(initial_points, int) or not 2 <= initial_points <= budget:
        raise ValueError("initial_points must be an integer in [2, budget].")

    started = perf_counter()
    if resume_from is None:
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
        history: dict[str, list] = {
            "evaluations": [],
            "best_value": [],
            "best_violation": [],
            "trust_radius": [],
            "feasible_count": [],
        }
    else:
        if not isinstance(resume_from, BlackBoxResult):
            raise TypeError("resume_from must be a BlackBoxResult.")
        previous_signature = resume_from.metadata.get("problem_signature")
        if previous_signature is not None and previous_signature != problem_signature:
            raise ValueError("resume_from was produced by a different black-box problem.")
        if resume_from.points.ndim != 2 or resume_from.points.shape[1] != dimension:
            raise ValueError("resume_from points do not match this problem's variable space.")
        if resume_from.evaluations >= budget:
            raise ValueError("budget must exceed resume_from.evaluations.")
        p_obs = resume_from.points.detach().to(device="cpu", dtype=torch.float64).clone()
        problem.space.validate(p_obs)
        x_obs = problem.space.encode(p_obs).to(dtype=dtype)
        y_obs = resume_from.values.detach().to(device="cpu", dtype=torch.float64).clone()
        v_obs = resume_from.violations.detach().to(device="cpu", dtype=torch.float64).clone()
        if resume_from.evaluations != len(p_obs) or y_obs.shape != (len(p_obs),):
            raise ValueError("resume_from observation counts are inconsistent.")
        if v_obs.shape != (len(p_obs), len(problem.constraints)):
            raise ValueError("resume_from violations do not match this problem's constraints.")
        history = {key: list(value) for key, value in resume_from.history.items()}
        for key in (
            "evaluations",
            "best_value",
            "best_violation",
            "trust_radius",
            "feasible_count",
        ):
            history.setdefault(key, [])
    seen = set(_keys(p_obs))
    radius = (
        float(history["trust_radius"][-1])
        if resume_from is not None and history["trust_radius"]
        else initial_radius
    )
    success_streak = 0
    failure_streak = 0
    iteration = 0
    initial_best, _ = _best_index(y_obs, v_obs, problem.direction)
    initial_total_v = _total_violation(v_obs)
    if not history["evaluations"] or history["evaluations"][-1] != len(y_obs):
        history["evaluations"].append(len(y_obs))
        history["best_value"].append(float(y_obs[initial_best].item()))
        history["best_violation"].append(float(initial_total_v[initial_best].item()))
        history["trust_radius"].append(radius)
        history["feasible_count"].append(int((initial_total_v <= 1e-10).sum().item()))

    while len(y_obs) < budget:
        best_before, feasible_before = _best_index(y_obs, v_obs, problem.direction)
        model_x = x_obs.to(device=compute_device, dtype=dtype)
        sign = 1.0 if problem.direction == "min" else -1.0
        signed_y = sign * y_obs.to(device=compute_device, dtype=dtype)
        model_x, model_y, model_indices = _model_subset(
            model_x,
            signed_y,
            incumbent=model_x[best_before],
            max_points=max_model_points,
        )
        objective_model = _RBFSurrogate(ridge=ridge, noise=noise)
        objective_model.fit(model_x, model_y)
        violation_model = None
        total_v = _total_violation(v_obs)
        feasible_count = int((total_v <= 1e-10).sum().item())
        enough_feasible = feasible_count >= min(max(3, batch_size), len(y_obs))
        if problem.constraints:
            violation_model = _RBFSurrogate(ridge=ridge, noise=noise)
            violation_model.fit(
                model_x,
                torch.log1p(total_v).to(device=compute_device, dtype=dtype)[model_indices],
            )

        pool_latent = _candidate_set(
            dimension=dimension,
            count=candidate_pool,
            incumbent=x_obs[best_before].to(device=compute_device, dtype=dtype),
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
        if acquisition == "lcb":
            acquisition_values = mean - exploration * std
        else:
            if feasible_before:
                incumbent_value = sign * y_obs[best_before].to(
                    device=compute_device,
                    dtype=dtype,
                )
            else:
                incumbent_value = model_y.amin()
            acquisition_values = -_expected_improvement(mean, std, incumbent_value)
        if violation_model is not None:
            violation_mean, violation_std = violation_model.predict(pool_latent)
            if enough_feasible:
                probability_feasible = _normal_cdf(
                    -violation_mean / violation_std.clamp_min(1e-12)
                ).clamp_min(1e-9)
                acquisition_values = (
                    acquisition_values - constraint_weight * probability_feasible.log()
                )
            else:
                # Build a minimally useful feasible design before allowing the
                # objective's units to dominate a single lucky feasible point.
                acquisition_values = violation_mean - exploration * violation_std

        take = min(batch_size, budget - len(y_obs), len(pool_latent))
        selected: list[int] = []
        working = acquisition_values.clone()
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
    runtime = perf_counter() - started
    return BlackBoxResult(
        best_point=problem._named_point(best_packed),
        best_value=float(y_obs[best].item()),
        best_packed=best_packed,
        feasible=feasible,
        total_violation=float(total_v[best].item()),
        evaluations=len(y_obs),
        runtime=runtime,
        points=p_obs,
        values=y_obs,
        violations=v_obs,
        history=history,
        metadata={
            "acquisition": acquisition,
            "device": str(compute_device),
            "surrogate_dtype": str(dtype).removeprefix("torch."),
            "max_model_points": max_model_points,
            "resumed": resume_from is not None,
            "prior_runtime": 0.0 if resume_from is None else resume_from.runtime,
            "cumulative_runtime": runtime if resume_from is None else resume_from.runtime + runtime,
            "problem_signature": problem_signature,
        },
    )


__all__ = ["BlackBoxResult", "blackbox_optimize"]
