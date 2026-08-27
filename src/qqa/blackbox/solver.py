"""Batch Bayesian-style optimisation with an adaptive RBF trust region."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from time import perf_counter

import torch

from qqa.blackbox.problem import BlackBoxProblem
from qqa.utils import require_cuda_if_requested, resolve_device


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
        for name, value in (("ridge", ridge), ("noise", noise)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0.")
        self.ridge = float(ridge)
        self.noise = float(noise)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            raise TypeError("RBF inputs and targets must be torch tensors.")
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("RBF inputs must have shape (points, dimensions) with points > 0.")
        if y.ndim not in (1, 2) or y.shape[0] != len(x):
            raise ValueError("RBF targets must have shape (points,) or (points, outputs).")
        if not x.is_floating_point() or not y.is_floating_point():
            raise TypeError("RBF inputs and targets must use a floating-point dtype.")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("RBF inputs and targets must be finite.")
        self.scalar_output = y.ndim == 1
        if self.scalar_output:
            y = y[:, None]
        self.x = x
        self.mean = y.mean(dim=0)
        self.std = y.std(dim=0, correction=0).clamp_min(1e-8)
        target = (y - self.mean) / self.std
        distances = torch.pdist(x)
        positive = distances[distances > 1e-10]
        self.lengthscale = (
            positive.median().clamp(0.05, 1.0) if positive.numel() else x.new_tensor(0.2)
        )
        squared = torch.cdist(x, x).square()
        kernel = torch.exp(-0.5 * squared / self.lengthscale.square())
        eye = torch.eye(len(x), dtype=x.dtype, device=x.device)
        # Starting at exactly zero makes ``jitter *= 10`` remain zero forever
        # after a failed Cholesky factorisation. A dtype-scale floor lets a
        # duplicate or nearly duplicate design recover deterministically.
        jitter = max(self.ridge + self.noise**2, torch.finfo(x.dtype).eps)
        for _ in range(6):
            factor, info = torch.linalg.cholesky_ex(kernel + jitter * eye)
            if int(info.max().item()) == 0:
                break
            jitter *= 10
        else:
            raise RuntimeError("RBF surrogate kernel is numerically singular.")
        self.factor = factor
        self.alpha = torch.cholesky_solve(target, factor)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squared = torch.cdist(x, self.x).square()
        cross = torch.exp(-0.5 * squared / self.lengthscale.square())
        mean = (cross @ self.alpha) * self.std + self.mean
        solved = torch.linalg.solve_triangular(self.factor, cross.T, upper=False)
        variance = (1.0 - solved.square().sum(dim=0)).clamp_min(1e-10)
        std = variance.sqrt()[:, None] * self.std
        if self.scalar_output:
            return mean[:, 0], std[:, 0]
        return mean, std


class _RFFSurrogate:
    """Bounded-cost random Fourier feature surrogate for larger campaigns."""

    def __init__(self, *, ridge: float, features: int, seed: int):
        self.ridge = max(float(ridge), 1e-10)
        self.features = int(features)
        self.seed = int(seed)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(2.0 / self.features) * torch.cos(x @ self.frequency + self.phase)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.scalar_output = y.ndim == 1
        targets = y[:, None] if self.scalar_output else y
        generator = torch.Generator(device=x.device)
        generator.manual_seed(self.seed)
        self.frequency = (
            torch.randn(
                (x.shape[1], self.features), device=x.device, dtype=x.dtype, generator=generator
            )
            / 0.2
        )
        self.phase = (
            2
            * math.pi
            * torch.rand(self.features, device=x.device, dtype=x.dtype, generator=generator)
        )
        features = self._transform(x)
        self.mean = targets.mean(dim=0)
        self.std = targets.std(dim=0, correction=0).clamp_min(1e-8)
        normalised = (targets - self.mean) / self.std
        gram = features.T @ features
        gram.diagonal().add_(self.ridge)
        self.factor = torch.linalg.cholesky(gram)
        self.weights = torch.cholesky_solve(features.T @ normalised, self.factor)
        residual = normalised - features @ self.weights
        self.noise = residual.square().mean(dim=0).clamp_min(1e-8).sqrt()

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._transform(x)
        mean = (features @ self.weights) * self.std + self.mean
        solved = torch.linalg.solve_triangular(self.factor, features.T, upper=False)
        leverage = solved.square().sum(dim=0).clamp_min(1e-8).sqrt()[:, None]
        std = (self.noise * self.std)[None, :] * (1.0 + leverage)
        if self.scalar_output:
            return mean[:, 0], std[:, 0]
        return mean, std


def _surrogate(
    kind: str,
    *,
    ridge: float,
    noise: float,
    features: int,
    seed: int,
):
    return (
        _RBFSurrogate(ridge=ridge, noise=noise)
        if kind == "rbf"
        else _RFFSurrogate(ridge=ridge, features=features, seed=seed)
    )


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


def _unseen_indices(observed: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Return first unseen candidate rows without per-point host hashing."""
    if len(candidates) == 0:
        return torch.empty(0, dtype=torch.long, device=candidates.device)
    observed = observed.to(candidates)
    combined = torch.cat([observed, candidates], dim=0)
    unique, inverse = torch.unique(combined, dim=0, return_inverse=True)
    group_count = len(unique)
    occupied = torch.zeros(group_count, dtype=torch.bool, device=candidates.device)
    if len(observed):
        occupied[inverse[: len(observed)]] = True
    groups = inverse[len(observed) :]
    positions = torch.arange(len(candidates), device=candidates.device)
    first = torch.full((group_count,), len(candidates), dtype=torch.long, device=candidates.device)
    first.scatter_reduce_(0, groups, positions, reduce="amin", include_self=True)
    keep = (~occupied[groups]) & (positions == first[groups])
    return torch.nonzero(keep, as_tuple=False).reshape(-1)


def _fingerprint(signature: dict[str, object]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evaluate_cached(
    problem: BlackBoxProblem,
    values: torch.Tensor,
    *,
    workers: int,
    database,
    fingerprint: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    if database is None:
        return problem.evaluate_batch(values, workers=workers)
    from qqa.blackbox.evaluation import (  # noqa: PLC0415
        AsynchronousEvaluationScheduler,
        EvaluationStatus,
    )

    with AsynchronousEvaluationScheduler(
        problem,
        workers=workers,
        database=database,
        problem_fingerprint=fingerprint,
        seed=seed,
    ) as scheduler:
        futures = [
            scheduler.submit(row, worker=index % workers) for index, row in enumerate(values)
        ]
        records = [future.result() for future in futures]
    failed = [record for record in records if record.status is not EvaluationStatus.COMPLETED]
    if failed:
        categories = sorted({record.exception_category or record.status.value for record in failed})
        raise RuntimeError(f"Black-box evaluation failed ({', '.join(categories)}).")
    objectives = torch.tensor([record.objective for record in records], dtype=torch.float64)
    violations = (
        torch.tensor([record.violations for record in records], dtype=torch.float64)
        if problem.constraints
        else torch.zeros((len(records), 0), dtype=torch.float64)
    )
    return objectives, violations, [problem._named_point(row) for row in values]


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


def _trust_region_centres(
    x: torch.Tensor,
    values: torch.Tensor,
    violations: torch.Tensor,
    *,
    direction: str,
    count: int,
) -> torch.Tensor:
    """Choose quality-ranked, spatially diverse observed centres."""
    total = _total_violation(violations)
    feasible = total <= 1e-10
    sign = 1.0 if direction == "min" else -1.0
    order = sorted(
        range(len(x)),
        key=lambda index: (
            not bool(feasible[index]),
            float(total[index]),
            float(sign * values[index]),
            index,
        ),
    )
    chosen = [order[0]]
    while len(chosen) < min(count, len(order)):
        remaining = torch.as_tensor(
            [index for index in order if index not in chosen], device=x.device
        )
        if not len(remaining):
            break
        separation = torch.cdist(x[remaining], x[chosen]).amin(dim=1)
        quality_rank = torch.linspace(1.0, 0.5, steps=len(remaining), device=x.device)
        chosen.append(int(remaining[torch.argmax(separation * quality_rank)].item()))
    return x[chosen]


def _candidate_set_multi(
    *,
    dimension: int,
    count: int,
    centres: torch.Tensor,
    radii: torch.Tensor,
    iteration: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(
        dimension, scramble=True, seed=seed + 7919 * (iteration + 1)
    )
    global_count = count // 3
    global_points = engine.draw(global_count).to(device=device, dtype=dtype)
    local_count = count - global_count
    base = engine.draw(local_count).to(device=device, dtype=dtype)
    region = torch.arange(local_count, device=device) % len(centres)
    noise = (base - 0.5) * (2 * radii[region, None])
    local_points = (centres[region] + noise).clamp(0.0, 1.0)
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


def _qqa_acquisition_batch(
    candidates: torch.Tensor,
    acquisition: torch.Tensor,
    *,
    take: int,
    radius: float,
    epochs: int,
    replicas: int,
) -> list[int]:
    """Select a diverse candidate subset through an opt-in QQA QUBO."""
    from qqa.annealing import anneal  # noqa: PLC0415
    from qqa.compile import SparseQUBO  # noqa: PLC0415
    from qqa.engines.qqa import SparseQUBOProblem  # noqa: PLC0415
    from qqa.repair import exact_k_projection  # noqa: PLC0415

    shortlist_size = min(128, len(candidates))
    shortlist = torch.topk(acquisition, k=shortlist_size, largest=False).indices
    points = candidates[shortlist]
    scores = acquisition[shortlist]
    scores = (scores - scores.min()) / (scores.max() - scores.min()).clamp_min(1e-12)
    left, right = torch.triu_indices(shortlist_size, shortlist_size, offset=1, device=points.device)
    distance = torch.linalg.vector_norm(points[left] - points[right], dim=1)
    diversity_cost = torch.exp(-distance.square() / max(radius**2, 1e-8))
    cardinality_penalty = 2.0
    linear = scores + cardinality_penalty * (1 - 2 * take)
    edge_weight = 2 * cardinality_penalty + diversity_cost
    problem = SparseQUBOProblem(
        SparseQUBO(linear, torch.stack((left, right)), edge_weight),
        name="blackbox-acquisition",
    )
    result = anneal(
        problem,
        sol_size=replicas,
        num_epochs=epochs,
        learning_rate=0.08,
        device=points.device,
        polish=True,
        verbose=False,
    )
    priority = 2.0 * result.best_sol.to(scores) - scores
    chosen = exact_k_projection(priority, min(take, shortlist_size)) > 0.5
    return shortlist[chosen].detach().cpu().tolist()


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
    acquisition_optimizer: str = "pool",
    qqa_acquisition_epochs: int = 60,
    qqa_acquisition_replicas: int = 16,
    constraint_weight: float = 10.0,
    initial_radius: float = 0.35,
    min_radius: float = 0.02,
    trust_regions: int = 1,
    ridge: float = 1e-6,
    noise: float = 0.0,
    max_model_points: int = 512,
    surrogate: str = "rbf",
    rff_features: int = 128,
    resume_from: BlackBoxResult | None = None,
    surrogate_dtype: str = "auto",
    evaluation_database: str | Path | object | None = None,
    problem_fingerprint: str | None = None,
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
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if budget < 2:
        raise ValueError("budget must be >= 2.")
    if (
        isinstance(candidate_pool, bool)
        or not isinstance(candidate_pool, int)
        or candidate_pool < max(32, batch_size)
    ):
        raise ValueError("candidate_pool must be an integer >= max(32, batch_size).")
    if acquisition not in {"expected_improvement", "lcb"}:
        raise ValueError("acquisition must be 'expected_improvement' or 'lcb'.")
    if acquisition_optimizer not in {"pool", "qqa"}:
        raise ValueError("acquisition_optimizer must be 'pool' or 'qqa'.")
    for name, integer_value in (
        ("qqa_acquisition_epochs", qqa_acquisition_epochs),
        ("qqa_acquisition_replicas", qqa_acquisition_replicas),
    ):
        if (
            isinstance(integer_value, bool)
            or not isinstance(integer_value, int)
            or integer_value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if (
        isinstance(max_model_points, bool)
        or not isinstance(max_model_points, int)
        or max_model_points < 16
    ):
        raise ValueError("max_model_points must be an integer >= 16.")
    if surrogate_dtype not in {"auto", "float32", "float64"}:
        raise ValueError("surrogate_dtype must be 'auto', 'float32', or 'float64'.")
    if surrogate not in {"rbf", "rff"}:
        raise ValueError("surrogate must be 'rbf' or 'rff'.")
    if isinstance(rff_features, bool) or not isinstance(rff_features, int) or rff_features < 8:
        raise ValueError("rff_features must be an integer >= 8.")
    for name, numeric_value, minimum_value in (
        ("exploration", exploration, 0.0),
        ("constraint_weight", constraint_weight, 0.0),
        ("initial_radius", initial_radius, 0.0),
        ("min_radius", min_radius, 0.0),
        ("ridge", ridge, 0.0),
        ("noise", noise, 0.0),
    ):
        if (
            isinstance(numeric_value, bool)
            or not isinstance(numeric_value, Real)
            or not math.isfinite(numeric_value)
            or numeric_value < minimum_value
        ):
            raise ValueError(f"{name} must be finite and >= {minimum_value}.")
    if initial_radius <= 0 or initial_radius > 1:
        raise ValueError("initial_radius must be in (0, 1].")
    if min_radius <= 0 or min_radius > initial_radius:
        raise ValueError("min_radius must be in (0, initial_radius].")
    if isinstance(trust_regions, bool) or not isinstance(trust_regions, int) or trust_regions < 1:
        raise ValueError("trust_regions must be a positive integer.")

    device = resolve_device(device)
    require_cuda_if_requested(device)
    compute_device = torch.device(device)
    if surrogate_dtype == "auto":
        dtype = torch.float32 if compute_device.type == "cuda" else torch.float64
    else:
        dtype = torch.float32 if surrogate_dtype == "float32" else torch.float64
    dimension = problem.space.dimension
    problem_signature = _problem_signature(problem)
    database = None
    if evaluation_database is not None:
        from qqa.blackbox.evaluation import EvaluationDatabase  # noqa: PLC0415

        if isinstance(evaluation_database, EvaluationDatabase):
            database = evaluation_database
        elif isinstance(evaluation_database, (str, Path)):
            database = EvaluationDatabase(evaluation_database)
        else:
            raise TypeError("evaluation_database must be a path, EvaluationDatabase, or None.")
    fingerprint = problem_fingerprint or _fingerprint(problem_signature)
    if initial_points is None:
        initial_points = min(budget, max(4 * (dimension + 1), 2 * batch_size))
    if (
        isinstance(initial_points, bool)
        or not isinstance(initial_points, int)
        or not 2 <= initial_points <= budget
    ):
        raise ValueError("initial_points must be an integer in [2, budget].")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer in [0, 2147483647].")

    started = perf_counter()
    if resume_from is None:
        initial_engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
        latent, physical = _project_latent(
            problem,
            initial_engine.draw(initial_points).to(device=compute_device, dtype=dtype),
        )
        latent = torch.unique(latent, dim=0)
        physical = problem.space.project(latent)
        y, violations, _ = _evaluate_cached(
            problem,
            physical,
            workers=workers,
            database=database,
            fingerprint=fingerprint,
            seed=seed,
        )
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
        if not torch.isfinite(y_obs).all() or not torch.isfinite(v_obs).all():
            raise ValueError("resume_from observations must be finite.")
        if torch.any(v_obs < 0):
            raise ValueError("resume_from violations must be non-negative.")
        history = {key: list(value) for key, value in resume_from.history.items()}
        for key in (
            "evaluations",
            "best_value",
            "best_violation",
            "trust_radius",
            "feasible_count",
        ):
            history.setdefault(key, [])
    if resume_from is not None and history["trust_radius"]:
        try:
            previous_radius = float(history["trust_radius"][-1])
        except (TypeError, ValueError) as exc:
            raise ValueError("resume_from trust-radius history is invalid.") from exc
        if not math.isfinite(previous_radius) or previous_radius <= 0:
            raise ValueError("resume_from trust-radius history is invalid.")
        radius = min(1.0, max(min_radius, previous_radius))
    else:
        radius = initial_radius
    radii = torch.full((trust_regions,), radius, dtype=dtype, device=compute_device)
    success_streak = [0] * trust_regions
    failure_streak = [0] * trust_regions
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
        objective_model = _surrogate(
            surrogate,
            ridge=ridge,
            noise=noise,
            features=rff_features,
            seed=seed + iteration,
        )
        objective_model.fit(model_x, model_y)
        violation_model = None
        total_v = _total_violation(v_obs)
        feasible_count = int((total_v <= 1e-10).sum().item())
        enough_feasible = feasible_count >= min(max(3, batch_size), len(y_obs))
        if problem.constraints:
            violation_model = _surrogate(
                surrogate,
                ridge=ridge,
                noise=noise,
                features=rff_features,
                seed=seed + 104729 + iteration,
            )
            violation_model.fit(
                model_x,
                torch.log1p(v_obs).to(device=compute_device, dtype=dtype)[model_indices],
            )

        if trust_regions == 1:
            centres = x_obs[best_before : best_before + 1].to(device=compute_device, dtype=dtype)
            pool_latent = _candidate_set(
                dimension=dimension,
                count=candidate_pool,
                incumbent=centres[0],
                radius=float(radii[0].item()),
                iteration=iteration,
                seed=seed,
                device=compute_device,
                dtype=dtype,
            )
        else:
            centres = _trust_region_centres(
                x_obs.to(device=compute_device, dtype=dtype),
                y_obs.to(device=compute_device, dtype=dtype),
                v_obs.to(device=compute_device, dtype=dtype),
                direction=problem.direction,
                count=trust_regions,
            )
            pool_latent = _candidate_set_multi(
                dimension=dimension,
                count=candidate_pool,
                centres=centres,
                radii=radii[: len(centres)],
                iteration=iteration,
                seed=seed,
                device=compute_device,
                dtype=dtype,
            )
        pool_latent, pool_physical = _project_latent(problem, pool_latent)
        unseen_indices = _unseen_indices(
            p_obs.to(device=pool_physical.device, dtype=pool_physical.dtype), pool_physical
        )
        if not len(unseen_indices):
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
                # Under the surrogate's conditional-independence
                # approximation, log joint feasibility is the sum of the
                # per-constraint log probabilities. Retaining each
                # constraint separately avoids a low aggregate violation
                # hiding one systematically difficult engineering limit.
                acquisition_values = (
                    acquisition_values - constraint_weight * probability_feasible.log().sum(dim=1)
                )
            else:
                # Build a minimally useful feasible design before allowing the
                # objective's units to dominate a single lucky feasible point.
                acquisition_values = violation_mean.sum(dim=1) - exploration * (
                    violation_std.square().sum(dim=1).sqrt()
                )

        take = min(batch_size, budget - len(y_obs), len(pool_latent))
        if acquisition_optimizer == "qqa" and take > 1:
            selected = _qqa_acquisition_batch(
                pool_latent,
                acquisition_values,
                take=take,
                radius=float(radii.max().item()),
                epochs=qqa_acquisition_epochs,
                replicas=qqa_acquisition_replicas,
            )
        else:
            selected = []
            working = acquisition_values.clone()
            for _ in range(take):
                index = int(torch.argmin(working).item())
                selected.append(index)
                distance = torch.cdist(pool_latent, pool_latent[index : index + 1]).squeeze(1)
                # Local penalisation prevents a parallel batch from
                # collapsing onto one surrogate optimum.
                working = working + torch.exp(
                    -distance.square() / max(float(radii.max().item()) ** 2, 1e-8)
                )
                working[selected] = float("inf")

        new_latent = pool_latent[selected]
        new_physical = pool_physical[selected]
        new_y, new_v, _ = _evaluate_cached(
            problem,
            new_physical,
            workers=workers,
            database=database,
            fingerprint=fingerprint,
            seed=seed,
        )
        x_obs = torch.cat([x_obs, new_latent.cpu()])
        p_obs = torch.cat([p_obs, new_physical.cpu()])
        y_obs = torch.cat([y_obs, new_y])
        v_obs = torch.cat([v_obs, new_v])

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
        best_point = x_obs[best_after].to(device=compute_device, dtype=dtype)
        region = int(torch.argmin(torch.linalg.vector_norm(centres - best_point, dim=1)).item())
        if improved:
            success_streak[region] += 1
            failure_streak[region] = 0
            if success_streak[region] >= 2:
                radii[region] = min(1.0, float(radii[region].item()) * 1.5)
                success_streak[region] = 0
        else:
            failure_streak[region] += 1
            success_streak[region] = 0
            if failure_streak[region] >= 3:
                radii[region] = max(min_radius, float(radii[region].item()) * 0.5)
                failure_streak[region] = 0
        radius = float(radii.max().item())

        total_v_now = _total_violation(v_obs)
        history["evaluations"].append(len(y_obs))
        history["best_value"].append(float(y_obs[best_after].item()))
        history["best_violation"].append(float(total_v_now[best_after].item()))
        history["trust_radius"].append(radius)
        history.setdefault("trust_radii", []).append(radii.detach().cpu().tolist())
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
            "acquisition_optimizer": acquisition_optimizer,
            "device": str(compute_device),
            "surrogate_dtype": str(dtype).removeprefix("torch."),
            "surrogate": surrogate,
            "rff_features": rff_features if surrogate == "rff" else None,
            "trust_regions": trust_regions,
            "max_model_points": max_model_points,
            "resumed": resume_from is not None,
            "prior_runtime": 0.0 if resume_from is None else resume_from.runtime,
            "cumulative_runtime": runtime if resume_from is None else resume_from.runtime + runtime,
            "problem_signature": problem_signature,
            "evaluation_cache": database is not None,
        },
    )


__all__ = ["BlackBoxResult", "blackbox_optimize"]
