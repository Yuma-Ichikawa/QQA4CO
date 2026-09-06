"""Unified Quasi-Quantum Annealing loop.

This module replaces the four legacy ``batch_annealing_*`` functions from the
original repository with a single :func:`anneal` routine that delegates
problem-specific behaviour to :mod:`qqa.relaxation` and :mod:`qqa.callbacks`.

Single-instance binary problems, batched-instance problems, and categorical
problems all share this same loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, fields, is_dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

from qqa.callbacks import Callback, CallbackState, HistoryRecorder
from qqa.polish import apply_polish_if_improves
from qqa.problems.base import COProblem
from qqa.relaxation import _default_penalty_from_forward
from qqa.schedule import LinearBGSchedule, Schedule
from qqa.utils import require_cuda_if_requested, resolve_device, safe_score_summary


@dataclass
class AnnealResult:
    """Result returned by :func:`anneal`.

    Attributes
    ----------
    best_sol:
        Tensor of the best discrete solution(s) found during annealing. Shape
        depends on the problem: ``(N, ...)`` for one winning single-instance
        state, or ``(num_instance, max_node)`` for batched-instance problems.
    best_obj:
        Best objective value observed. ``float`` for single-instance problems,
        ``numpy.ndarray`` of shape ``(num_instance,)`` for batched-instance.
    runtime:
        Wall-clock time of the annealing loop in seconds.
    history:
        Dict of per-epoch metrics (``loss_mean``, ``penalty_mean``,
        ``diversity``, ``bg``). Empty if ``record_history=False``.
    callbacks:
        List of callback instances that were active. Useful for retrieving
        e.g. ``TrajectoryTracker.values``.
    """

    best_sol: torch.Tensor
    best_obj: Any
    runtime: float
    history: dict = field(default_factory=dict)
    callbacks: list[Callback] = field(default_factory=list)
    score: dict = field(default_factory=dict)
    """Human-readable problem-specific score produced by
    :py:meth:`COProblem.score_summary`.

    * **Single-instance**: standard dict
      ``{label, value, unit, feasible, extra}`` with scalar fields.
    * **Batched-instance** (``problem.num_instance > 1``): same keys, but
      ``value`` and ``feasible`` are ``np.ndarray`` of length
      ``num_instance``, and ``extra`` carries arrays plus a
      ``feasible_count`` tally. ``score`` is empty for batched problems
      whose class did not override :py:meth:`COProblem.score_summary`."""
    polished_sol: torch.Tensor | None = None
    """Domain-locally-optimal version of :attr:`best_sol`, populated when
    :func:`anneal` is called with ``polish=True`` (the default) on a QUBO,
    quadratic-spin, or categorical problem. ``best_sol`` / ``best_obj`` /
    ``score`` are replaced only after a strict improvement."""
    final_population: torch.Tensor | None = None
    """Projected final replica population, populated only when
    :func:`anneal` is called with ``return_population=True``. Hybrid solvers
    use it to pass several diverse QQA incumbents to exact solvers without
    making ordinary results unnecessarily large."""
    diagnostics: dict = field(default_factory=dict)
    """Solver-level diagnostics such as adaptive restart counts and the
    numerical-stability controls used for the run."""
    archive: Any = None
    """Historical feasibility/quality/diversity archive retained across epochs."""


class _LightweightAdamW:
    """Single-parameter AdamW without ``torch.optim``'s lazy import cost."""

    def __init__(
        self,
        parameter: torch.Tensor,
        *,
        learning_rate: float,
        weight_decay: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.parameter = parameter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.steps = 0
        self.state = {
            parameter: {
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
        }

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        if set_to_none:
            self.parameter.grad = None
        elif self.parameter.grad is not None:
            self.parameter.grad.zero_()

    def step(self) -> None:
        gradient = self.parameter.grad
        if gradient is None:
            return
        self.steps += 1
        state = self.state[self.parameter]
        first = state["exp_avg"]
        second = state["exp_avg_sq"]
        with torch.no_grad():
            if self.weight_decay:
                self.parameter.mul_(1.0 - self.learning_rate * self.weight_decay)
            first.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
            second.mul_(self.beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - self.beta2,
            )
            bias1 = 1.0 - self.beta1**self.steps
            bias2 = 1.0 - self.beta2**self.steps
            denominator = second.sqrt() / math.sqrt(bias2)
            denominator.add_(self.epsilon)
            self.parameter.addcdiv_(
                first,
                denominator,
                value=-self.learning_rate / bias1,
            )


class _MirrorDescentOptimizer:
    """Minimal optimizer adapter delegated to a simplex relaxation."""

    def __init__(self, parameter: torch.Tensor, relaxation: Any, *, learning_rate: float) -> None:
        mirror_step = getattr(relaxation, "mirror_step_", None)
        if not callable(mirror_step):
            raise TypeError(
                "optimizer='mirror-descent' requires a relaxation exposing mirror_step_."
            )
        self.parameter = parameter
        self.relaxation = relaxation
        self.learning_rate = float(learning_rate)
        self.state: dict[torch.Tensor, dict[str, torch.Tensor]] = {parameter: {}}

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        if set_to_none:
            self.parameter.grad = None
        elif self.parameter.grad is not None:
            self.parameter.grad.zero_()

    def step(self) -> None:
        self.relaxation.mirror_step_(self.parameter, self.learning_rate)


def _is_instance_problem(problem) -> bool:
    return hasattr(problem, "num_instance")


def _replica_merit(discrete_losses: torch.Tensor) -> torch.Tensor:
    """Return a scale-balanced scalar rank for each parallel replica."""
    if discrete_losses.ndim == 1:
        return discrete_losses
    flattened = discrete_losses.reshape(discrete_losses.shape[0], -1)
    centre = flattened.median(dim=0).values
    scale = (flattened - centre).abs().median(dim=0).values.clamp_min(1e-8)
    return ((flattened - centre) / scale).mean(dim=1)


def _lexicographic_argmin(keys: torch.Tensor) -> torch.Tensor:
    """Return a stable lexicographic row argmin without a host round-trip."""
    if keys.ndim != 2 or not len(keys):
        raise ValueError("keys must be a non-empty rank-two tensor.")
    order = torch.arange(len(keys), device=keys.device)
    for column in range(keys.shape[1] - 1, -1, -1):
        ranked = torch.argsort(keys[order, column], stable=True)
        order = order[ranked]
    return order[0]


def _replica_median(values: torch.Tensor) -> torch.Tensor:
    """Return the lower median without CUDA's nondeterministic index path."""
    if values.is_cuda and torch.are_deterministic_algorithms_enabled():
        ordered = torch.sort(values, dim=0).values
        return ordered[(ordered.shape[0] - 1) // 2]
    return values.median(dim=0).values


def _lexicographic_less(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compare two one-dimensional keys entirely on their current device."""
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Lexicographic keys must be aligned one-dimensional tensors.")
    less = torch.zeros((), dtype=torch.bool, device=left.device)
    equal = torch.ones((), dtype=torch.bool, device=left.device)
    for column in range(len(left)):
        less |= equal & (left[column] < right[column])
        equal &= left[column] == right[column]
    return less


def _schedule_checkpoint_descriptor(schedule: Schedule) -> dict[str, Any]:
    """Describe a schedule without source paths or executable payloads."""
    parameters: dict[str, Any] = {}
    mutable_state: dict[str, Any] = {}
    if is_dataclass(schedule):
        names = [item.name for item in fields(schedule)]
    else:
        names = sorted(getattr(schedule, "__dict__", {}))
    for name in names:
        value = getattr(schedule, name)
        if isinstance(value, tuple):
            value = list(value)
        target = mutable_state if name.startswith("_") else parameters
        target[name] = value
    return {
        "type": f"{type(schedule).__module__}.{type(schedule).__qualname__}",
        "parameters": parameters,
        "state": mutable_state,
    }


def _schedule_value(schedule: Schedule, epoch: int, num_epochs: int) -> float:
    """Evaluate a public schedule once and reject non-finite scalar outputs."""
    raw_value = schedule(epoch, num_epochs)
    if isinstance(raw_value, bool):
        raise ValueError("schedule must return a finite real scalar, not a boolean.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("schedule must return a finite real scalar.") from exc
    if not math.isfinite(value):
        raise ValueError("schedule must return a finite real scalar.")
    return value


def _reset_optimizer_rows(
    optimizer: Any,
    parameter: torch.Tensor,
    rows: torch.Tensor,
    enabled: torch.Tensor | None = None,
) -> None:
    """Clear Adam moments for replicas whose latent state was restarted."""
    state = optimizer.state.get(parameter, {})
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if torch.is_tensor(value) and value.shape == parameter.shape:
            if enabled is None:
                value.index_fill_(0, rows, 0)
            else:
                value[rows] = torch.where(
                    enabled,
                    torch.zeros_like(value[rows]),
                    value[rows],
                )


@torch.no_grad()
def _apply_optimizer_step_scale_(
    parameter: torch.Tensor,
    origin: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Apply role/preconditioner scaling to an optimizer's actual update."""
    if parameter.shape != origin.shape:
        raise ValueError("parameter and origin must have the same shape.")
    try:
        torch.broadcast_shapes(parameter.shape, scale.shape)
    except RuntimeError as exc:
        raise ValueError("scale must broadcast to the parameter shape.") from exc
    parameter.copy_(origin + (parameter - origin) * scale)


def _restart_replicas(
    *,
    x: torch.Tensor,
    best_sol: torch.Tensor,
    discrete_losses: torch.Tensor,
    relaxation,
    optimizer: Any,
    fraction: float,
    jitter: float,
    learning_rate: float,
    archive_centres: torch.Tensor | None = None,
    enabled: torch.Tensor | None = None,
) -> int:
    """Restart weak replicas with a global/local basin-hopping mixture."""
    count = min(x.shape[0] - 1, max(1, math.ceil(x.shape[0] * fraction)))
    if count < 1:
        return 0
    worst = torch.topk(_replica_merit(discrete_losses), k=count, largest=True).indices
    encode = getattr(relaxation, "encode", None)
    elite = best_sol.to(device=x.device, dtype=x.dtype)
    centres = (
        elite.unsqueeze(0)
        if archive_centres is None
        else archive_centres.to(device=x.device, dtype=x.dtype)
    )
    if encode is not None:
        elite = encode(elite)
        centres = encode(centres)
    if elite.shape != x.shape[1:]:
        return 0

    # Half of the weak replicas intensify around the incumbent; the rest
    # restart globally. This is a bounded basin-hopping step, not mutation of
    # the preserved incumbent itself.
    elite_count = (count + 1) // 2
    with torch.no_grad():
        replacements = torch.rand_like(x[worst])
        if elite_count:
            archive_rows = torch.arange(elite_count, device=x.device) % len(centres)
            local = centres[archive_rows].clone()
            local.add_(jitter * torch.randn_like(local))
            replacements[:elite_count] = local
        x[worst] = replacements if enabled is None else torch.where(enabled, replacements, x[worst])
        relaxation.perturb_(x, learning_rate, 0.0)
        _reset_optimizer_rows(optimizer, x, worst, enabled)
    return count


def anneal(
    problem,
    *,
    sol_size: int = 100,
    learning_rate: float = 1.0,
    temp: float = 0.0,
    schedule: Schedule | None = None,
    min_bg: float | None = None,
    max_bg: float | None = None,
    curve_rate: int = 2,
    div_param: float = 0.0,
    num_epochs: int = 10_000,
    time_limit: float | None = None,
    check_interval: int = 1000,
    device: str | torch.device = "cpu",
    callbacks: Sequence[Callback] = (),
    record_history: bool = True,
    verbose: bool = True,
    mixed_precision: Literal["fp32", "bf16"] = "fp32",
    initial_state: Any = None,
    polish: bool = True,
    return_population: bool = False,
    weight_decay: float = 0.0,
    optimizer: Literal["adamw", "lightweight-adamw", "mirror-descent"] = "adamw",
    gradient_clip_norm: float | None = None,
    restart_patience: int | None = None,
    restart_fraction: float = 0.15,
    restart_jitter: float = 0.10,
    compile_core: bool = False,
    cuda_graphs: bool = False,
    normalize_loss: bool = True,
    robust_scaling: bool = True,
    heterogeneous_replicas: bool = False,
    replica_exchange_interval: int | None = None,
    factor_preconditioning: bool = True,
    curvature_aware_beta: bool = True,
    archive_size: int = 64,
    archive_interval: int | None = None,
    checkpoint_path: str | None = None,
    checkpoint_interval: int | None = None,
    resume_from: str | None = None,
) -> AnnealResult:
    """Run Quasi-Quantum Annealing on ``problem``.

    Parameters
    ----------
    problem:
        Any :class:`~qqa.problems.COProblem` subclass. Must expose
        ``loss_fn(x)`` and a ``relaxation`` attribute.
    sol_size:
        Number of parallel candidate solutions (batch size).
    learning_rate:
        AdamW learning rate for the relaxed variable.
    temp:
        Langevin noise temperature. If ``0`` no noise is added.
    schedule:
        Callable ``(epoch, num_epochs) -> bg``. If ``None`` a
        :class:`LinearBGSchedule` is built from ``min_bg``/``max_bg``.
    min_bg, max_bg:
        Convenience override for the default linear schedule.
    curve_rate:
        Exponent of the QQA penalty (must be even for symmetric binary wells).
    div_param:
        Weight of the diversity term. Set to 0 to disable.
    num_epochs:
        Number of gradient steps.
    time_limit:
        Optional wall-clock budget in seconds. The loop checks this deadline
        before each epoch, making it suitable for sharing one total budget
        with a downstream exact solver.
    check_interval:
        How often to print progress logs.
    device:
        torch device.
    callbacks:
        Additional callbacks. A :class:`HistoryRecorder` is prepended when
        ``record_history=True``.
    record_history:
        If True, loss/penalty/diversity/bg are recorded per epoch.
    verbose:
        If True, print periodic progress.
    mixed_precision:
        If ``"bf16"`` and ``device`` is CUDA, the forward pass and loss
        evaluation run inside ``torch.amp.autocast`` with bfloat16. The
        relaxed variable, gradients, reductions, and AdamW state stay in
        float32 for numerical stability. Measure the effect on the target
        hardware and model; no fixed speedup or quality claim is assumed.
        Defaults to ``"fp32"``. On CPU this option is downgraded to ``"fp32"``.
    initial_state:
        Optional warm-start bitstring. Shape ``(N,)`` (broadcast to every
        chain with light Gaussian jitter so the population still
        diversifies) or ``(sol_size, N)`` (used verbatim). Particularly
        effective on near-bipartite MaxCut where
        :func:`qqa.warmstart.bfs_2color` already gives a 0.96+ ApR seed.
        Ignored for batched-instance problems.
    polish:
        If ``True`` (default), run a domain-aware monotone local search on the
        winning QUBO, quadratic-spin, or categorical solution and replace
        ``best_sol`` / ``best_obj`` / ``score`` only when it is strictly
        better. Unsupported and batched-instance problems are skipped.
    return_population:
        If true, retain the projected final replica population in
        :attr:`AnnealResult.final_population`. Defaults to false to keep
        result objects compact.
    weight_decay:
        AdamW decay applied to the latent coordinates. Defaults to zero:
        shrinking every coordinate toward binary zero introduces an
        objective-independent bias and is not part of the QQA dynamics.
    optimizer:
        ``"adamw"`` preserves the standard Torch implementation.
        ``"lightweight-adamw"`` uses the same single-parameter update without
        Torch optimizer discovery overhead and is intended for short hybrid
        callbacks. ``"mirror-descent"`` is an opt-in exponentiated-gradient
        update for :class:`~qqa.relaxation.MirrorDescentCategoricalRelaxation`.
    gradient_clip_norm:
        Optional global norm cap for the latent gradient. Useful for
        user-defined objectives with large coefficients or singular
        nonlinear derivatives.
    restart_patience:
        Enable adaptive basin recovery after this many consecutive epochs
        without an incumbent improvement. Weak replicas are split between
        incumbent-centred jitter and fresh global samples. ``None`` disables
        restarts and preserves the original loop exactly.
    restart_fraction:
        Fraction of replicas replaced at a restart. At least one incumbent
        replica is always retained.
    restart_jitter:
        Latent-space standard deviation around the incumbent for the local
        half of each restart.
    compile_core:
        Opt-in ``torch.compile(fullgraph=True)`` for the static objective
        evaluation. Unsupported user callables fail explicitly instead of
        silently falling back through graph breaks.
    """
    runtime_start = perf_counter()
    if not isinstance(sol_size, int) or isinstance(sol_size, bool) or sol_size < 1:
        raise ValueError(f"sol_size must be >= 1, got {sol_size}.")
    if not isinstance(num_epochs, int) or isinstance(num_epochs, bool) or num_epochs < 0:
        raise ValueError(f"num_epochs must be >= 0, got {num_epochs}.")
    if time_limit is not None and (
        isinstance(time_limit, bool) or not math.isfinite(time_limit) or time_limit < 0
    ):
        raise ValueError("time_limit must be finite and non-negative, or None.")
    if isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")
    if isinstance(temp, bool) or not math.isfinite(temp) or temp < 0:
        raise ValueError(f"temp must be >= 0, got {temp}.")
    if (
        not isinstance(check_interval, int)
        or isinstance(check_interval, bool)
        or check_interval < 1
    ):
        raise ValueError(f"check_interval must be >= 1, got {check_interval}.")
    if isinstance(div_param, bool) or not math.isfinite(div_param) or not 0.0 <= div_param <= 1.0:
        raise ValueError(f"div_param must be in [0, 1], got {div_param}.")
    if mixed_precision not in ("fp32", "bf16"):
        raise ValueError(f"mixed_precision must be 'fp32' or 'bf16', got {mixed_precision!r}.")
    if isinstance(weight_decay, bool) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(f"weight_decay must be finite and >= 0, got {weight_decay}.")
    if optimizer not in {"adamw", "lightweight-adamw", "mirror-descent"}:
        raise ValueError("optimizer must be 'adamw', 'lightweight-adamw', or 'mirror-descent'.")
    if gradient_clip_norm is not None and (
        isinstance(gradient_clip_norm, bool)
        or not math.isfinite(gradient_clip_norm)
        or gradient_clip_norm <= 0
    ):
        raise ValueError(f"gradient_clip_norm must be finite and > 0, got {gradient_clip_norm}.")
    if restart_patience is not None and (
        not isinstance(restart_patience, int)
        or isinstance(restart_patience, bool)
        or restart_patience < 1
    ):
        raise ValueError("restart_patience must be a positive integer or None.")
    if (
        isinstance(restart_fraction, bool)
        or not math.isfinite(restart_fraction)
        or not 0 < restart_fraction < 1
    ):
        raise ValueError("restart_fraction must be finite and in (0, 1).")
    if (
        isinstance(restart_jitter, bool)
        or not math.isfinite(restart_jitter)
        or not 0 <= restart_jitter <= 1
    ):
        raise ValueError("restart_jitter must be finite and in [0, 1].")
    for name, value in (
        ("record_history", record_history),
        ("verbose", verbose),
        ("polish", polish),
        ("return_population", return_population),
        ("compile_core", compile_core),
        ("cuda_graphs", cuda_graphs),
        ("normalize_loss", normalize_loss),
        ("robust_scaling", robust_scaling),
        ("heterogeneous_replicas", heterogeneous_replicas),
        ("factor_preconditioning", factor_preconditioning),
        ("curvature_aware_beta", curvature_aware_beta),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean.")
    if replica_exchange_interval is not None and (
        isinstance(replica_exchange_interval, bool)
        or not isinstance(replica_exchange_interval, int)
        or replica_exchange_interval < 1
    ):
        raise ValueError("replica_exchange_interval must be a positive integer or None.")
    if isinstance(archive_size, bool) or not isinstance(archive_size, int) or archive_size < 0:
        raise ValueError("archive_size must be a non-negative integer.")
    if archive_interval is not None and (
        isinstance(archive_interval, bool)
        or not isinstance(archive_interval, int)
        or archive_interval < 1
    ):
        raise ValueError("archive_interval must be a positive integer or None.")
    if checkpoint_interval is not None and (
        isinstance(checkpoint_interval, bool)
        or not isinstance(checkpoint_interval, int)
        or checkpoint_interval < 1
    ):
        raise ValueError("checkpoint_interval must be a positive integer or None.")
    if checkpoint_interval is not None and checkpoint_path is None:
        raise ValueError("checkpoint_interval requires checkpoint_path.")
    if resume_from is not None and initial_state is not None:
        raise ValueError("resume_from and initial_state are mutually exclusive.")
    if cuda_graphs and optimizer != "adamw":
        raise ValueError("cuda_graphs requires optimizer='adamw'.")
    if cuda_graphs and heterogeneous_replicas and replica_exchange_interval is not None:
        raise ValueError("cuda_graphs and replica exchange cannot share mutable optimizer state.")
    if optimizer == "mirror-descent" and weight_decay:
        raise ValueError("mirror-descent does not support Euclidean weight_decay.")
    # Mirror the validation that the pignn / cpra trainers already enforce:
    # the binary / spin penalty 1 - (1 - 2p)^c is asymmetric for odd c and
    # silently breaks the discrete attractor at γ > 0. CategoricalRelaxation
    # uses a different penalty form (K p - 1)^c that *is* well-defined for
    # odd c, so it is exempted.
    from qqa.relaxation import CategoricalRelaxation  # noqa: PLC0415

    if not isinstance(curve_rate, int) or isinstance(curve_rate, bool) or curve_rate < 1:
        raise ValueError(f"curve_rate must be a positive integer, got {curve_rate}.")
    if curve_rate % 2 != 0 and not isinstance(
        getattr(problem, "relaxation", None), CategoricalRelaxation
    ):
        raise ValueError(
            f"curve_rate must be a positive even integer for binary/spin "
            f"relaxations, got {curve_rate}."
        )

    device = resolve_device(device)
    # Surface a helpful message when CUDA is requested but unavailable,
    # before torch raises its own (cryptic) error deep inside .to().
    require_cuda_if_requested(device)

    _is_cuda = (isinstance(device, str) and device.startswith("cuda")) or (
        isinstance(device, torch.device) and device.type == "cuda"
    )
    if cuda_graphs and not _is_cuda:
        raise ValueError("cuda_graphs requires a CUDA device.")
    use_amp = mixed_precision == "bf16" and _is_cuda
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    )

    if schedule is None:
        schedule = LinearBGSchedule(
            -2.0 if min_bg is None else min_bg,
            0.1 if max_bg is None else max_bg,
        )
    elif not callable(schedule):
        raise TypeError("schedule must be callable or None.")

    schedule_descriptor = _schedule_checkpoint_descriptor(schedule)
    checkpoint_config = {
        "sol_size": sol_size,
        "num_epochs": num_epochs,
        "optimizer": optimizer,
        "curve_rate": curve_rate,
        "learning_rate": learning_rate,
        "temperature": temp,
        "diversity_weight": div_param,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "restart_patience": restart_patience,
        "restart_fraction": restart_fraction,
        "restart_jitter": restart_jitter,
        "mixed_precision": mixed_precision,
        "compile_core": compile_core,
        "cuda_graphs": cuda_graphs,
        "normalize_loss": normalize_loss,
        "robust_scaling": robust_scaling,
        "heterogeneous_replicas": heterogeneous_replicas,
        "replica_exchange_interval": replica_exchange_interval,
        "factor_preconditioning": factor_preconditioning,
        "curvature_aware_beta": curvature_aware_beta,
        "archive_size": archive_size,
        "archive_interval": archive_interval or max(1, num_epochs // 32),
        "check_interval": check_interval,
        "polish": polish,
        "schedule": {
            "type": schedule_descriptor["type"],
            "parameters": schedule_descriptor["parameters"],
        },
    }
    if checkpoint_path is not None or resume_from is not None:
        from qqa.runtime.security import validate_portable_payload

        validate_portable_payload(checkpoint_config)
        validate_portable_payload(schedule_descriptor["state"])

    relax = problem.relaxation
    loss_function = (
        torch.compile(problem.loss_fn, fullgraph=True) if compile_core else problem.loss_fn
    )

    cb_list: list[Callback] = []
    recorder: HistoryRecorder | None = None
    if record_history:
        recorder = HistoryRecorder()
        cb_list.append(recorder)
    cb_list.extend(callbacks)
    archive_callback = None
    if archive_size and sol_size > 1:
        from qqa.local.archive import HistoricalEliteCallback

        archive_callback = HistoricalEliteCallback(
            maximum_size=archive_size,
            interval=archive_interval or max(1, num_epochs // 32),
        )
        cb_list.append(archive_callback)

    is_batch = _is_instance_problem(problem)
    restored_checkpoint = None
    checkpoint_fingerprint = None
    resume_semantics = "fresh"
    start_epoch = 0
    if resume_from is not None or checkpoint_path is not None:
        from qqa.runtime.checkpoint import fingerprint_problem, load_checkpoint

        checkpoint_fingerprint = fingerprint_problem(problem)
        if resume_from is not None:
            restored_checkpoint = load_checkpoint(resume_from, device=device)
            if restored_checkpoint.model_fingerprint != checkpoint_fingerprint:
                raise ValueError("Checkpoint model fingerprint does not match this problem.")
            for key, expected in checkpoint_config.items():
                if key in {"num_epochs", "archive_interval"}:
                    continue
                if restored_checkpoint.config.get(key) != expected:
                    raise ValueError(f"Checkpoint {key} does not match the requested run.")
            saved_epochs = int(restored_checkpoint.config.get("num_epochs", num_epochs))
            resume_semantics = "resume" if saved_epochs == num_epochs else "extend"
            saved_schedule_state = restored_checkpoint.metadata.get("schedule_state", {})
            if set(saved_schedule_state) != set(schedule_descriptor["state"]):
                raise ValueError("Checkpoint schedule state does not match the requested schedule.")
            for name, value in saved_schedule_state.items():
                object.__setattr__(schedule, name, value)
            start_epoch = restored_checkpoint.epoch
            if not 0 <= start_epoch <= num_epochs:
                raise ValueError("Checkpoint epoch lies outside the requested run.")
    if initial_state is not None and is_batch:
        # Warm-starting batched-instance problems would require a
        # per-instance seed tensor and is rarely useful — skip silently
        # rather than impose a confusing shape contract.
        initial_state = None
    if restored_checkpoint is not None:
        if "latent" not in restored_checkpoint.tensors:
            raise ValueError("Checkpoint has no latent population tensor.")
        x = restored_checkpoint.tensors["latent"].to(device=device)
        if x.shape[0] != sol_size:
            raise ValueError("Checkpoint population size does not match sol_size.")
        x = x.detach().clone().requires_grad_(True)
        cpu_rng_state = restored_checkpoint.tensors.get("cpu_rng_state")
        if cpu_rng_state is None:
            raise ValueError("Checkpoint has no CPU random-generator state.")
        torch.random.set_rng_state(cpu_rng_state.detach().cpu())
        if torch.device(device).type == "cuda":
            device_rng_state = restored_checkpoint.tensors.get("device_rng_state")
            if device_rng_state is None:
                raise ValueError("CUDA checkpoint has no device random-generator state.")
            torch.cuda.set_rng_state(device_rng_state.detach().cpu(), device=device)
        if archive_callback is not None:
            archive_callback.restore_checkpoint_tensors(restored_checkpoint.tensors)
    elif initial_state is None:
        x = relax.init(sol_size, problem, device)
    else:
        seed_dtype = getattr(problem, "dtype", torch.float32)
        from qqa.runtime.population import WarmStateBundle, compose_warm_population

        if isinstance(initial_state, WarmStateBundle):
            composed = compose_warm_population(
                initial_state,
                replicas=sol_size,
                device=device,
                dtype=seed_dtype,
            )
            if composed is None:
                raise ValueError("WarmStateBundle contains no candidate state.")
            seed = composed
        elif torch.is_tensor(initial_state):
            seed = initial_state.detach().to(device=device, dtype=seed_dtype)
        else:
            raise TypeError("initial_state must be a tensor, WarmStateBundle, or None.")
        encode = getattr(relax, "encode", None)
        if encode is not None:
            seed = encode(seed)
        structured_shape = (
            int(getattr(problem, "num_node", 0)),
            int(getattr(problem, "num_category", 0)),
        )
        if seed.ndim == 2 and structured_shape[1] >= 2 and seed.shape == structured_shape:
            seed = seed.unsqueeze(0).expand(sol_size, -1, -1).contiguous()
        elif seed.dim() == 1:
            seed = seed.unsqueeze(0).expand(sol_size, -1).contiguous()
        if seed.shape[0] != sol_size:
            raise ValueError(
                f"initial_state.shape[0]={seed.shape[0]} but sol_size={sol_size}; "
                "supply either (N,) or (sol_size, N)."
            )
        # Inject a tiny Gaussian jitter (σ=0.05) so the chains are not bit-for-bit
        # identical at t=0 — otherwise the diversity term has nothing to spread
        # apart and the population collapses immediately. The clamp keeps the
        # seed inside the [0, 1] cube even when the user passes raw {0,1} bits.
        x = seed + 0.05 * torch.randn_like(seed)
        # Restore the *relaxation's* latent domain.  A universal clamp to
        # [0, 1] corrupts logit-based softmax/ST estimators and used to make
        # structured warm starts silently lose their intended geometry.
        relax.perturb_(x, learning_rate, 0.0)
        x.requires_grad_(True)
    optimiser: Any
    if optimizer == "adamw":
        adamw_options: dict[str, Any] = {}
        if cuda_graphs:
            adamw_options.update(capturable=True, foreach=False)
        optimiser = torch.optim.AdamW(
            [x],
            lr=learning_rate,
            weight_decay=weight_decay,
            **adamw_options,
        )
    elif optimizer == "lightweight-adamw":
        optimiser = _LightweightAdamW(
            x,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimiser = _MirrorDescentOptimizer(
            x,
            relax,
            learning_rate=learning_rate,
        )
    if restored_checkpoint is not None:
        restored_state = optimiser.state.setdefault(x, {})
        for key in ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            tensor = restored_checkpoint.tensors.get(f"optimizer_{key}")
            if tensor is not None:
                target_device = x.device if key != "step" or cuda_graphs else torch.device("cpu")
                restored_state[key] = tensor.to(device=target_device).detach().clone()
        if hasattr(optimiser, "steps"):
            optimiser.steps = int(restored_checkpoint.metadata.get("optimizer_steps", 0))
    if cuda_graphs:
        x.grad = torch.zeros_like(x)
        optimiser_state = optimiser.state[x]
        optimiser_state.setdefault("step", torch.zeros((), device=x.device))
        optimiser_state.setdefault("exp_avg", torch.zeros_like(x))
        optimiser_state.setdefault("exp_avg_sq", torch.zeros_like(x))

    from qqa.runtime.population import (
        ReplicaPortfolio,
        estimate_convexification_beta,
        factor_preconditioner,
    )

    preconditioner = (
        factor_preconditioner(problem, x) if factor_preconditioning else torch.ones_like(x[0])
    )

    hp = {"div_param": float(div_param), "restart_count": 0}
    restart_count_device = torch.zeros((), dtype=torch.int64, device=x.device)
    restart_epoch_mask = torch.zeros(num_epochs, dtype=torch.bool, device=x.device)
    exchange_count_device = torch.zeros((), dtype=torch.int64, device=x.device)
    exchange_epoch_mask = torch.zeros(num_epochs, dtype=torch.bool, device=x.device)
    if restored_checkpoint is not None:
        for name, target in (
            ("restart_count", restart_count_device),
            ("exchange_count", exchange_count_device),
        ):
            saved = restored_checkpoint.tensors.get(name)
            if saved is not None:
                if saved.numel() != 1:
                    raise ValueError(f"Checkpoint {name} must be scalar.")
                target.copy_(saved.to(device=x.device, dtype=target.dtype).reshape(()))
        for name, target in (
            ("restart_epoch_mask", restart_epoch_mask),
            ("exchange_epoch_mask", exchange_epoch_mask),
        ):
            saved = restored_checkpoint.tensors.get(name)
            if saved is not None:
                saved = saved.to(device=x.device, dtype=torch.bool).reshape(-1)
                if len(saved) < start_epoch or start_epoch > len(target):
                    raise ValueError(f"Checkpoint {name} does not cover its completed epochs.")
                target[:start_epoch].copy_(saved[:start_epoch])

    with torch.no_grad():
        x_disc = relax.project(x)
        loss_disc = loss_function(x_disc)
        candidate_key_function = getattr(problem, "incumbent_keys", None)
        incumbent_keys: Callable[[torch.Tensor], torch.Tensor] | None = (
            candidate_key_function if callable(candidate_key_function) else None
        )
        if normalize_loss and robust_scaling:
            objective_center = _replica_median(loss_disc)
            objective_mad = _replica_median((loss_disc - objective_center).abs())
            objective_fallback = _replica_median(loss_disc.abs()).clamp_min(1.0)
            objective_scale = torch.where(
                objective_mad > torch.finfo(loss_disc.dtype).eps,
                objective_mad,
                objective_fallback,
            )
        else:
            objective_center = torch.zeros_like(loss_disc[0])
            objective_scale = torch.ones_like(loss_disc[0])
        best_key_gpu = None
        best_obj: Any
        if not is_batch and incumbent_keys is not None:
            candidate_keys = incumbent_keys(x_disc)
            selected_index = _lexicographic_argmin(candidate_keys)
            best_key_gpu = candidate_keys[selected_index].detach().clone()
            best_sol = x_disc[selected_index].detach().clone()
            ranking = getattr(problem, "ranking_objective", loss_function)
            best_obj = float(ranking(best_sol.unsqueeze(0))[0].item())
        elif is_batch:
            min_vals, min_idx = torch.min(loss_disc, dim=0)
            best_obj = min_vals.detach().cpu().numpy().astype(np.float64)
            columns = torch.arange(x_disc.size(1), device=x_disc.device)
            best_sol = x_disc[min_idx, columns].detach().clone()
        else:
            min_val, min_idx = torch.min(loss_disc, dim=0)
            best_obj = float(min_val.item())
            # Store the single winning replica (not the whole batch) so that
            # downstream code — ``problem.score_summary``, CLI, notebooks —
            # sees a clean ``(N, ...)`` tensor rather than ``(B, N, ...)``.
            best_sol = x_disc[int(min_idx.item())].detach().clone()
    if restored_checkpoint is not None:
        for name in ("objective_center", "objective_scale"):
            saved = restored_checkpoint.tensors.get(name)
            current = objective_center if name == "objective_center" else objective_scale
            if saved is None or saved.shape != current.shape:
                raise ValueError(f"Checkpoint has no compatible {name} tensor.")
            if name == "objective_center":
                objective_center = saved.to(device=x.device, dtype=loss_disc.dtype)
            else:
                objective_scale = saved.to(device=x.device, dtype=loss_disc.dtype)
    dimensions = max(
        1,
        relax.num_variables(problem) * int(getattr(problem, "num_instance", 1) or 1),
    )
    scale_for_convexification = float(objective_scale.reshape(-1).median().item())
    convexification_beta = (
        estimate_convexification_beta(
            problem,
            objective_scale=scale_for_convexification,
            dimensions=dimensions,
        )
        if curvature_aware_beta and curve_rate == 2
        else 0.0
    )
    replica_portfolio = (
        ReplicaPortfolio(sol_size, convexification_beta=convexification_beta)
        if heterogeneous_replicas and sol_size > 1
        else None
    )
    learning_rate_scale = (
        replica_portfolio.learning_rate_scale(x.device, x.dtype)
        if replica_portfolio is not None
        else torch.ones(sol_size, device=x.device, dtype=x.dtype)
    )
    learning_rate_scale = learning_rate_scale.reshape(sol_size, *((1,) * (x.ndim - 1)))
    optimizer_step_scale = learning_rate_scale * preconditioner
    adaptive_optimizer = optimizer in {"adamw", "lightweight-adamw"}
    optimizer_origin = torch.empty_like(x) if adaptive_optimizer else None

    @torch.no_grad()
    def capture_optimizer_origin() -> None:
        if optimizer_origin is not None:
            optimizer_origin.copy_(x)

    @torch.no_grad()
    def apply_effective_step_scale() -> None:
        """Scale the actual Adam update; scaling its input gradient cancels."""
        if optimizer_origin is None:
            return
        _apply_optimizer_step_scale_(x, optimizer_origin, optimizer_step_scale)

    # Keep incumbent comparisons and solution selection device-resident.  The
    # host receives the final value once after the loop (or when an explicit
    # user callback chooses to inspect it).
    best_obj_gpu = torch.as_tensor(best_obj, device=x.device, dtype=loss_disc.dtype)
    if restored_checkpoint is not None:
        checkpoint_best_sol = restored_checkpoint.tensors.get("best_solution")
        checkpoint_best_obj = restored_checkpoint.tensors.get("best_objective")
        if checkpoint_best_sol is None or checkpoint_best_obj is None:
            raise ValueError("Checkpoint has no incumbent tensors.")
        best_sol = checkpoint_best_sol.to(device=x.device, dtype=x_disc.dtype).detach().clone()
        best_obj_gpu = (
            checkpoint_best_obj.to(device=x.device, dtype=loss_disc.dtype).detach().clone()
        )
        best_obj = (
            best_obj_gpu.detach().cpu().numpy().astype(np.float64)
            if is_batch
            else float(best_obj_gpu.item())
        )
        if best_key_gpu is not None:
            if incumbent_keys is None:
                raise RuntimeError("Incumbent keys disappeared during checkpoint restoration.")
            checkpoint_best_key = restored_checkpoint.tensors.get("best_incumbent_key")
            best_key_gpu = (
                incumbent_keys(best_sol.unsqueeze(0))[0].detach().clone()
                if checkpoint_best_key is None
                else checkpoint_best_key.to(device=x.device, dtype=loss_disc.dtype)
            )
    restart_reference = (
        best_obj_gpu.detach().clone() if best_key_gpu is None else best_key_gpu.detach().clone()
    )
    adaptive_reference = restart_reference.detach().clone()
    if restored_checkpoint is not None:
        for name, current in (
            ("restart_reference", restart_reference),
            ("adaptive_reference", adaptive_reference),
        ):
            saved = restored_checkpoint.tensors.get(name)
            if saved is None or saved.shape != current.shape:
                raise ValueError(f"Checkpoint has no compatible {name} tensor.")
            if name == "restart_reference":
                restart_reference = saved.to(device=x.device, dtype=current.dtype)
            else:
                adaptive_reference = saved.to(device=x.device, dtype=current.dtype)
    # Adaptive schedule observation is an explicit host control point. Keep
    # it aligned with check_interval so ordinary GPU runs do not gain hidden
    # synchronisations in addition to user-requested progress checks.
    adaptive_interval = check_interval
    # Pre-seed ``state`` with the post-init evaluation so ``on_train_end`` has
    # a valid CallbackState even when ``num_epochs == 0``. The loop below will
    # overwrite it as it iterates.
    initial_bg = _schedule_value(schedule, start_epoch, num_epochs)
    state = CallbackState(
        epoch=-1,
        num_epochs=num_epochs,
        bg=initial_bg,
        x=x,
        losses=torch.zeros(1, device=x.device),
        penalties=torch.zeros(1, device=x.device),
        diversity=torch.zeros((), device=x.device),
        best_obj=best_obj_gpu,
        hyperparams=hp,
        problem=problem,
        relaxation=relax,
    )
    for cb in cb_list:
        cb.on_train_begin(state)
    if restored_checkpoint is not None:
        for cb in cb_list:
            restore_callback = getattr(cb, "restore_checkpoint_tensors", None)
            if callable(restore_callback) and cb is not archive_callback:
                restore_callback(restored_checkpoint.tensors)

    def loss_terms(
        bg_value: float | torch.Tensor,
        diversity_weight: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with amp_ctx:
            x_fwd = relax.forward(x)
            losses_value = loss_function(x_fwd)
            penalty_from_forward = getattr(relax, "penalty_from_forward", None)
            if penalty_from_forward is None:
                penalties_value = _default_penalty_from_forward(relax, x, x_fwd, curve_rate)
            else:
                penalties_value = penalty_from_forward(x, x_fwd, curve_rate)
            diversity_value = (
                relax.diversity(x) if sol_size > 1 else torch.zeros((), device=x.device)
            )
            if normalize_loss:
                normalised_losses = (losses_value - objective_center) / objective_scale
                beta = torch.as_tensor(bg_value, device=x.device, dtype=x.dtype)
                while beta.ndim < penalties_value.ndim:
                    beta = beta.unsqueeze(-1)
                objective_term = normalised_losses.mean()
                penalty_term = (penalties_value * beta).mean() / dimensions
                diversity_term = -diversity_value / dimensions
                total_value = (objective_term + penalty_term) * (
                    1 - diversity_weight
                ) + diversity_term * diversity_weight
            else:
                diversity_term = -diversity_value * sol_size
                total_value = (losses_value.sum() + (penalties_value * bg_value).sum()) * (
                    1 - diversity_weight
                ) + diversity_term * diversity_weight
        return losses_value, penalties_value, diversity_value, total_value

    captured_step = None
    if cuda_graphs:
        from qqa.gpu import CUDAGraphStep

        def graph_optimizer_step(
            bg_value: torch.Tensor,
            diversity_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            optimiser.zero_grad(set_to_none=False)
            losses_value, penalties_value, diversity_value, total_value = loss_terms(
                bg_value, diversity_weight
            )
            total_value.backward()
            gradient = x.grad
            if gradient is None:
                raise RuntimeError("Autograd returned no latent gradient.")
            if optimizer_origin is not None:
                capture_optimizer_origin()
            else:
                gradient.mul_(optimizer_step_scale)
            if gradient_clip_norm is not None:
                gradient_scale = (
                    gradient_clip_norm / (torch.linalg.vector_norm(gradient) + 1e-12)
                ).clamp(max=1.0)
                gradient.mul_(gradient_scale)
            optimiser.step()
            apply_effective_step_scale()
            relax.perturb_(x, learning_rate, temp)
            return losses_value, penalties_value, diversity_value

        optimiser_tensors = tuple(
            value for value in optimiser.state[x].values() if torch.is_tensor(value)
        )
        initial_beta = (
            replica_portfolio.beta(
                initial_bg,
                0.0 if num_epochs <= 1 else start_epoch / (num_epochs - 1),
                device=x.device,
                dtype=x.dtype,
            )
            if replica_portfolio is not None
            else x.new_tensor(initial_bg)
        )
        graph_gradient = x.grad
        if graph_gradient is None:
            raise RuntimeError("CUDA graph capture requires an allocated latent gradient.")
        captured_step = CUDAGraphStep(
            graph_optimizer_step,
            (
                initial_beta,
                x.new_tensor(float(hp["div_param"])),
            ),
            state_tensors=(
                x,
                graph_gradient,
                *(() if optimizer_origin is None else (optimizer_origin,)),
                *optimiser_tensors,
            ),
        )

    def save_runtime_checkpoint(epoch: int, *, completed: bool) -> None:
        if checkpoint_path is None:
            return
        from qqa.runtime.checkpoint import Checkpoint, save_checkpoint

        optimiser_state = optimiser.state.get(x, {})
        tensors = {
            "latent": x.detach(),
            "best_solution": best_sol.detach(),
            "best_objective": best_obj_gpu.detach(),
            "cpu_rng_state": torch.random.get_rng_state(),
            "restart_reference": restart_reference.detach(),
            "adaptive_reference": adaptive_reference.detach(),
            "objective_center": objective_center.detach(),
            "objective_scale": objective_scale.detach(),
            "restart_count": restart_count_device.detach(),
            "restart_epoch_mask": restart_epoch_mask.detach(),
            "exchange_count": exchange_count_device.detach(),
            "exchange_epoch_mask": exchange_epoch_mask.detach(),
        }
        if best_key_gpu is not None:
            tensors["best_incumbent_key"] = best_key_gpu.detach()
        if x.device.type == "cuda":
            tensors["device_rng_state"] = torch.cuda.get_rng_state(x.device)
        if archive_callback is not None:
            tensors.update(archive_callback.checkpoint_tensors())
        for callback in cb_list:
            checkpoint_callback = getattr(callback, "checkpoint_tensors", None)
            if callable(checkpoint_callback) and callback is not archive_callback:
                for name, value in checkpoint_callback().items():
                    if name in tensors:
                        raise ValueError(f"Duplicate callback checkpoint tensor: {name}.")
                    tensors[name] = value
        for key in ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = optimiser_state.get(key)
            if torch.is_tensor(value):
                tensors[f"optimizer_{key}"] = value.detach()
        save_checkpoint(
            Checkpoint(
                model_fingerprint=str(checkpoint_fingerprint),
                config=checkpoint_config,
                epoch=epoch,
                tensors=tensors,
                metadata={
                    "completed": completed,
                    "optimizer_steps": int(getattr(optimiser, "steps", epoch)),
                    "schedule_state": _schedule_checkpoint_descriptor(schedule)["state"],
                },
            ),
            checkpoint_path,
        )

    completed_epochs = start_epoch
    deadline_reached = False
    for epoch in range(start_epoch, num_epochs):
        # CUDA launches asynchronously.  Synchronising only for an explicit
        # wall-clock deadline prevents a queued epoch stream from appearing
        # to fit the budget and then overrunning while results transfer back
        # to the host.  Unbounded throughput-oriented runs remain unchanged.
        if time_limit is not None and _is_cuda:
            torch.cuda.synchronize(device)
        if time_limit is not None and perf_counter() - runtime_start >= time_limit:
            deadline_reached = True
            break
        update_relaxation = getattr(relax, "set_progress", None)
        progress = 1.0 if num_epochs <= 1 else epoch / (num_epochs - 1)
        if callable(update_relaxation):
            update_relaxation(progress)
        bg = initial_bg if epoch == start_epoch else _schedule_value(schedule, epoch, num_epochs)
        beta_value: float | torch.Tensor = (
            replica_portfolio.beta(bg, progress, device=x.device, dtype=x.dtype)
            if replica_portfolio is not None
            else bg
        )
        if captured_step is not None:
            losses, penalties, diversity = captured_step.replay(
                torch.as_tensor(beta_value, device=x.device, dtype=x.dtype),
                x.new_tensor(float(hp["div_param"])),
            )
        else:
            optimiser.zero_grad(set_to_none=True)
            losses, penalties, diversity, total = loss_terms(beta_value, hp["div_param"])
            # Backward and AdamW remain full precision outside autocast.
            total.backward()
            gradient = x.grad
            if gradient is None:
                raise RuntimeError("Autograd returned no latent gradient.")
            if optimizer_origin is not None:
                capture_optimizer_origin()
            else:
                gradient.mul_(optimizer_step_scale)
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([x], gradient_clip_norm)
            optimiser.step()
            apply_effective_step_scale()
            relax.perturb_(x, learning_rate, temp)

        with torch.no_grad():
            x_disc = relax.project(x)
            loss_disc = loss_function(x_disc)
            if is_batch:
                min_vals, min_idx = torch.min(loss_disc, dim=0)
                improved_mask = min_vals < best_obj_gpu
                selected = x_disc[min_idx, torch.arange(x_disc.size(1), device=x.device)]
                best_sol = torch.where(improved_mask.unsqueeze(-1), selected, best_sol)
                best_obj_gpu = torch.minimum(best_obj_gpu, min_vals)
            else:
                if best_key_gpu is not None:
                    if incumbent_keys is None:
                        raise RuntimeError("Incumbent keys disappeared during annealing.")
                    candidate_keys = incumbent_keys(x_disc)
                    min_idx = _lexicographic_argmin(candidate_keys)
                    candidate_key = candidate_keys[min_idx]
                    improved = _lexicographic_less(candidate_key, best_key_gpu)
                    selected = x_disc[min_idx]
                    best_sol = torch.where(improved, selected, best_sol)
                    best_key_gpu = torch.where(improved, candidate_key, best_key_gpu)
                    ranking = getattr(problem, "ranking_objective", loss_function)
                    candidate_objective = ranking(selected.unsqueeze(0))[0]
                    best_obj_gpu = torch.where(improved, candidate_objective, best_obj_gpu)
                else:
                    min_val, min_idx = torch.min(loss_disc, dim=0)
                    improved = min_val < best_obj_gpu
                    selected = x_disc[min_idx]
                    best_sol = torch.where(improved, selected, best_sol)
                    best_obj_gpu = torch.minimum(best_obj_gpu, min_val)
            if (
                replica_portfolio is not None
                and replica_exchange_interval is not None
                and epoch < num_epochs - 1
                and (epoch + 1) % replica_exchange_interval == 0
            ):
                exchanged = replica_portfolio.exchange_(
                    x,
                    loss_disc,
                    optimiser,
                    epoch=epoch,
                )
                exchange_count_device.add_(exchanged)
                exchange_epoch_mask[epoch] = exchanged > 0
        state = CallbackState(
            epoch=epoch,
            num_epochs=num_epochs,
            bg=bg,
            x=x,
            losses=losses.detach(),
            penalties=penalties.detach(),
            diversity=diversity.detach() if torch.is_tensor(diversity) else diversity,
            best_obj=best_obj_gpu.detach(),
            hyperparams=hp,
            problem=problem,
            relaxation=relax,
        )
        for cb in cb_list:
            cb.on_epoch_end(state)
        completed_epochs = epoch + 1

        observe_schedule = getattr(schedule, "observe", None)
        if callable(observe_schedule) and (
            (epoch + 1) % adaptive_interval == 0 or epoch == num_epochs - 1
        ):
            window_improved = (
                torch.any(best_obj_gpu < adaptive_reference - 1e-12)
                if best_key_gpu is None
                else _lexicographic_less(best_key_gpu, adaptive_reference)
            )
            dimensions = max(
                1, relax.num_variables(problem) * int(getattr(problem, "num_instance", 1) or 1)
            )
            diversity_ratio = diversity / dimensions
            observe_schedule(
                improved=bool(window_improved.item()),
                diversity_ratio=float(diversity_ratio.item()),
            )
            adaptive_reference = (
                best_obj_gpu.detach().clone()
                if best_key_gpu is None
                else best_key_gpu.detach().clone()
            )

        if verbose and (epoch % check_interval == 0 or epoch == num_epochs - 1):
            display_best = (
                best_obj_gpu.detach().cpu().numpy().astype(np.float64)
                if is_batch
                else float(best_obj_gpu.item())
            )
            _print_progress(epoch, display_best, losses, penalties, diversity, bg, hp["div_param"])

        if (
            restart_patience is not None
            and sol_size > 1
            and epoch < num_epochs - 1
            and (epoch + 1) % restart_patience == 0
        ):
            improved_in_window = (
                torch.any(best_obj_gpu < restart_reference - 1e-12)
                if best_key_gpu is None
                else _lexicographic_less(best_key_gpu, restart_reference)
            )
            restart_enabled = ~improved_in_window
            restarted = _restart_replicas(
                x=x,
                best_sol=best_sol,
                discrete_losses=loss_disc,
                relaxation=relax,
                optimizer=optimiser,
                fraction=restart_fraction,
                jitter=restart_jitter,
                learning_rate=learning_rate,
                archive_centres=(
                    None
                    if archive_callback is None
                    else archive_callback.device_restart_centres(
                        max(1, math.ceil(sol_size * restart_fraction / 2))
                    )
                ),
                enabled=restart_enabled,
            )
            if restarted:
                restart_count_device.add_(restart_enabled.to(torch.int64) * restarted)
                restart_epoch_mask[epoch] = restart_enabled
            restart_reference = (
                best_obj_gpu.detach().clone()
                if best_key_gpu is None
                else best_key_gpu.detach().clone()
            )
        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and completed_epochs % checkpoint_interval == 0
        ):
            save_runtime_checkpoint(completed_epochs, completed=False)

    if time_limit is not None and _is_cuda:
        torch.cuda.synchronize(device)
    runtime = perf_counter() - runtime_start
    best_obj = (
        best_obj_gpu.detach().cpu().numpy().astype(np.float64)
        if is_batch
        else float(best_obj_gpu.item())
    )
    restart_count = int(restart_count_device.item())
    restart_epochs = restart_epoch_mask.nonzero(as_tuple=False).flatten().cpu().tolist()
    exchange_count = int(exchange_count_device.item())
    exchange_epochs = exchange_epoch_mask.nonzero(as_tuple=False).flatten().cpu().tolist()
    hp["restart_count"] = restart_count
    if verbose:
        print("\n" + "=" * 30 + " [FINAL] " + "=" * 30)
        print(f"  BEST LOSS : {best_obj}")
        print(f"  RUN TIME  : {runtime:.2f} s")
        print("=" * 69)

    for cb in cb_list:
        cb.on_train_end(state)

    history: dict[str, Any] = recorder.history if recorder is not None else {}
    if record_history:
        history["restart_epochs"] = restart_epochs
        history["restart_count"] = restart_count
        history["exchange_epochs"] = exchange_epochs
        history["exchange_count"] = exchange_count

    # Human-readable score.
    # * Single-instance: ``best_obj`` is a Python float and ``score`` is the
    #   per-solution metric.
    # * Batched-instance: if the problem provides a per-instance
    #   ``score_summary`` (the ``*Instance`` classes do), call it so callers
    #   get per-instance feasibility / value arrays. If not (legacy custom
    #   problems), leave ``score`` empty rather than silently mis-formatting.
    score: dict = {}
    if not is_batch:
        score = safe_score_summary(problem, best_sol, fallback_obj=float(best_obj))
    elif type(problem).score_summary is not COProblem.score_summary:
        try:
            score = problem.score_summary(best_sol)
        except Exception as exc:  # noqa: BLE001 - surface but never abort
            score = {"label": "loss", "feasible": False, "extra": {"error": str(exc)}}

    # Default-on domain-aware polish. Batched-instance problems are skipped
    # because their best_sol is one row per independent model rather than one
    # search state. A strict improvement hot-swaps best_sol / best_obj / score.
    prev_obj = best_obj
    deadline_reached |= time_limit is not None and perf_counter() - runtime_start >= time_limit
    best_sol, best_obj, polished_sol = apply_polish_if_improves(
        problem,
        best_sol,
        best_obj,
        polish=polish and not is_batch and not deadline_reached,
    )
    # The incumbent is initialised from a non-empty replica population before
    # polishing, so the helper's optional return cannot be ``None`` here.
    if best_sol is None:
        raise RuntimeError("Annealing finished without its initialized incumbent.")
    if polished_sol is not None and best_obj < prev_obj:
        score = safe_score_summary(problem, best_sol, fallback_obj=float(best_obj))
        if verbose:
            print(f"  POLISH    : 1-flip improved best_obj -> {best_obj}")
    if checkpoint_path is not None:
        best_obj_gpu = torch.as_tensor(best_obj, device=x.device, dtype=loss_disc.dtype)
        save_runtime_checkpoint(completed_epochs, completed=completed_epochs >= num_epochs)

    return AnnealResult(
        best_sol=best_sol,
        best_obj=best_obj,
        runtime=runtime,
        history=history,
        callbacks=cb_list,
        score=score,
        polished_sol=polished_sol,
        final_population=x_disc.detach().clone() if return_population else None,
        diagnostics={
            "weight_decay": float(weight_decay),
            "optimizer": optimizer,
            "gradient_clip_norm": gradient_clip_norm,
            "restart_patience": restart_patience,
            "restart_fraction": float(restart_fraction),
            "restart_jitter": float(restart_jitter),
            "restart_count": restart_count,
            "restart_events": len(restart_epochs),
            "completed_epochs": completed_epochs,
            "time_limit": time_limit,
            "deadline_reached": deadline_reached,
            "compile_core": compile_core,
            "cuda_graphs": cuda_graphs,
            "normalize_loss": normalize_loss,
            "robust_scaling": robust_scaling,
            "objective_center": objective_center.detach().cpu().tolist(),
            "objective_scale": objective_scale.detach().cpu().tolist(),
            "heterogeneous_replicas": heterogeneous_replicas,
            "replica_roles": (
                [] if replica_portfolio is None else replica_portfolio.roles("cpu").tolist()
            ),
            "replica_exchange_interval": replica_exchange_interval,
            "replica_exchange_count": exchange_count,
            "replica_exchange_events": len(exchange_epochs),
            "replica_exchange_mode": "heuristic_role_exchange",
            "factor_preconditioning": factor_preconditioning,
            "curvature_aware_beta": curvature_aware_beta,
            "convexification_beta": convexification_beta,
            "historical_archive": (
                None if archive_callback is None else archive_callback.archive.diagnostics()
            ),
            "resumed": restored_checkpoint is not None,
            "resume_semantics": resume_semantics,
            "checkpoint_written": checkpoint_path is not None,
        },
        archive=None if archive_callback is None else archive_callback.archive,
    )


def _print_progress(epoch, best_obj, losses, penalties, diversity, bg, div_param):
    mean_loss = float(losses.detach().mean().item())
    mean_pen = float(penalties.detach().mean().item())
    div_val = float(diversity.item()) if torch.is_tensor(diversity) else float(diversity)
    print("\n" + "=" * 30 + " [LOG] " + "=" * 32)
    print(f"[ EPOCH {epoch} ]")
    print(f"  Best Loss So Far : {best_obj}")
    print(f"  Mean(Loss)       : {mean_loss:.4f}")
    print(f"  Mean(Penalty)    : {mean_pen:.4f}")
    print(f"  BG               : {bg:.4f}")
    print(f"  DIV Value        : {div_val:.4f}")
    print(f"  div_param        : {div_param:.4f}")
    print("=" * 69)
