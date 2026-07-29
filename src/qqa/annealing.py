"""Unified Quasi-Quantum Annealing loop.

This module replaces the four legacy ``batch_annealing_*`` functions from the
original repository with a single :func:`anneal` routine that delegates
problem-specific behaviour to :mod:`qqa.relaxation` and :mod:`qqa.callbacks`.

Single-instance binary problems, batched-instance problems, and categorical
problems all share this same loop.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from time import time
from typing import Any, Literal

import numpy as np
import torch

from qqa.callbacks import Callback, CallbackState, HistoryRecorder
from qqa.polish import apply_polish_if_improves
from qqa.problems.base import COProblem
from qqa.relaxation import _default_penalty_from_forward
from qqa.schedule import LinearBGSchedule
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


def _reset_optimizer_rows(
    optimizer: torch.optim.Optimizer,
    parameter: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Clear Adam moments for replicas whose latent state was restarted."""
    state = optimizer.state.get(parameter, {})
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if torch.is_tensor(value) and value.shape == parameter.shape:
            value[rows].zero_()


def _restart_replicas(
    *,
    x: torch.Tensor,
    best_sol: torch.Tensor,
    discrete_losses: torch.Tensor,
    relaxation,
    optimizer: torch.optim.Optimizer,
    fraction: float,
    jitter: float,
    learning_rate: float,
) -> int:
    """Restart weak replicas with a global/local basin-hopping mixture."""
    count = min(x.shape[0] - 1, max(1, math.ceil(x.shape[0] * fraction)))
    if count < 1:
        return 0
    worst = torch.topk(_replica_merit(discrete_losses), k=count, largest=True).indices
    encode = getattr(relaxation, "encode", None)
    elite = best_sol.to(device=x.device, dtype=x.dtype)
    if encode is not None:
        elite = encode(elite)
    if elite.shape != x.shape[1:]:
        return 0

    # Half of the weak replicas intensify around the incumbent; the rest
    # restart globally. This is a bounded basin-hopping step, not mutation of
    # the preserved incumbent itself.
    elite_count = (count + 1) // 2
    with torch.no_grad():
        replacements = torch.rand_like(x[worst])
        if elite_count:
            local = elite.unsqueeze(0).expand(elite_count, *elite.shape).clone()
            local.add_(jitter * torch.randn_like(local))
            replacements[:elite_count] = local
        x[worst] = replacements
        relaxation.perturb_(x, learning_rate, 0.0)
        _reset_optimizer_rows(optimizer, x, worst)
    return count


def anneal(
    problem,
    *,
    sol_size: int = 100,
    learning_rate: float = 1.0,
    temp: float = 0.0,
    schedule: LinearBGSchedule | None = None,
    min_bg: float | None = None,
    max_bg: float | None = None,
    curve_rate: int = 2,
    div_param: float = 0.0,
    num_epochs: int = 10_000,
    check_interval: int = 1000,
    device: str | torch.device = "cpu",
    callbacks: Sequence[Callback] = (),
    record_history: bool = True,
    verbose: bool = True,
    mixed_precision: Literal["fp32", "bf16"] = "fp32",
    initial_state: torch.Tensor | None = None,
    polish: bool = True,
    return_population: bool = False,
    weight_decay: float = 0.0,
    gradient_clip_norm: float | None = None,
    restart_patience: int | None = None,
    restart_fraction: float = 0.15,
    restart_jitter: float = 0.10,
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
        Exponent of the QQA penalty (must be even for the convex regime).
    div_param:
        Weight of the diversity term. Set to 0 to disable.
    num_epochs:
        Number of gradient steps.
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
        evaluation run inside ``torch.amp.autocast`` with bfloat16 — typically
        a 1.3x-2x speedup on Ampere/Hopper/Blackwell with negligible accuracy
        loss for QUBO/spin objectives. The relaxed variable, gradients, and
        AdamW state stay in float32 for numerical stability. Defaults to
        ``"fp32"`` so behaviour is bit-for-bit identical to the legacy loop.
        On CPU this option is silently downgraded to ``"fp32"``.
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
    """
    if not isinstance(sol_size, int) or isinstance(sol_size, bool) or sol_size < 1:
        raise ValueError(f"sol_size must be >= 1, got {sol_size}.")
    if not isinstance(num_epochs, int) or isinstance(num_epochs, bool) or num_epochs < 0:
        raise ValueError(f"num_epochs must be >= 0, got {num_epochs}.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")
    if not math.isfinite(temp) or temp < 0:
        raise ValueError(f"temp must be >= 0, got {temp}.")
    if (
        not isinstance(check_interval, int)
        or isinstance(check_interval, bool)
        or check_interval < 1
    ):
        raise ValueError(f"check_interval must be >= 1, got {check_interval}.")
    if not math.isfinite(div_param) or not 0.0 <= div_param <= 1.0:
        raise ValueError(f"div_param must be in [0, 1], got {div_param}.")
    if mixed_precision not in ("fp32", "bf16"):
        raise ValueError(f"mixed_precision must be 'fp32' or 'bf16', got {mixed_precision!r}.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(f"weight_decay must be finite and >= 0, got {weight_decay}.")
    if gradient_clip_norm is not None and (
        not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0
    ):
        raise ValueError(f"gradient_clip_norm must be finite and > 0, got {gradient_clip_norm}.")
    if restart_patience is not None and (
        not isinstance(restart_patience, int)
        or isinstance(restart_patience, bool)
        or restart_patience < 1
    ):
        raise ValueError("restart_patience must be a positive integer or None.")
    if not math.isfinite(restart_fraction) or not 0 < restart_fraction < 1:
        raise ValueError("restart_fraction must be finite and in (0, 1).")
    if not math.isfinite(restart_jitter) or not 0 <= restart_jitter <= 1:
        raise ValueError("restart_jitter must be finite and in [0, 1].")
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
    use_amp = mixed_precision == "bf16" and _is_cuda
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    )

    if schedule is None:
        schedule = LinearBGSchedule(
            -2.0 if min_bg is None else min_bg,
            0.1 if max_bg is None else max_bg,
        )

    relax = problem.relaxation

    cb_list: list[Callback] = []
    recorder: HistoryRecorder | None = None
    if record_history:
        recorder = HistoryRecorder()
        cb_list.append(recorder)
    cb_list.extend(callbacks)

    runtime_start = time()
    is_batch = _is_instance_problem(problem)
    if initial_state is not None and is_batch:
        # Warm-starting batched-instance problems would require a
        # per-instance seed tensor and is rarely useful — skip silently
        # rather than impose a confusing shape contract.
        initial_state = None
    if initial_state is None:
        x = relax.init(sol_size, problem, device)
    else:
        seed_dtype = getattr(problem, "dtype", torch.float32)
        seed = initial_state.detach().to(device=device, dtype=seed_dtype)
        encode = getattr(relax, "encode", None)
        if encode is not None:
            seed = encode(seed)
        if seed.dim() == 1:
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
        x = (seed + 0.05 * torch.randn_like(seed)).clamp_(0.0, 1.0)
        x.requires_grad_(True)
    optimizer = torch.optim.AdamW([x], lr=learning_rate, weight_decay=weight_decay)

    hp = {"div_param": float(div_param), "restart_count": 0}
    restart_epochs: list[int] = []
    restart_count = 0
    stagnant_epochs = 0

    with torch.no_grad():
        x_disc = relax.project(x)
        loss_disc = problem.loss_fn(x_disc)
        if is_batch:
            min_vals, min_idx = torch.min(loss_disc, dim=0)
            best_obj = min_vals.detach().cpu().numpy().astype(np.float64)
            best_sol = x_disc[min_idx, torch.arange(x_disc.size(1))].detach().clone()
        else:
            min_val, min_idx = torch.min(loss_disc, dim=0)
            best_obj = float(min_val.item())
            # Store the single winning replica (not the whole batch) so that
            # downstream code — ``problem.score_summary``, CLI, notebooks —
            # sees a clean ``(N, ...)`` tensor rather than ``(B, N, ...)``.
            best_sol = x_disc[int(min_idx.item())].detach().clone()

    # Pre-seed ``state`` with the post-init evaluation so ``on_train_end`` has
    # a valid CallbackState even when ``num_epochs == 0``. The loop below will
    # overwrite it as it iterates.
    state = CallbackState(
        epoch=-1,
        num_epochs=num_epochs,
        bg=float(schedule(0, num_epochs)),
        x=x,
        losses=torch.zeros(1, device=x.device),
        penalties=torch.zeros(1, device=x.device),
        diversity=torch.zeros((), device=x.device),
        best_obj=best_obj,
        hyperparams=hp,
        problem=problem,
        relaxation=relax,
    )
    for cb in cb_list:
        cb.on_train_begin(state)

    # Pre-allocate a GPU scalar to track the best objective for single-instance
    # problems. We only sync it to the host on the (rare) ``check_interval``
    # tick or when an actual improvement happens, instead of every epoch.
    if not is_batch:
        best_obj_gpu = torch.tensor(best_obj, device=x.device, dtype=loss_disc.dtype)
    else:
        # Batched problems: keep ``best_obj`` on GPU as well so per-epoch
        # comparisons skip the cpu().numpy() roundtrip.
        best_obj_gpu = torch.as_tensor(best_obj, device=x.device, dtype=loss_disc.dtype)

    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        bg = float(schedule(epoch, num_epochs))

        with amp_ctx:
            x_fwd = relax.forward(x)
            losses = problem.loss_fn(x_fwd)  # (B,) or (B, I)
            # Reuse the cached forward output for relaxations that would
            # otherwise re-run their forward inside ``penalty`` (notably
            # CategoricalRelaxation, which does a simplex normalisation per call).
            pfwd = getattr(relax, "penalty_from_forward", None)
            if pfwd is None:
                penalties = _default_penalty_from_forward(relax, x, x_fwd, curve_rate)
            else:
                penalties = pfwd(x, x_fwd, curve_rate)  # matching shape
            diversity = relax.diversity(x) if sol_size > 1 else torch.tensor(0.0, device=x.device)
            div_term = -diversity * sol_size

            # Unified weighted objective: uses sums so that (B, I) problems
            # contribute each instance equally.
            dp = hp["div_param"]
            total = (losses.sum() + (penalties * bg).sum()) * (1 - dp) + div_term * dp
        # backward()/step() must run outside autocast so AdamW updates the
        # float32 master weights with full precision.
        total.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([x], gradient_clip_norm)
        optimizer.step()

        relax.perturb_(x, learning_rate, temp)

        with torch.no_grad():
            x_disc = relax.project(x)
            loss_disc = problem.loss_fn(x_disc)
            improved_this_epoch = False
            if is_batch:
                min_vals, min_idx = torch.min(loss_disc, dim=0)
                # GPU-side improvement mask; one ``.any().item()`` sync per
                # epoch (vs the previous full ``cpu().numpy()`` transfer).
                improved_mask = min_vals < best_obj_gpu
                if improved_mask.any().item():
                    improved_this_epoch = True
                    sel = x_disc[min_idx, torch.arange(x_disc.size(1), device=x.device)]
                    best_sol = torch.where(improved_mask.unsqueeze(-1), sel, best_sol)
                    best_obj_gpu = torch.minimum(best_obj_gpu, min_vals)
                    best_obj = best_obj_gpu.detach().cpu().numpy().astype(np.float64)
            else:
                min_val, min_idx = torch.min(loss_disc, dim=0)
                # One sync per epoch in the common (non-improving) case.
                if (min_val < best_obj_gpu).item():
                    improved_this_epoch = True
                    best_obj_gpu = min_val.detach()
                    best_obj = float(best_obj_gpu.item())
                    best_sol = x_disc[int(min_idx.item())].detach().clone()
        stagnant_epochs = 0 if improved_this_epoch else stagnant_epochs + 1

        state = CallbackState(
            epoch=epoch,
            num_epochs=num_epochs,
            bg=bg,
            x=x,
            losses=losses.detach(),
            penalties=penalties.detach(),
            diversity=diversity.detach() if torch.is_tensor(diversity) else diversity,
            best_obj=best_obj,
            hyperparams=hp,
            problem=problem,
            relaxation=relax,
        )
        for cb in cb_list:
            cb.on_epoch_end(state)

        if verbose and (epoch % check_interval == 0 or epoch == num_epochs - 1):
            _print_progress(epoch, best_obj, losses, penalties, diversity, bg, hp["div_param"])

        if (
            restart_patience is not None
            and sol_size > 1
            and epoch < num_epochs - 1
            and stagnant_epochs >= restart_patience
        ):
            restarted = _restart_replicas(
                x=x,
                best_sol=best_sol,
                discrete_losses=loss_disc,
                relaxation=relax,
                optimizer=optimizer,
                fraction=restart_fraction,
                jitter=restart_jitter,
                learning_rate=learning_rate,
            )
            if restarted:
                restart_count += restarted
                restart_epochs.append(epoch)
                hp["restart_count"] = restart_count
            stagnant_epochs = 0

    runtime = time() - runtime_start
    if verbose:
        print("\n" + "=" * 30 + " [FINAL] " + "=" * 30)
        print(f"  BEST LOSS : {best_obj}")
        print(f"  RUN TIME  : {runtime:.2f} s")
        print("=" * 69)

    for cb in cb_list:
        cb.on_train_end(state)

    history = recorder.history if recorder is not None else {}
    if record_history:
        history["restart_epochs"] = restart_epochs
        history["restart_count"] = restart_count

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
    best_sol, best_obj, polished_sol = apply_polish_if_improves(
        problem, best_sol, best_obj, polish=polish and not is_batch
    )
    if polished_sol is not None and best_obj < prev_obj:
        score = safe_score_summary(problem, best_sol, fallback_obj=float(best_obj))
        if verbose:
            print(f"  POLISH    : 1-flip improved best_obj -> {best_obj}")

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
            "gradient_clip_norm": gradient_clip_norm,
            "restart_patience": restart_patience,
            "restart_fraction": float(restart_fraction),
            "restart_jitter": float(restart_jitter),
            "restart_count": restart_count,
            "restart_events": len(restart_epochs),
        },
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
