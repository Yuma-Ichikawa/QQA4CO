"""Unified Quasi-Quantum Annealing loop.

This module replaces the four legacy ``batch_annealing_*`` functions from the
original repository with a single :func:`anneal` routine that delegates
problem-specific behaviour to :mod:`qqa.relaxation` and :mod:`qqa.callbacks`.

Single-instance binary problems, batched-instance problems, and categorical
problems all share this same loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from time import time
from typing import Any, Literal

import numpy as np
import torch

from qqa.callbacks import Callback, CallbackState, HistoryRecorder
from qqa.relaxation import _default_penalty_from_forward
from qqa.schedule import LinearBGSchedule


@dataclass
class AnnealResult:
    """Result returned by :func:`anneal`.

    Attributes
    ----------
    best_sol:
        Tensor of the best discrete solution(s) found during annealing. Shape
        depends on the problem: ``(sol_size, N)`` for single-instance, or
        ``(num_instance, max_node)`` for batched-instance problems.
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
    :py:meth:`COProblem.score_summary` (``label``, ``value``, ``unit``,
    ``feasible``, ``extra``). Empty for batched-instance problems."""


def _is_instance_problem(problem) -> bool:
    return hasattr(problem, "num_instance")


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
    """
    if sol_size < 1:
        raise ValueError(f"sol_size must be >= 1, got {sol_size}.")
    if num_epochs < 0:
        raise ValueError(f"num_epochs must be >= 0, got {num_epochs}.")
    if mixed_precision not in ("fp32", "bf16"):
        raise ValueError(f"mixed_precision must be 'fp32' or 'bf16', got {mixed_precision!r}.")

    # Surface a helpful message when CUDA is requested but unavailable,
    # before torch raises its own (cryptic) error deep inside .to().
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled torch build, or pass device='cpu'."
        )

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
    x = relax.init(sol_size, problem, device)
    optimizer = torch.optim.AdamW([x], lr=learning_rate)

    hp = {"div_param": float(div_param)}
    is_batch = _is_instance_problem(problem)

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
        best_obj_gpu = torch.tensor(best_obj, device=x.device, dtype=torch.float32)
    else:
        # Batched problems: keep ``best_obj`` on GPU as well so per-epoch
        # comparisons skip the cpu().numpy() roundtrip.
        best_obj_gpu = torch.as_tensor(best_obj, device=x.device, dtype=torch.float32)

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
        optimizer.step()

        relax.perturb_(x, learning_rate, temp)

        with torch.no_grad():
            x_disc = relax.project(x)
            loss_disc = problem.loss_fn(x_disc)
            if is_batch:
                min_vals, min_idx = torch.min(loss_disc, dim=0)
                # GPU-side improvement mask; one ``.any().item()`` sync per
                # epoch (vs the previous full ``cpu().numpy()`` transfer).
                improved_mask = min_vals < best_obj_gpu
                if improved_mask.any().item():
                    sel = x_disc[min_idx, torch.arange(x_disc.size(1), device=x.device)]
                    best_sol = torch.where(improved_mask.unsqueeze(-1), sel, best_sol)
                    best_obj_gpu = torch.minimum(best_obj_gpu, min_vals)
                    best_obj = best_obj_gpu.detach().cpu().numpy().astype(np.float64)
            else:
                min_val, min_idx = torch.min(loss_disc, dim=0)
                # One sync per epoch in the common (non-improving) case.
                if (min_val < best_obj_gpu).item():
                    best_obj_gpu = min_val.detach()
                    best_obj = float(best_obj_gpu.item())
                    best_sol = x_disc[int(min_idx.item())].detach().clone()

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

    runtime = time() - runtime_start
    if verbose:
        print("\n" + "=" * 30 + " [FINAL] " + "=" * 30)
        print(f"  BEST LOSS : {best_obj}")
        print(f"  RUN TIME  : {runtime:.2f} s")
        print("=" * 69)

    for cb in cb_list:
        cb.on_train_end(state)

    history = recorder.history if recorder is not None else {}

    # Human-readable score. Only meaningful for single-instance problems,
    # where ``best_sol`` is a single solution tensor.
    score: dict = {}
    if not is_batch:
        try:
            score = problem.score_summary(best_sol)
        except Exception as exc:  # noqa: BLE001 - surface but never abort
            # ``feasible=False`` (not True) so that downstream UIs / CLI don't
            # mis-advertise an unchecked solution as valid just because the
            # scorer itself crashed.
            score = {
                "label": "loss",
                "value": float(best_obj),
                "unit": "",
                "feasible": False,
                "extra": {"error": str(exc)},
            }

    return AnnealResult(
        best_sol=best_sol,
        best_obj=best_obj,
        runtime=runtime,
        history=history,
        callbacks=cb_list,
        score=score,
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
