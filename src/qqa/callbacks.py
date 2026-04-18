"""Callbacks for the QQA annealing loop.

Callbacks receive a ``CallbackState`` snapshot at the end of every epoch and
can record metrics, adjust hyper-parameters, or track auxiliary objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CallbackState:
    """Mutable context passed to callbacks at each epoch.

    The annealing loop writes fields here. Callbacks may read any field and
    may write to ``extras`` or mutate ``hyperparams`` (e.g. ``div_param``).
    """

    epoch: int
    num_epochs: int
    bg: float
    x: torch.Tensor
    losses: torch.Tensor
    penalties: torch.Tensor
    diversity: torch.Tensor
    best_obj: Any
    hyperparams: dict
    problem: Any
    relaxation: Any
    extras: dict = field(default_factory=dict)


class Callback:
    """Base class. Override ``on_epoch_end`` (and optionally other hooks)."""

    def on_train_begin(self, state: CallbackState) -> None:  # pragma: no cover
        pass

    def on_epoch_end(self, state: CallbackState) -> None:  # pragma: no cover
        pass

    def on_train_end(self, state: CallbackState) -> None:  # pragma: no cover
        pass


class HistoryRecorder(Callback):
    """Record loss / penalty / diversity statistics per epoch."""

    def __init__(self) -> None:
        self.history: dict[str, list] = {
            "loss_mean": [],
            "loss_std": [],
            "loss_min": [],
            "penalty_mean": [],
            "penalty_std": [],
            "diversity": [],
            "bg": [],
            "best_obj": [],
        }

    def on_epoch_end(self, state: CallbackState) -> None:
        losses = state.losses.detach()
        penalties = state.penalties.detach()
        self.history["loss_mean"].append(float(losses.mean().item()))
        self.history["loss_std"].append(float(losses.std().item()) if losses.numel() > 1 else 0.0)
        self.history["loss_min"].append(float(losses.min().item()))
        self.history["penalty_mean"].append(float(penalties.mean().item()))
        self.history["penalty_std"].append(
            float(penalties.std().item()) if penalties.numel() > 1 else 0.0
        )
        div = state.diversity
        self.history["diversity"].append(float(div.item()) if torch.is_tensor(div) else float(div))
        self.history["bg"].append(state.bg)
        bo = state.best_obj
        if hasattr(bo, "tolist"):
            self.history["best_obj"].append(bo.tolist())
        else:
            self.history["best_obj"].append(float(bo))


class AutoDivTuner(Callback):
    """Adaptively tune ``div_param`` to target a desired diversity ratio.

    At each epoch: ``ratio = diversity / (sol_size * N)``. The controller
    nudges ``div_param`` by ``lr * (ratio - target)`` and clips to ``[0, 1]``.
    """

    def __init__(self, target: float = 0.3, lr: float = 1e-3) -> None:
        self.target = target
        self.lr = lr

    def on_epoch_end(self, state: CallbackState) -> None:
        sol_size = state.x.shape[0]
        if sol_size <= 1:
            return
        num_vars = state.relaxation.num_variables(state.problem)
        div_val = (
            float(state.diversity.item())
            if torch.is_tensor(state.diversity)
            else float(state.diversity)
        )
        ratio = div_val / (sol_size * num_vars)
        diff = ratio - self.target
        dp = state.hyperparams.get("div_param", 0.0)
        dp = max(0.0, min(1.0, dp + self.lr * diff))
        state.hyperparams["div_param"] = dp


class PopulationTracker(Callback):
    """Snapshot the parallel population for post-hoc parallel-search visualisation.

    Records, every ``stride`` epochs:

    * ``loss`` — the ``(sol_size,)`` per-replica loss.
    * ``x``    — optionally, the continuous variables (heavier but lets you
      reconstruct PCA trajectories or per-variable heatmaps).

    Attributes:
        epochs: list of recorded epochs.
        loss:   list of ``(sol_size,)`` numpy arrays.
        x:      list of ``(sol_size, ...)`` numpy arrays when
            ``record_x=True``; otherwise empty.
    """

    def __init__(self, stride: int = 10, record_x: bool = True, max_replicas: int | None = None):
        self.stride = max(1, int(stride))
        self.record_x = bool(record_x)
        self.max_replicas = max_replicas
        self.epochs: list[int] = []
        self.loss: list[Any] = []
        self.x: list[Any] = []

    def on_epoch_end(self, state: CallbackState) -> None:
        if state.epoch % self.stride != 0 and state.epoch != state.num_epochs - 1:
            return
        self.epochs.append(int(state.epoch))
        losses = state.losses.detach().cpu().numpy()
        if self.max_replicas is not None:
            losses = losses[: self.max_replicas]
        self.loss.append(losses)
        if self.record_x:
            x = state.x.detach().cpu().numpy()
            if self.max_replicas is not None:
                x = x[: self.max_replicas]
            self.x.append(x)


class TrajectoryTracker(Callback):
    """Track a secondary problem's objective per epoch.

    Useful for e.g. monitoring the "true" MIS size while optimising a
    penalised QUBO formulation.
    """

    def __init__(self, aux_problem, mode: str = "mean") -> None:
        if mode not in ("mean", "min"):
            raise ValueError("mode must be 'mean' or 'min'")
        self.aux_problem = aux_problem
        self.mode = mode
        self.values: list[float] = []

    def on_epoch_end(self, state: CallbackState) -> None:
        with torch.no_grad():
            x_disc = state.relaxation.project(state.x)
            loss_aux = self.aux_problem.loss_fn(x_disc)
            val = -loss_aux.mean().item() if self.mode == "mean" else -loss_aux.min().item()
        self.values.append(float(val))
