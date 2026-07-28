"""Callbacks for the QQA annealing loop.

Callbacks receive a ``CallbackState`` snapshot at the end of every epoch and
can record metrics, adjust hyper-parameters, or track auxiliary objectives.
"""

from __future__ import annotations

import math
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
    """Record loss / penalty / diversity statistics per epoch.

    Performance notes
    -----------------
    The recorder is on the hot path of every annealing step. Naively calling
    ``tensor.item()`` for every metric forces a CUDA ``device->host`` sync at
    each epoch and dominates the wall-clock when the kernels themselves are
    cheap (small problems / GPU). To avoid that:

    * Per-epoch scalars are kept as **GPU 0-dim tensors** and appended to a
      Python list. There is no ``.item()`` call inside :meth:`on_epoch_end`.
    * :meth:`on_train_end` performs a **single** ``torch.stack(...).cpu()``
      per metric, which costs one sync regardless of ``num_epochs``.
    * ``stride`` skips intermediate epochs entirely (the last epoch is always
      recorded so that final-state observers see a non-empty history).

    The exposed ``self.history`` dict is still a ``dict[str, list[float]]``
    (or ``list[list[float]]`` for ``best_obj`` of batched-instance problems),
    so existing code that reads ``recorder.history["loss_mean"][-1]`` is
    unchanged.
    """

    def __init__(self, *, stride: int = 1) -> None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}.")
        self.stride = int(stride)
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
        # Per-epoch GPU scalars accumulated until ``on_train_end``.
        self._loss_mean_buf: list[torch.Tensor] = []
        self._loss_std_buf: list[torch.Tensor] = []
        self._loss_min_buf: list[torch.Tensor] = []
        self._penalty_mean_buf: list[torch.Tensor] = []
        self._penalty_std_buf: list[torch.Tensor] = []
        self._diversity_buf: list[torch.Tensor] = []
        # ``bg`` is a Python float and ``best_obj`` is already host-side
        # (or a numpy array for batched problems), so they cost nothing to
        # append directly.
        self._bg_buf: list[float] = []
        self._best_obj_buf: list = []

    def _should_record(self, state: CallbackState) -> bool:
        # Always record the final epoch so post-hoc consumers see the
        # terminal state regardless of stride.
        return state.epoch % self.stride == 0 or state.epoch == state.num_epochs - 1

    def on_epoch_end(self, state: CallbackState) -> None:
        if not self._should_record(state):
            return
        losses = state.losses.detach()
        penalties = state.penalties.detach()

        # Keep all reductions on the device. ``losses.std()`` of a 1-element
        # tensor returns NaN; mirror the legacy behaviour (0.0) by using a
        # zero-valued GPU scalar instead of forcing a device sync to branch.
        self._loss_mean_buf.append(losses.mean())
        if losses.numel() > 1:
            self._loss_std_buf.append(losses.std())
        else:
            self._loss_std_buf.append(torch.zeros((), device=losses.device, dtype=losses.dtype))
        self._loss_min_buf.append(losses.min())

        self._penalty_mean_buf.append(penalties.mean())
        if penalties.numel() > 1:
            self._penalty_std_buf.append(penalties.std())
        else:
            self._penalty_std_buf.append(
                torch.zeros((), device=penalties.device, dtype=penalties.dtype)
            )

        div = state.diversity
        if torch.is_tensor(div):
            self._diversity_buf.append(div.detach())
        else:
            # Float diversity (e.g. sol_size == 1 path) — wrap as a CPU scalar
            # so the final ``stack`` works without a device transfer.
            self._diversity_buf.append(torch.tensor(float(div)))

        self._bg_buf.append(state.bg)
        bo = state.best_obj
        if hasattr(bo, "tolist"):
            self._best_obj_buf.append(bo.tolist())
        else:
            self._best_obj_buf.append(float(bo))

    def on_train_end(self, state: CallbackState) -> None:  # noqa: ARG002 - state unused
        # Bulk device->host transfer: one sync per metric instead of per epoch.
        def _flush(buf: list[torch.Tensor]) -> list[float]:
            if not buf:
                return []
            # Tensors may live on different devices if a custom callback added
            # entries; promote to CPU first to keep ``stack`` happy.
            return [float(t) for t in torch.stack([t.detach().to("cpu") for t in buf]).tolist()]

        self.history["loss_mean"] = _flush(self._loss_mean_buf)
        self.history["loss_std"] = _flush(self._loss_std_buf)
        self.history["loss_min"] = _flush(self._loss_min_buf)
        self.history["penalty_mean"] = _flush(self._penalty_mean_buf)
        self.history["penalty_std"] = _flush(self._penalty_std_buf)
        self.history["diversity"] = _flush(self._diversity_buf)
        self.history["bg"] = list(self._bg_buf)
        self.history["best_obj"] = list(self._best_obj_buf)


class AutoDivTuner(Callback):
    """Adaptively tune ``div_param`` to target a desired diversity ratio.

    At each epoch: ``ratio = diversity / N``. The controller
    nudges ``div_param`` by ``lr * (ratio - target)`` and clips to ``[0, 1]``.

    ``Relaxation.diversity`` is already a standard deviation over the
    population axis, so dividing by ``sol_size`` a second time would make the
    measured ratio shrink as more replicas are added.
    """

    def __init__(self, target: float = 0.3, lr: float = 1e-3) -> None:
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"target must be in [0, 1], got {target}.")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}.")
        self.target = target
        self.lr = lr

    def on_epoch_end(self, state: CallbackState) -> None:
        sol_size = state.x.shape[0]
        if sol_size <= 1:
            return
        num_vars = state.relaxation.num_variables(state.problem)
        # diversity is summed across all non-batch axes, so for batched-instance
        # problems (shape (B, I, N)) the denominator must include I to keep
        # ``ratio`` in roughly the same range as the single-instance path.
        num_inst = int(getattr(state.problem, "num_instance", 1) or 1)
        div_val = (
            float(state.diversity.item())
            if torch.is_tensor(state.diversity)
            else float(state.diversity)
        )
        ratio = div_val / (num_vars * num_inst)
        # Negative feedback: increase the diversity weight when observed
        # diversity is below target and decrease it when diversity is high.
        diff = self.target - ratio
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
