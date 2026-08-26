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

    * Per-epoch scalars are written into preallocated device tensors. There
      is no ``.item()`` call inside :meth:`on_epoch_end`.
    * :meth:`on_train_end` slices and transfers each buffer once, which costs
      one synchronisation regardless of ``num_epochs``.
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
        # A preallocated device ring avoids one Python tensor object per
        # metric per epoch. Columns are loss mean/std/min, penalty mean/std,
        # diversity, and BG.
        self._device_history: torch.Tensor | None = None
        self._best_device_history: torch.Tensor | None = None
        self._records = 0

    def on_train_begin(self, state: CallbackState) -> None:
        records = max(1, (state.num_epochs + self.stride - 1) // self.stride + 1)
        dtype = state.x.dtype if state.x.is_floating_point() else torch.float32
        self._device_history = torch.empty(
            (records, 7),
            device=state.x.device,
            dtype=dtype,
        )
        best = torch.as_tensor(state.best_obj, device=state.x.device, dtype=dtype)
        self._best_device_history = torch.empty(
            (records, *best.shape),
            device=state.x.device,
            dtype=dtype,
        )
        self._records = 0

    def _should_record(self, state: CallbackState) -> bool:
        # Always record the final epoch so post-hoc consumers see the
        # terminal state regardless of stride.
        return state.epoch % self.stride == 0 or state.epoch == state.num_epochs - 1

    def on_epoch_end(self, state: CallbackState) -> None:
        if not self._should_record(state):
            return
        losses = state.losses.detach()
        penalties = state.penalties.detach()

        if self._device_history is None:
            self.on_train_begin(state)
        assert self._device_history is not None
        loss_std = losses.std() if losses.numel() > 1 else losses.new_zeros(())
        penalty_std = penalties.std() if penalties.numel() > 1 else penalties.new_zeros(())
        div = state.diversity
        div = div.detach() if torch.is_tensor(div) else losses.new_tensor(float(div))
        self._device_history[self._records] = torch.stack(
            (
                losses.mean(),
                loss_std,
                losses.min(),
                penalties.mean(),
                penalty_std,
                div.to(losses),
                losses.new_tensor(state.bg),
            )
        ).to(self._device_history)
        assert self._best_device_history is not None
        self._best_device_history[self._records] = torch.as_tensor(
            state.best_obj,
            device=self._best_device_history.device,
            dtype=self._best_device_history.dtype,
        )
        self._records += 1

    def on_train_end(self, state: CallbackState) -> None:  # noqa: ARG002 - state unused
        matrix = (
            torch.empty((0, 7))
            if self._device_history is None
            else self._device_history[: self._records].detach().cpu()
        )
        for index, name in enumerate(
            ("loss_mean", "loss_std", "loss_min", "penalty_mean", "penalty_std", "diversity", "bg")
        ):
            self.history[name] = [float(value) for value in matrix[:, index].tolist()]
        best = (
            []
            if self._best_device_history is None
            else self._best_device_history[: self._records].detach().cpu().tolist()
        )
        self.history["best_obj"] = [
            float(value) if not isinstance(value, list) else value for value in best
        ]


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
