"""Backward-compatible wrappers for the original ``batch_annealing_*`` API.

These functions exist so that code written against QQA4CO pre-0.2 continues
to work. They emit a :class:`DeprecationWarning` and forward to the unified
:func:`qqa.anneal` loop.

New code should use :func:`qqa.anneal` directly.
"""

from __future__ import annotations

import warnings

from qqa.annealing import anneal
from qqa.callbacks import AutoDivTuner, TrajectoryTracker
from qqa.schedule import LinearBGSchedule


def _deprecated(old: str, new: str = "qqa.anneal") -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in a future release; use {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def batch_annealing(
    problem,
    sol_size: int = 100,
    learning_rate: float = 1.0,
    temp: float = 0.0,
    min_bg: float = -2.0,
    max_bg: float = 0.1,
    curve_rate: int = 2,
    div_param: float = 0.0,
    num_epochs: int = 10_000,
    check_interval: int = 1000,
    device: str = "cpu",
    plot_dynamics: bool = False,
    auto_divparam: bool = False,
    div_target: float = 0.3,
    div_param_lr: float = 1e-3,
):
    """Legacy single-instance batch annealing. Returns ``(best_sol, best_obj, runtime)``."""
    _deprecated("qqa.legacy.batch_annealing")
    callbacks = []
    if auto_divparam:
        callbacks.append(AutoDivTuner(target=div_target, lr=div_param_lr))
    result = anneal(
        problem,
        sol_size=sol_size,
        learning_rate=learning_rate,
        temp=temp,
        schedule=LinearBGSchedule(min_bg, max_bg),
        curve_rate=curve_rate,
        div_param=div_param,
        num_epochs=num_epochs,
        check_interval=check_interval,
        device=device,
        callbacks=callbacks,
        verbose=True,
    )
    if plot_dynamics:
        from qqa.visualization import plot_history

        plot_history(result, title="Batch Annealing (Single-Instance)")
    return result.best_sol, result.best_obj, result.runtime


def batch_instance_annealing(
    problem,
    sol_size: int = 100,
    learning_rate: float = 1.0,
    temp: float = 0.0,
    min_bg: float = -2.0,
    max_bg: float = 0.1,
    curve_rate: int = 2,
    div_param: float = 0.0,
    num_epochs: int = 10_000,
    check_interval: int = 1000,
    device: str = "cpu",
    plot_dynamics: bool = False,
):
    """Legacy vectorised multi-instance annealing. Returns ``(best_sol, best_obj, runtime)``."""
    _deprecated("qqa.legacy.batch_instance_annealing")
    result = anneal(
        problem,
        sol_size=sol_size,
        learning_rate=learning_rate,
        temp=temp,
        schedule=LinearBGSchedule(min_bg, max_bg),
        curve_rate=curve_rate,
        div_param=div_param,
        num_epochs=num_epochs,
        check_interval=check_interval,
        device=device,
        verbose=True,
    )
    if plot_dynamics:
        from qqa.visualization import plot_history

        plot_history(result, title="Batch Instance Annealing")
    return result.best_sol, result.best_obj, result.runtime


def batch_annealing_mis_trajectory(
    problem,
    problem_P1,
    sol_size: int = 100,
    learning_rate: float = 1.0,
    temp: float = 0.0,
    min_bg: float = -2.0,
    max_bg: float = 0.1,
    curve_rate: int = 2,
    div_param: float = 0.0,
    num_epochs: int = 10_000,
    check_interval: int = 1000,
    mode: str = "mean",
    device: str = "cpu",
    plot_dynamics: bool = False,
    auto_divparam: bool = False,
    div_target: float = 0.3,
    div_param_lr: float = 1e-3,
):
    """Legacy MIS-trajectory annealing. Returns ``(best_obj, runtime, dynamics_memory)``."""
    _deprecated("qqa.legacy.batch_annealing_mis_trajectory")
    tracker = TrajectoryTracker(problem_P1, mode=mode)
    callbacks = [tracker]
    if auto_divparam:
        callbacks.append(AutoDivTuner(target=div_target, lr=div_param_lr))
    result = anneal(
        problem,
        sol_size=sol_size,
        learning_rate=learning_rate,
        temp=temp,
        schedule=LinearBGSchedule(min_bg, max_bg),
        curve_rate=curve_rate,
        div_param=div_param,
        num_epochs=num_epochs,
        check_interval=check_interval,
        device=device,
        callbacks=callbacks,
        verbose=True,
    )
    if plot_dynamics:
        from qqa.visualization import plot_history

        plot_history(result, title="Batch Annealing MIS Trajectory")
    dynamics_memory = {"MIS_DYNAMICS": tracker.values}
    return result.best_obj, result.runtime, dynamics_memory


def batch_annealing_categorical(
    problem,
    sol_size: int = 100,
    learning_rate: float = 1.0,
    temp: float = 0.0,
    min_bg: float = -2.0,
    max_bg: float = 0.1,
    curve_rate: int = 2,
    div_param: float = 0.0,
    num_epochs: int = 10_000,
    check_interval: int = 1000,
    device: str = "cpu",
    plot_dynamics: bool = False,
    auto_divparam: bool = False,
    div_target: float = 0.3,
    div_param_lr: float = 1e-3,
):
    """Legacy categorical annealing. Returns ``(best_string, best_loss, runtime)``.

    ``best_string`` is the argmax class per node, extracted from the one-hot
    best solution (shape ``(N,)``), matching the legacy signature.
    """
    _deprecated("qqa.legacy.batch_annealing_categorical")
    callbacks = []
    if auto_divparam:
        callbacks.append(AutoDivTuner(target=div_target, lr=div_param_lr))
    result = anneal(
        problem,
        sol_size=sol_size,
        learning_rate=learning_rate,
        temp=temp,
        schedule=LinearBGSchedule(min_bg, max_bg),
        curve_rate=curve_rate,
        div_param=div_param,
        num_epochs=num_epochs,
        check_interval=check_interval,
        device=device,
        callbacks=callbacks,
        verbose=True,
    )
    # ``result.best_sol`` is the single winning replica with shape ``(N, K)``.
    best_sol = result.best_sol
    best_string = best_sol.argmax(dim=-1).detach()
    if plot_dynamics:
        from qqa.visualization import plot_history

        plot_history(result, title="Batch Annealing (Categorical)")
    return best_string, result.best_obj, result.runtime
