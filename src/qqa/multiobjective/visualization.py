"""Visualisations for two-, three-, and many-objective Pareto fronts."""

from __future__ import annotations

import numpy as np

from qqa.multiobjective.solver import ParetoResult


def plot_pareto(
    result: ParetoResult,
    *,
    backend: str = "plotly",
    title: str = "QQA Pareto front",
    show: bool = True,
):
    """Plot 2-D/3-D fronts or parallel coordinates for 4+ objectives.

    The scale-invariant knee selected by :meth:`ParetoResult.select` is
    highlighted so a user can distinguish a recommended compromise from the
    objective-axis extremes without manually normalising mixed units.
    """
    if not isinstance(result, ParetoResult):
        raise TypeError("result must be a ParetoResult.")
    values = result.objectives.detach().cpu().numpy()
    if backend not in {"plotly", "matplotlib"}:
        raise ValueError("backend must be 'plotly' or 'matplotlib'.")
    knee = result.select()
    if backend == "matplotlib":
        return _plot_matplotlib(
            values,
            result.objective_names,
            result.directions,
            knee,
            title,
            show,
        )
    return _plot_plotly(
        values,
        result.objective_names,
        result.directions,
        knee,
        title,
        show,
    )


def _axis_label(name, direction):
    arrow = "↓" if direction == "min" else "↑"
    return f"{name} ({arrow})"


def _plot_matplotlib(values, names, directions, knee, title, show):
    import matplotlib.pyplot as plt

    count = values.shape[1]
    labels = [
        _axis_label(name, direction) for name, direction in zip(names, directions, strict=True)
    ]
    if count == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(values[:, 0], values[:, 1], c=np.arange(len(values)), cmap="viridis")
        ax.scatter(
            values[knee, 0],
            values[knee, 1],
            marker="*",
            s=240,
            color="#dc2626",
            edgecolor="white",
            linewidth=1.2,
            label="Recommended knee",
            zorder=5,
        )
        ax.set(xlabel=labels[0], ylabel=labels[1], title=title)
        ax.legend()
    elif count == 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(values[:, 0], values[:, 1], values[:, 2], c=np.arange(len(values)))
        ax.scatter(
            *values[knee],
            marker="*",
            s=260,
            color="#dc2626",
            edgecolor="white",
            linewidth=1.2,
            label="Recommended knee",
        )
        ax.set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2], title=title)
        ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(max(9, count * 1.5), 6))
        lower, upper = values.min(axis=0), values.max(axis=0)
        normalised = (values - lower) / np.maximum(upper - lower, 1e-12)
        for row in normalised:
            ax.plot(range(count), row, alpha=0.35)
        ax.plot(
            range(count),
            normalised[knee],
            color="#dc2626",
            linewidth=3,
            marker="o",
            label="Recommended knee",
        )
        ax.set_xticks(range(count), labels, rotation=25, ha="right")
        ax.set(ylabel="Normalised objective", title=title)
        ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _plot_plotly(values, names, directions, knee, title, show):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional installs
        raise ImportError("Install `qqa[plotly]` for the Plotly backend.") from exc

    count = values.shape[1]
    labels = [
        _axis_label(name, direction) for name, direction in zip(names, directions, strict=True)
    ]
    if count == 2:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=values[:, 0],
                y=values[:, 1],
                mode="markers+lines",
                marker={"size": 8, "color": np.arange(len(values)), "colorscale": "Viridis"},
                text=[f"solution {index}" for index in range(len(values))],
                hovertemplate="%{text}<br>%{x:.6g}<br>%{y:.6g}<extra></extra>",
                name="Pareto front",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[values[knee, 0]],
                y=[values[knee, 1]],
                mode="markers",
                marker={"size": 16, "symbol": "star", "color": "#dc2626"},
                name="Recommended knee",
                hovertemplate="Recommended knee<br>%{x:.6g}<br>%{y:.6g}<extra></extra>",
            )
        )
        figure.update_xaxes(title=labels[0])
        figure.update_yaxes(title=labels[1])
    elif count == 3:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter3d(
                x=values[:, 0],
                y=values[:, 1],
                z=values[:, 2],
                mode="markers",
                marker={"size": 5, "color": np.arange(len(values)), "colorscale": "Viridis"},
                name="Pareto front",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[values[knee, 0]],
                y=[values[knee, 1]],
                z=[values[knee, 2]],
                mode="markers",
                marker={"size": 9, "symbol": "diamond", "color": "#dc2626"},
                name="Recommended knee",
            )
        )
        figure.update_layout(
            scene={
                "xaxis_title": labels[0],
                "yaxis_title": labels[1],
                "zaxis_title": labels[2],
            }
        )
    else:
        colors = np.zeros(len(values))
        colors[knee] = 1
        figure = go.Figure(
            go.Parcoords(
                line={
                    "color": colors,
                    "colorscale": [[0.0, "#94a3b8"], [0.5, "#94a3b8"], [1.0, "#dc2626"]],
                    "showscale": False,
                },
                dimensions=[
                    {"label": label, "values": values[:, index]}
                    for index, label in enumerate(labels)
                ],
            )
        )
    figure.update_layout(title=title, template="plotly_white")
    if show:
        figure.show()
    return figure


def plot_pareto_diagnostics(
    result: ParetoResult,
    *,
    backend: str = "plotly",
    title: str = "QQA Pareto search diagnostics",
    show: bool = True,
):
    """Plot archive growth, feasibility, violation, penalty, and restarts.

    This diagnostic view complements :func:`plot_pareto`: it makes adaptive
    constraint enforcement and basin-recovery events auditable instead of
    presenting only the final nondominated point cloud.
    """
    if not isinstance(result, ParetoResult):
        raise TypeError("result must be a ParetoResult.")
    if backend not in {"plotly", "matplotlib"}:
        raise ValueError("backend must be 'plotly' or 'matplotlib'.")
    history = result.history
    required = {"epoch", "pareto_size", "feasible_ratio", "mean_violation", "penalty_rho"}
    missing = sorted(required - history.keys())
    if missing:
        raise ValueError(f"Pareto history is missing diagnostics: {missing}.")
    if backend == "matplotlib":
        return _diagnostics_matplotlib(history, title, show)
    return _diagnostics_plotly(history, title, show)


def _restart_epochs(history):
    epochs = np.asarray(history["epoch"], dtype=float)
    restarts = np.asarray(history.get("restarts", np.zeros_like(epochs)), dtype=float)
    if not len(epochs):
        return np.asarray([], dtype=float)
    changed = np.diff(np.r_[0.0, restarts]) > 0
    return epochs[changed]


def _diagnostics_matplotlib(history, title, show):
    import matplotlib.pyplot as plt

    epochs = np.asarray(history["epoch"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="white")
    axes[0, 0].plot(epochs, history["pareto_size"], color="#2563eb")
    axes[0, 0].set(title="Nondominated archive", ylabel="Pareto solutions")
    axes[0, 1].plot(epochs, history["feasible_ratio"], color="#16a34a")
    axes[0, 1].set(title="Population feasibility", ylabel="Feasible share", ylim=(-0.02, 1.02))
    axes[1, 0].semilogy(
        epochs,
        np.maximum(np.asarray(history["mean_violation"], dtype=float), 1e-14),
        color="#dc2626",
    )
    axes[1, 0].set(title="Normalised constraint residual", ylabel="Mean violation")
    axes[1, 1].semilogy(
        epochs,
        np.maximum(np.asarray(history["penalty_rho"], dtype=float), 1e-14),
        color="#7c3aed",
    )
    axes[1, 1].set(title="Adaptive augmented Lagrangian", ylabel="Penalty ρ")
    for restart_epoch in _restart_epochs(history):
        for axis in axes.ravel():
            axis.axvline(restart_epoch, color="#f59e0b", alpha=0.25, linewidth=1)
    for axis in axes.ravel():
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def _diagnostics_plotly(history, title, show):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - optional installs
        raise ImportError("Install `qqa[plotly]` for the Plotly backend.") from exc

    epochs = history["epoch"]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Nondominated archive",
            "Population feasibility",
            "Normalised constraint residual",
            "Adaptive penalty ρ",
        ),
    )
    traces = (
        (1, 1, history["pareto_size"], "Pareto solutions", "#2563eb"),
        (1, 2, history["feasible_ratio"], "Feasible share", "#16a34a"),
        (2, 1, history["mean_violation"], "Mean violation", "#dc2626"),
        (2, 2, history["penalty_rho"], "Penalty ρ", "#7c3aed"),
    )
    for row, column, values, name, color in traces:
        figure.add_trace(
            go.Scatter(
                x=epochs,
                y=values,
                mode="lines",
                name=name,
                line={"color": color, "width": 2.5},
            ),
            row=row,
            col=column,
        )
    for restart_epoch in _restart_epochs(history):
        figure.add_vline(
            x=float(restart_epoch),
            line={"color": "#f59e0b", "width": 1, "dash": "dot"},
        )
    figure.update_yaxes(range=[0, 1], row=1, col=2)
    figure.update_yaxes(type="log", row=2, col=1)
    figure.update_yaxes(type="log", row=2, col=2)
    figure.update_xaxes(title_text="Epoch")
    figure.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        height=720,
        legend={"orientation": "h", "y": -0.12},
    )
    if show:
        figure.show()
    return figure


__all__ = ["plot_pareto", "plot_pareto_diagnostics"]
