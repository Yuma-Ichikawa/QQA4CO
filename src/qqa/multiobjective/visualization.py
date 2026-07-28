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
    """Plot 2-D/3-D fronts or parallel coordinates for 4+ objectives."""
    if not isinstance(result, ParetoResult):
        raise TypeError("result must be a ParetoResult.")
    values = result.objectives.detach().cpu().numpy()
    if backend not in {"plotly", "matplotlib"}:
        raise ValueError("backend must be 'plotly' or 'matplotlib'.")
    if backend == "matplotlib":
        return _plot_matplotlib(values, result.objective_names, title, show)
    return _plot_plotly(values, result.objective_names, title, show)


def _plot_matplotlib(values, names, title, show):
    import matplotlib.pyplot as plt

    count = values.shape[1]
    if count == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(values[:, 0], values[:, 1], c=np.arange(len(values)), cmap="viridis")
        ax.set(xlabel=names[0], ylabel=names[1], title=title)
    elif count == 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(values[:, 0], values[:, 1], values[:, 2], c=np.arange(len(values)))
        ax.set(xlabel=names[0], ylabel=names[1], zlabel=names[2], title=title)
    else:
        fig, ax = plt.subplots(figsize=(max(9, count * 1.5), 6))
        lower, upper = values.min(axis=0), values.max(axis=0)
        normalised = (values - lower) / np.maximum(upper - lower, 1e-12)
        for row in normalised:
            ax.plot(range(count), row, alpha=0.35)
        ax.set_xticks(range(count), names, rotation=25, ha="right")
        ax.set(ylabel="Normalised objective", title=title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _plot_plotly(values, names, title, show):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional installs
        raise ImportError("Install `qqa[plotly]` for the Plotly backend.") from exc

    count = values.shape[1]
    if count == 2:
        figure = go.Figure(
            go.Scatter(
                x=values[:, 0],
                y=values[:, 1],
                mode="markers+lines",
                marker={"size": 8, "color": np.arange(len(values)), "colorscale": "Viridis"},
                text=[f"solution {index}" for index in range(len(values))],
                hovertemplate="%{text}<br>%{x:.6g}<br>%{y:.6g}<extra></extra>",
            )
        )
        figure.update_xaxes(title=names[0])
        figure.update_yaxes(title=names[1])
    elif count == 3:
        figure = go.Figure(
            go.Scatter3d(
                x=values[:, 0],
                y=values[:, 1],
                z=values[:, 2],
                mode="markers",
                marker={"size": 5, "color": np.arange(len(values)), "colorscale": "Viridis"},
            )
        )
        figure.update_layout(
            scene={
                "xaxis_title": names[0],
                "yaxis_title": names[1],
                "zaxis_title": names[2],
            }
        )
    else:
        figure = go.Figure(
            go.Parcoords(
                line={"color": np.arange(len(values)), "colorscale": "Viridis"},
                dimensions=[
                    {"label": name, "values": values[:, index]} for index, name in enumerate(names)
                ],
            )
        )
    figure.update_layout(title=title, template="plotly_white")
    if show:
        figure.show()
    return figure


__all__ = ["plot_pareto"]
