"""Black-box optimisation diagnostics."""

from __future__ import annotations

import numpy as np

from qqa.blackbox.solver import BlackBoxResult


def plot_blackbox(
    result: BlackBoxResult,
    *,
    backend: str = "plotly",
    title: str = "QQA black-box optimisation",
    show: bool = True,
):
    """Plot incumbent convergence, feasibility, and trust-region adaptation."""
    if not isinstance(result, BlackBoxResult):
        raise TypeError("result must be a BlackBoxResult.")
    if backend not in {"plotly", "matplotlib"}:
        raise ValueError("backend must be 'plotly' or 'matplotlib'.")
    if backend == "matplotlib":
        return _plot_matplotlib(result, title, show)
    return _plot_plotly(result, title, show)


def _plot_matplotlib(result, title, show):
    import matplotlib.pyplot as plt

    history = result.history
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    evaluations = history["evaluations"]
    axes[0].plot(evaluations, history["best_value"], marker="o")
    axes[0].set(xlabel="Evaluations", ylabel="Incumbent objective", title="Convergence")
    axes[1].semilogy(
        evaluations,
        np.maximum(history["best_violation"], 1e-14),
        marker="o",
        color="tab:red",
    )
    axes[1].set(xlabel="Evaluations", ylabel="Violation", title="Feasibility")
    axes[2].plot(evaluations, history["trust_radius"], marker="o", color="tab:green")
    axes[2].set(xlabel="Evaluations", ylabel="Radius", title="Adaptive trust region")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def _plot_plotly(result, title, show):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install `qqa[plotly]` for the Plotly backend.") from exc

    history = result.history
    evaluations = history["evaluations"]
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Convergence", "Constraint violation", "Trust region"),
    )
    for column, y, name, color in (
        (1, history["best_value"], "Best objective", "#2563eb"),
        (2, history["best_violation"], "Violation", "#dc2626"),
        (3, history["trust_radius"], "Radius", "#16a34a"),
    ):
        figure.add_trace(
            go.Scatter(
                x=evaluations,
                y=y,
                mode="lines+markers",
                name=name,
                line={"color": color},
            ),
            row=1,
            col=column,
        )
    figure.update_yaxes(type="log", row=1, col=2)
    figure.update_xaxes(title="Evaluations")
    figure.update_layout(title=title, template="plotly_white", height=450, showlegend=False)
    if show:
        figure.show()
    return figure


__all__ = ["plot_blackbox"]
