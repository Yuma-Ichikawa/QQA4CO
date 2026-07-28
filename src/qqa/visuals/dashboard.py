"""High-density visual diagnostics for solver results."""

from __future__ import annotations

from typing import Any

import numpy as np

from qqa.visuals._data import constraint_rows, solution_rows, trajectory

_KIND_COLORS = {
    "binary": "#0f766e",
    "integer": "#b45309",
    "real": "#6d28d9",
    "variable": "#1e3a8a",
}


def plot_result_dashboard(
    result: Any,
    problem: Any | None,
    *,
    backend: str,
    title: str,
    show: bool,
):
    """Render convergence, variables, constraints, and search dynamics."""
    if backend == "plotly":
        return _dashboard_plotly(result, problem, title=title, show=show)
    return _dashboard_matplotlib(result, problem, title=title, show=show)


def plot_variable_solution(
    result: Any,
    problem: Any | None,
    *,
    backend: str,
    title: str,
    show: bool,
):
    """Plot physical values and domains for every solution coordinate."""
    rows = solution_rows(result, problem)
    labels = [row["label"] for row in rows]
    normalised = [row["normalised"] for row in rows]
    colors = [_KIND_COLORS[row["kind"]] for row in rows]

    if backend == "plotly":
        import plotly.graph_objects as go

        custom = [[row["value"], row["lower"], row["upper"], row["kind"]] for row in rows]
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=normalised,
                marker_color=colors,
                customdata=custom,
                hovertemplate=(
                    "<b>%{x}</b><br>value=%{customdata[0]:.6g}"
                    "<br>domain=[%{customdata[1]:.6g}, %{customdata[2]:.6g}]"
                    "<br>type=%{customdata[3]}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title={"text": title, "x": 0.5},
            xaxis_title="Variable",
            yaxis_title="Normalised position in domain",
            yaxis_range=[0, 1.05],
            template="plotly_white",
            height=440,
        )
        if show:
            fig.show()
        return fig

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.55), 4.5), facecolor="white")
    ax.bar(labels, normalised, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalised position in domain")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_constraint_diagnostics(
    result: Any,
    problem: Any | None,
    *,
    backend: str,
    title: str,
    show: bool,
):
    """Plot raw constraint violations with feasibility thresholds."""
    rows = constraint_rows(result, problem)
    if not rows:
        raise ValueError("No constraint diagnostics are available for this result.")
    names = [str(row["name"]) for row in rows]
    violations = [max(0.0, float(row["violation"])) for row in rows]
    tolerances = [float(row["tolerance"]) for row in rows]
    colors = ["#0f766e" if row["feasible"] else "#be123c" for row in rows]

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_bar(
            x=names,
            y=violations,
            name="Violation",
            marker_color=colors,
            customdata=np.asarray(tolerances)[:, None],
            hovertemplate=(
                "<b>%{x}</b><br>violation=%{y:.6g}<br>tolerance=%{customdata[0]:.6g}<extra></extra>"
            ),
        )
        fig.add_scatter(
            x=names,
            y=tolerances,
            name="Tolerance",
            mode="markers",
            marker={"symbol": "line-ew-open", "size": 18, "color": "#111827"},
        )
        fig.update_layout(
            title={"text": title, "x": 0.5},
            yaxis_title="Raw violation",
            template="plotly_white",
            height=420,
        )
        if show:
            fig.show()
        return fig

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.2), 4.5), facecolor="white")
    ax.bar(names, violations, color=colors, label="Violation")
    ax.scatter(names, tolerances, marker="_", s=250, color="#111827", label="Tolerance")
    ax.set_ylabel("Raw violation")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _dashboard_plotly(result: Any, problem: Any | None, *, title: str, show: bool):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = solution_rows(result, problem)
    constraints = constraint_rows(result, problem)
    epochs, best = trajectory(result)
    history = getattr(result, "history", {}) or {}

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Best-known objective",
            "Solution across declared domains",
            "Constraint health",
            "Annealing dynamics",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
    )
    if best.size:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=best,
                name="Best objective",
                mode="lines",
                line={"color": "#0f766e", "width": 2.5},
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=[row["label"] for row in rows],
            y=[row["normalised"] for row in rows],
            name="Variables",
            marker_color=[_KIND_COLORS[row["kind"]] for row in rows],
            customdata=[[row["value"], row["lower"], row["upper"], row["kind"]] for row in rows],
            hovertemplate=(
                "<b>%{x}</b><br>value=%{customdata[0]:.6g}"
                "<br>domain=[%{customdata[1]:.6g}, %{customdata[2]:.6g}]"
                "<br>type=%{customdata[3]}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    if constraints:
        fig.add_trace(
            go.Bar(
                x=[row["name"] for row in constraints],
                y=[max(0.0, float(row["violation"])) for row in constraints],
                name="Violation",
                marker_color=["#0f766e" if row["feasible"] else "#be123c" for row in constraints],
                customdata=[
                    [row["lhs"], row["sense"], row["rhs"], row["tolerance"]] for row in constraints
                ],
                hovertemplate=(
                    "<b>%{x}</b><br>violation=%{y:.6g}<br>"
                    "%{customdata[0]:.6g} %{customdata[1]} %{customdata[2]:.6g}"
                    "<br>tolerance=%{customdata[3]:.6g}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[row["name"] for row in constraints],
                y=[row["tolerance"] for row in constraints],
                name="Tolerance",
                mode="markers",
                marker={"symbol": "line-ew-open", "size": 16, "color": "#111827"},
            ),
            row=2,
            col=1,
        )
    else:
        fig.add_annotation(
            text="No explicit constraints",
            x=0.2,
            y=0.2,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#64748b"},
        )

    diversity = np.asarray(history.get("diversity", []), dtype=float)
    bg = np.asarray(history.get("bg", []), dtype=float)
    if diversity.size:
        fig.add_trace(
            go.Scatter(
                x=np.arange(diversity.size),
                y=diversity,
                name="Diversity",
                mode="lines",
                line={"color": "#6d28d9", "width": 2},
            ),
            row=2,
            col=2,
        )
    if bg.size:
        fig.add_trace(
            go.Scatter(
                x=np.arange(bg.size),
                y=bg,
                name="BG schedule",
                mode="lines",
                line={"color": "#b45309", "width": 2, "dash": "dot"},
            ),
            row=2,
            col=2,
        )

    fig.update_yaxes(title_text="Objective", row=1, col=1)
    fig.update_yaxes(title_text="Normalised value", range=[0, 1.05], row=1, col=2)
    fig.update_yaxes(title_text="Raw violation", row=2, col=1)
    fig.update_yaxes(title_text="Metric value", row=2, col=2)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Variable", row=1, col=2)
    fig.update_xaxes(title_text="Constraint", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        height=780,
        barmode="overlay",
        legend={"orientation": "h", "y": -0.12},
        margin={"l": 60, "r": 30, "t": 90, "b": 90},
    )
    if show:
        fig.show()
    return fig


def _dashboard_matplotlib(result: Any, problem: Any | None, *, title: str, show: bool):
    import matplotlib.pyplot as plt

    rows = solution_rows(result, problem)
    constraints = constraint_rows(result, problem)
    epochs, best = trajectory(result)
    history = getattr(result, "history", {}) or {}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    ax_obj, ax_var, ax_con, ax_dyn = axes.ravel()
    if best.size:
        ax_obj.plot(epochs, best, color="#0f766e", linewidth=2.5)
    ax_obj.set(title="Best-known objective", xlabel="Epoch", ylabel="Objective")

    ax_var.bar(
        [row["label"] for row in rows],
        [row["normalised"] for row in rows],
        color=[_KIND_COLORS[row["kind"]] for row in rows],
    )
    ax_var.set(title="Solution across declared domains", ylabel="Normalised value", ylim=(0, 1.05))
    ax_var.tick_params(axis="x", rotation=45)

    if constraints:
        names = [row["name"] for row in constraints]
        ax_con.bar(
            names,
            [max(0.0, float(row["violation"])) for row in constraints],
            color=["#0f766e" if row["feasible"] else "#be123c" for row in constraints],
        )
        ax_con.scatter(
            names,
            [row["tolerance"] for row in constraints],
            marker="_",
            s=220,
            color="#111827",
        )
    else:
        ax_con.text(0.5, 0.5, "No explicit constraints", ha="center", va="center")
    ax_con.set(title="Constraint health", ylabel="Raw violation")

    diversity = np.asarray(history.get("diversity", []), dtype=float)
    bg = np.asarray(history.get("bg", []), dtype=float)
    if diversity.size:
        ax_dyn.plot(diversity, label="Diversity", color="#6d28d9")
    if bg.size:
        ax_dyn.plot(bg, label="BG schedule", color="#b45309", linestyle="--")
    ax_dyn.set(title="Annealing dynamics", xlabel="Epoch", ylabel="Metric value")
    if diversity.size or bg.size:
        ax_dyn.legend()

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
