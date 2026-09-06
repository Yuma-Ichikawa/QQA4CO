"""Primal/dual/event cockpit for backend-neutral :class:`SolveResult`."""

from __future__ import annotations

from typing import Any


def _event_rows(result: Any) -> list[dict[str, Any]]:
    return [
        event.to_dict() if callable(getattr(event, "to_dict", None)) else dict(event)
        for event in getattr(result, "events", ())
    ]


def _short_label(value: str, maximum: int = 28) -> str:
    return value if len(value) <= maximum else value[: maximum - 1] + "…"


def plot_optimization_cockpit(
    result: Any,
    *,
    backend: str = "plotly",
    title: str = "QQA Optimization Cockpit",
    show: bool = False,
):
    """Render anytime primal/dual progress, feasibility, and phase timings."""
    if backend not in {"plotly", "matplotlib"}:
        raise ValueError("backend must be 'plotly' or 'matplotlib'.")
    rows = _event_rows(result)
    primal = [row for row in rows if row.get("kind") == "IncumbentImproved"]
    dual = [row for row in rows if row.get("kind") == "DualBoundImproved"]
    timing = getattr(result, "timings", None)
    phases = ["compile", "warmup", "search", "repair", "certification"]
    phase_values = [float(getattr(timing, name, 0.0)) for name in phases]
    violations = getattr(result, "violations", None)
    constraint_names = [row.name for row in getattr(violations, "rows", ())]
    constraint_labels = [_short_label(name) for name in constraint_names]
    constraint_values = [max(0.0, row.scaled_residual) for row in getattr(violations, "rows", ())]
    objective_value = getattr(result, "best_obj", None)
    objective_number = None if objective_value is None else float(objective_value)
    bound_value = getattr(result, "best_bound", None)
    status = getattr(result, "status", "unknown")
    status_text = str(getattr(status, "value", status))
    guarantee = getattr(result, "guarantee_level", "unknown")
    guarantee_text = str(getattr(guarantee, "value", guarantee))
    feasibility = getattr(getattr(violations, "status", None), "value", "unknown")
    if feasibility == "unknown":
        empty_constraint_message = "Constraint verification unavailable"
    elif feasibility == "feasible":
        empty_constraint_message = "No declared constraints<br>Variable domains verified"
    else:
        empty_constraint_message = "No residual details available<br>Candidate is infeasible"

    if backend == "plotly":
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as exc:
            raise ImportError("Install `qqa[plotly]` for the Plotly cockpit.") from exc
        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Anytime primal / dual",
                "Phase timing",
                "Constraint residual",
                "Outcome",
            ),
            specs=[[{}, {}], [{}, {"type": "indicator"}]],
        )
        if primal:
            figure.add_trace(
                go.Scatter(
                    x=[row["elapsed_seconds"] for row in primal],
                    y=[row["payload"]["objective"] for row in primal],
                    mode="lines+markers",
                    name="Primal incumbent",
                ),
                row=1,
                col=1,
            )
        if dual:
            figure.add_trace(
                go.Scatter(
                    x=[row["elapsed_seconds"] for row in dual],
                    y=[row["payload"]["bound"] for row in dual],
                    mode="lines+markers",
                    name="Dual bound",
                ),
                row=1,
                col=1,
            )
        figure.add_trace(go.Bar(x=phases, y=phase_values, name="Seconds"), row=1, col=2)
        if constraint_values:
            figure.add_trace(
                go.Bar(
                    x=constraint_values,
                    y=constraint_labels,
                    customdata=constraint_names,
                    orientation="h",
                    hovertemplate="%{customdata}<br>scaled residual=%{x:.4g}<extra></extra>",
                    name="Scaled residual",
                ),
                row=2,
                col=1,
            )
        else:
            figure.add_annotation(
                text=empty_constraint_message,
                showarrow=False,
                font={"size": 16, "color": "#475569"},
                row=2,
                col=1,
            )
        if objective_number is None:
            figure.add_annotation(
                text=f"{status_text} · {guarantee_text}<br>objective unavailable",
                showarrow=False,
                font={"size": 16, "color": "#475569"},
                row=2,
                col=2,
            )
        else:
            indicator_options: dict[str, Any] = {
                "mode": "number" if bound_value is None else "number+delta",
                "value": objective_number,
                "number": {"valueformat": ".6g"},
                "title": {
                    "text": f"{status_text} · {guarantee_text}<br>"
                    + ("objective" if bound_value is None else "objective / bound")
                },
            }
            if bound_value is not None:
                indicator_options["delta"] = {"reference": float(bound_value)}
            figure.add_trace(go.Indicator(**indicator_options), row=2, col=2)
        figure.update_layout(title={"text": title, "x": 0.5}, template="plotly_white", height=720)
        figure.update_xaxes(title_text="Elapsed seconds", row=1, col=1)
        figure.update_xaxes(
            title_text="Scaled residual" if constraint_values else None,
            visible=bool(constraint_values),
            row=2,
            col=1,
        )
        if show:
            figure.show()
        return figure

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor="white")
    if primal:
        axes[0, 0].plot(
            [row["elapsed_seconds"] for row in primal],
            [row["payload"]["objective"] for row in primal],
            label="Primal incumbent",
        )
    if dual:
        axes[0, 0].plot(
            [row["elapsed_seconds"] for row in dual],
            [row["payload"]["bound"] for row in dual],
            label="Dual bound",
        )
    axes[0, 0].set_xlabel("Elapsed seconds")
    if primal or dual:
        axes[0, 0].legend()
    axes[0, 1].bar(phases, phase_values, color="#2563eb")
    axes[0, 1].tick_params(axis="x", rotation=25)
    if constraint_values:
        axes[1, 0].barh(constraint_labels, constraint_values, color="#dc2626")
        axes[1, 0].set_xlabel("Scaled residual")
        axes[1, 0].tick_params(axis="y", labelsize=8)
        axes[1, 0].invert_yaxis()
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(
            0.5,
            0.55,
            empty_constraint_message.replace("<br>", "\n"),
            ha="center",
            va="center",
            color="#475569",
            fontsize=14,
        )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.5,
        0.55,
        f"status: {status_text}\nobjective: "
        f"{'unavailable' if objective_number is None else f'{objective_number:.6g}'}\n"
        f"feasibility: {feasibility}\nguarantee: {guarantee_text}",
        ha="center",
        va="center",
        fontsize=14,
    )
    figure.suptitle(title)
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes


__all__ = ["plot_optimization_cockpit"]
