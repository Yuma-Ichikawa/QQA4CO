"""Visualize page — inspect the last annealing run."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from _common import (  # noqa: E402
    apply_theme,
    palette,
    plotly_layout,
    render_score_card,
    theme_toggle_in_sidebar,
)

from qqa import visualization as viz  # noqa: E402

st.set_page_config(page_title="Visualize — QQA", page_icon="⚛️", layout="wide")
theme_toggle_in_sidebar()
apply_theme()
st.title("Visualize")

result = st.session_state.get("last_result")
problem = st.session_state.get("last_problem")
if result is None:
    st.warning("Run QQA on the Solve page first.")
    st.stop()

pop_tracker = st.session_state.get("last_pop_tracker")

# Score card at the top of the page so the headline metric is always visible.
raw = result.best_obj if isinstance(result.best_obj, float) else None
render_score_card(result.score, raw_loss=raw)

(
    tab_hist,
    tab_best,
    tab_sched,
    tab_sol,
    tab_pop,
    tab_pca,
    tab_ridge,
    tab_fate,
) = st.tabs(
    [
        "Dynamics",
        "Best trajectory",
        "Schedule",
        "Solution",
        "Parallel population",
        "PCA trajectory",
        "Ridgeline",
        "Replica fate",
    ]
)

def _retheme(fig):
    """Reskin a Plotly figure produced by ``viz`` to the current theme."""
    try:
        fig.update_layout(**plotly_layout())
    except Exception:
        pass
    return fig


with tab_hist:
    fig = viz.plot_history(result, backend="plotly", show=False)
    st.plotly_chart(_retheme(fig), width='stretch')

with tab_best:
    fig = viz.plot_best_trajectory(result, backend="plotly", show=False)
    st.plotly_chart(_retheme(fig), width='stretch')

with tab_sched:
    if result.history and "bg" in result.history:
        import plotly.graph_objects as go

        bg = np.asarray(result.history["bg"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(bg)),
                y=bg,
                mode="lines",
                line={"color": palette()["palette"][0], "width": 2},
            )
        )
        fig.update_layout(
            **plotly_layout(
                title={"text": "Applied bg schedule"},
                xaxis_title="Epoch",
                yaxis_title="bg",
                height=400,
            )
        )
        st.plotly_chart(fig, width='stretch')

with tab_sol:
    try:
        fig = viz.plot_solution_heatmap(result, problem=problem, backend="plotly", show=False)
        st.plotly_chart(_retheme(fig), width='stretch')
    except Exception as e:
        st.info(f"No solution heatmap available: {e}")

with tab_pop:
    if pop_tracker is None or not pop_tracker.loss:
        st.info("No population snapshots recorded for this run.")
    else:
        fig = viz.plot_population_evolution(pop_tracker, backend="plotly", show=False)
        st.plotly_chart(_retheme(fig), width='stretch')
        st.caption(
            "Each row is one of the `sol_size` replicas (sorted by final loss). "
            "Colour encodes per-replica loss."
        )

with tab_pca:
    if pop_tracker is None or not pop_tracker.x:
        st.info("No x-snapshots recorded (population_tracker requires record_x=True).")
    else:
        try:
            fig = viz.plot_population_embedding(pop_tracker, backend="plotly", show=False)
            st.plotly_chart(_retheme(fig), width='stretch')
            st.caption(
                "2D PCA projection of the entire continuous-variable population over time. "
                "Each faint grey line is one replica's trajectory; markers are coloured by epoch."
            )
        except Exception as e:
            st.info(f"PCA could not be computed: {e}")

with tab_ridge:
    if pop_tracker is None or not pop_tracker.loss:
        st.info("No population snapshots recorded for this run.")
    else:
        import plotly.graph_objects as go

        p = palette()
        snaps = pop_tracker.loss
        epochs = pop_tracker.epochs
        # Show at most ~18 epochs stacked top-to-bottom.
        stride = max(1, len(snaps) // 18)
        idx = list(range(0, len(snaps), stride))
        fig = go.Figure()
        for k, i in enumerate(idx):
            s = np.asarray(snaps[i])
            # Simple density: histogram-based KDE proxy using np.histogram.
            hist, edges = np.histogram(s, bins=32)
            centres = (edges[:-1] + edges[1:]) / 2
            dens = hist / max(1, hist.max())
            base = -k * 1.15
            fig.add_trace(
                go.Scatter(
                    x=centres, y=base + dens, mode="lines",
                    line={"color": p["palette"][k % len(p["palette"])], "width": 1.2},
                    fill="tonexty" if k > 0 else None,
                    name=f"epoch {epochs[i]}",
                    showlegend=False,
                )
            )
            fig.add_annotation(
                x=centres.min(), y=base + 0.05,
                text=f"ep {epochs[i]}", showarrow=False,
                font={"size": 10, "color": p["muted"]},
                xanchor="left",
            )
        fig.update_layout(
            **plotly_layout(
                title={"text": "Loss distribution — ridgeline over time"},
                xaxis_title="Loss (per replica)",
                yaxis_title="",
                height=max(420, 28 * len(idx)),
                showlegend=False,
                yaxis={"showticklabels": False, "showgrid": False, "zeroline": False,
                       "gridcolor": palette()["grid"], "linecolor": palette()["border"]},
            )
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("Stacked loss distributions. Early rows (top) → late rows (bottom).")

with tab_fate:
    if pop_tracker is None or not pop_tracker.loss:
        st.info("No population snapshots recorded for this run.")
    else:
        import plotly.graph_objects as go

        p = palette()
        mat = np.stack(pop_tracker.loss, axis=1)  # (sol_size, T)
        final = mat[:, -1]
        order = np.argsort(final)
        ranks = np.argsort(order)  # replica_idx -> rank
        fig = go.Figure()
        B, T = mat.shape
        for i in range(B):
            colour_idx = ranks[i] / max(1, B - 1)
            # Colour gradient from accent → accent2 by final rank.
            c = p["palette"][0] if colour_idx < 0.5 else p["palette"][1]
            fig.add_trace(
                go.Scatter(
                    x=pop_tracker.epochs,
                    y=mat[i],
                    mode="lines",
                    line={"color": c, "width": 0.6},
                    opacity=0.25 + 0.5 * (1 - colour_idx),
                    showlegend=False,
                    hovertemplate=f"replica {i} · final rank {int(ranks[i])}<br>loss=%{{y:.4f}}<extra></extra>",
                )
            )
        fig.update_layout(
            **plotly_layout(
                title={"text": "Replica fate — each line is one of the sol_size replicas"},
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=460,
            )
        )
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Top-ranked replicas are drawn in the primary accent, lower-ranked "
            "ones fade toward the secondary accent."
        )
