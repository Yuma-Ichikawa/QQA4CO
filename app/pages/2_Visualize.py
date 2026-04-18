"""Visualize page — inspect the last annealing run."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from _common import apply_theme  # noqa: E402

from qqa import visualization as viz  # noqa: E402

st.set_page_config(page_title="Visualize — QQA", page_icon="⚛️", layout="wide")
apply_theme()
st.title("Visualize")

result = st.session_state.get("last_result")
problem = st.session_state.get("last_problem")
if result is None:
    st.warning("Run QQA on the Solve page first.")
    st.stop()

pop_tracker = st.session_state.get("last_pop_tracker")

tab_hist, tab_best, tab_sched, tab_sol, tab_pop, tab_pca = st.tabs(
    [
        "Dynamics",
        "Best trajectory",
        "Schedule",
        "Solution",
        "Parallel population",
        "PCA trajectory",
    ]
)

with tab_hist:
    fig = viz.plot_history(result, backend="plotly", show=False)
    st.plotly_chart(fig, use_container_width=True)

with tab_best:
    fig = viz.plot_best_trajectory(result, backend="plotly", show=False)
    st.plotly_chart(fig, use_container_width=True)

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
                line={"color": "#a78bfa", "width": 2},
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title={"text": "Applied bg schedule", "x": 0.5},
            xaxis_title="Epoch",
            yaxis_title="bg",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_sol:
    try:
        fig = viz.plot_solution_heatmap(result, problem=problem, backend="plotly", show=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"No solution heatmap available: {e}")

with tab_pop:
    if pop_tracker is None or not pop_tracker.loss:
        st.info("No population snapshots recorded for this run.")
    else:
        fig = viz.plot_population_evolution(pop_tracker, backend="plotly", show=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each row is one of the `sol_size` replicas (sorted by final loss). "
            "Colour encodes per-replica loss. The white curve is the best-of-batch."
        )

with tab_pca:
    if pop_tracker is None or not pop_tracker.x:
        st.info("No x-snapshots recorded (population_tracker requires record_x=True).")
    else:
        try:
            fig = viz.plot_population_embedding(pop_tracker, backend="plotly", show=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "2D PCA projection of the entire continuous-variable population over time. "
                "Each faint grey line is one replica's trajectory; markers are coloured by epoch."
            )
        except Exception as e:
            st.info(f"PCA could not be computed: {e}")
