"""Visualize page — inspect the last annealing run."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from _common import (  # noqa: E402
    apply_theme,
    empty_state_card,
    palette,
    paper_link_footer,
    plotly_layout,
    render_score_card,
    sidebar_brand,
    theme_toggle_in_sidebar,
)
from _solution_viz import render_solution_view  # noqa: E402

from qqa import visualization as viz  # noqa: E402

st.set_page_config(page_title="Visualize — QQA", page_icon="⚛️", layout="wide")
sidebar_brand()
theme_toggle_in_sidebar()
apply_theme()
st.title("Visualize")

result = st.session_state.get("last_result")
problem = st.session_state.get("last_problem")
if result is None:
    empty_state_card(
        title="No annealing run loaded yet",
        body=(
            "Define a problem on <b>Home</b>, then jump to <b>Solve</b> "
            "and click <i>Run QQA</i>. The result will appear here for "
            "deeper inspection — score, history, parallel-coordinates, "
            "and per-problem solution view."
        ),
        cta_label="Open Solve",
        cta_page="pages/1_Solve.py",
    )
    st.stop()

pop_tracker = st.session_state.get("last_pop_tracker")

# Score card at the top of the page so the headline metric is always visible.
raw = result.best_obj if isinstance(result.best_obj, float) else None
render_score_card(result.score, raw_loss=raw)

(
    tab_sol,
    tab_hist,
    tab_best,
    tab_sched,
    tab_pop,
    tab_pca,
    tab_div,
    tab_spec,
    tab_ridge,
    tab_fate,
) = st.tabs(
    [
        "Solution",
        "Dynamics",
        "Best trajectory",
        "Schedule",
        "Parallel population",
        "3D PCA flow",
        "Diversity",
        "Loss spectrogram",
        "Ridgeline",
        "Replica fate",
    ]
)


def _retheme(fig):
    """Reskin a Plotly figure produced by ``viz`` to the current theme."""
    with contextlib.suppress(Exception):
        fig.update_layout(**plotly_layout())
    return fig


with tab_hist:
    fig = viz.plot_history(result, backend="plotly", show=False)
    st.plotly_chart(_retheme(fig), width="stretch")

with tab_best:
    fig = viz.plot_best_trajectory(result, backend="plotly", show=False)
    st.plotly_chart(_retheme(fig), width="stretch")

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
        st.plotly_chart(fig, width="stretch")

with tab_sol:
    cfg = st.session_state.get("problem_config", {})
    st.markdown("#### Problem-aware view")
    st.caption(
        "The solution is rendered natively for the selected problem type "
        "(TSP tour, N-Queens board, highlighted IS, colouring, ...)."
    )
    render_solution_view(problem, result, cfg)
    with st.expander("Raw solution heatmap", expanded=False):
        try:
            fig = viz.plot_solution_heatmap(result, problem=problem, backend="plotly", show=False)
            st.plotly_chart(_retheme(fig), width="stretch")
        except Exception as e:
            st.info(f"No solution heatmap available: {e}")

with tab_pop:
    if pop_tracker is None or not pop_tracker.loss:
        st.info("No population snapshots recorded for this run.")
    else:
        fig = viz.plot_population_evolution(pop_tracker, backend="plotly", show=False)
        st.plotly_chart(_retheme(fig), width="stretch")
        st.caption(
            "Each row is one of the `sol_size` replicas (sorted by final loss). "
            "Colour encodes per-replica loss."
        )

with tab_pca:
    if pop_tracker is None or not pop_tracker.x:
        st.info("No x-snapshots recorded (population_tracker requires record_x=True).")
    else:
        try:
            import plotly.graph_objects as go

            xs = pop_tracker.x  # list[T] of (B, ...) ndarrays
            epochs = np.asarray(pop_tracker.epochs)
            X = np.stack([np.asarray(snap).reshape(snap.shape[0], -1) for snap in xs], axis=0)
            # X: (T, B, D). Stack into a flat (T*B, D) cloud and run a
            # truncated SVD so PCA-3D scales to N≈10⁴ variables comfortably.
            T_, B, D = X.shape
            flat = X.reshape(T_ * B, D)
            mean = flat.mean(axis=0, keepdims=True)
            flat0 = flat - mean
            ncomp = min(3, flat0.shape[1], flat0.shape[0])
            try:
                # Faster than np.linalg.svd for tall matrices.
                _, _, vt = np.linalg.svd(flat0, full_matrices=False)
            except np.linalg.LinAlgError:
                vt = np.eye(D)[:ncomp]
            comps = vt[:ncomp].T  # (D, ncomp)
            proj = (flat0 @ comps).reshape(T_, B, ncomp)  # (T, B, 3)
            if ncomp < 3:
                pad = np.zeros((T_, B, 3 - ncomp))
                proj = np.concatenate([proj, pad], axis=-1)

            final_loss = np.asarray(pop_tracker.loss[-1])
            cmin = float(final_loss.min())
            cmax = float(final_loss.max())
            p = palette()
            fig = go.Figure()
            # One faint trajectory per replica.
            stride = max(1, B // 64)  # cap visible lines for readability
            for i in range(0, B, stride):
                fig.add_trace(
                    go.Scatter3d(
                        x=proj[:, i, 0],
                        y=proj[:, i, 1],
                        z=proj[:, i, 2],
                        mode="lines",
                        line={"color": p["muted"], "width": 1.5},
                        opacity=0.18,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            # Snapshot scatter coloured by epoch — gives the "flow" feel.
            for k in range(T_):
                fig.add_trace(
                    go.Scatter3d(
                        x=proj[k, :, 0],
                        y=proj[k, :, 1],
                        z=proj[k, :, 2],
                        mode="markers",
                        marker={
                            "size": 2.5,
                            "color": [int(epochs[k])] * B,
                            "colorscale": "Viridis",
                            "cmin": int(epochs.min()),
                            "cmax": int(epochs.max()),
                            "showscale": k == T_ - 1,
                            "colorbar": {"title": "epoch"} if k == T_ - 1 else None,
                            "opacity": 0.55,
                        },
                        showlegend=False,
                        name=f"epoch {int(epochs[k])}",
                        hovertemplate=f"epoch {int(epochs[k])}<br>PC1=%{{x:.3f}}, PC2=%{{y:.3f}}, PC3=%{{z:.3f}}<extra></extra>",
                    )
                )
            # Overlay final population coloured by per-replica final loss.
            fig.add_trace(
                go.Scatter3d(
                    x=proj[-1, :, 0],
                    y=proj[-1, :, 1],
                    z=proj[-1, :, 2],
                    mode="markers",
                    marker={
                        "size": 4.2,
                        "color": final_loss,
                        "colorscale": "Plasma",
                        "cmin": cmin,
                        "cmax": cmax,
                        "showscale": True,
                        "colorbar": {"title": "final loss", "x": 1.12},
                        "line": {"color": "white", "width": 0.6},
                    },
                    name="final",
                    showlegend=False,
                    hovertemplate="final<br>loss=%{marker.color:.4f}<extra></extra>",
                )
            )
            fig.update_layout(
                **plotly_layout(
                    title={"text": "3D PCA flow of the parallel population"},
                    height=620,
                    showlegend=False,
                )
            )
            fig.update_scenes(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
                bgcolor=palette()["surface"],
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Truncated-SVD PCA (3 components) of the continuous-variable "
                "population. Each grey thread is one replica's trajectory; "
                "snapshot dots are coloured by epoch (Viridis); the final "
                "population is overlaid in Plasma by per-replica final loss. "
                "Drag to rotate."
            )
            with st.expander("Show 2D projection", expanded=False):
                fig2 = viz.plot_population_embedding(pop_tracker, backend="plotly", show=False)
                st.plotly_chart(_retheme(fig2), width="stretch")
        except Exception as e:
            st.info(f"PCA could not be computed: {e}")

with tab_div:
    if pop_tracker is None or not pop_tracker.loss:
        st.info("No population snapshots recorded for this run.")
    else:
        import plotly.graph_objects as go

        p = palette()
        epochs = np.asarray(pop_tracker.epochs)
        loss_mat = np.stack(pop_tracker.loss, axis=0)  # (T, B)
        # Loss spread = std over replicas — collapses to ~0 once the
        # population converges.
        loss_std = loss_mat.std(axis=1)
        loss_min = loss_mat.min(axis=1)
        loss_med = np.median(loss_mat, axis=1)

        # Genotypic diversity: mean pairwise variance of the projected
        # bits. Cheap proxy for Hamming distance and works for
        # categorical / spin variables alike. Falls back gracefully
        # when ``record_x`` was off.
        geno_div: np.ndarray | None = None
        if pop_tracker.x:
            try:
                xs = [np.asarray(s).reshape(s.shape[0], -1) for s in pop_tracker.x]
                geno_div = np.array([s.var(axis=0).mean() for s in xs])
            except Exception:
                geno_div = None

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=loss_std,
                mode="lines",
                line={"color": p["palette"][0], "width": 2.4},
                name="loss σ across replicas",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=loss_med - loss_min,
                mode="lines",
                line={"color": p["palette"][1], "width": 2.0, "dash": "dot"},
                name="median − min loss",
            )
        )
        if geno_div is not None and geno_div.size == epochs.size:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=geno_div,
                    mode="lines",
                    line={
                        "color": p["palette"][2 % len(p["palette"])],
                        "width": 2.0,
                    },
                    name="genotypic variance ⟨Var_b x⟩",
                    yaxis="y2",
                )
            )
        layout_kwargs = plotly_layout(
            title={"text": "Population diversity over time"},
            xaxis_title="Epoch",
            yaxis_title="Loss spread",
            height=420,
        )
        if geno_div is not None and geno_div.size == epochs.size:
            layout_kwargs["yaxis2"] = {
                "overlaying": "y",
                "side": "right",
                "title": "⟨Var_b x⟩",
                "showgrid": False,
            }
        fig.update_layout(**layout_kwargs)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Tracks how quickly the parallel population collapses. "
            "Healthy CRA / repulsion runs keep the genotypic variance "
            "above zero for longer."
        )

with tab_spec:
    if pop_tracker is None or not pop_tracker.loss:
        st.info("No population snapshots recorded for this run.")
    else:
        import plotly.graph_objects as go

        epochs = np.asarray(pop_tracker.epochs)
        loss_mat = np.stack(pop_tracker.loss, axis=0)  # (T, B)
        # Per-row histogram → (T, n_bins) heatmap.
        n_bins = 64
        lo = float(loss_mat.min())
        hi = float(loss_mat.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
        spec = np.zeros((len(epochs), n_bins), dtype=float)
        for t in range(len(epochs)):
            h, _ = np.histogram(loss_mat[t], bins=edges)
            spec[t] = h
        # Row-normalise to keep the colour scale stable when sol_size is large.
        row_max = spec.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        spec = spec / row_max
        centres = 0.5 * (edges[:-1] + edges[1:])
        fig = go.Figure(
            go.Heatmap(
                z=spec.T,
                x=epochs,
                y=centres,
                colorscale="Magma",
                colorbar={"title": "density"},
                hovertemplate="epoch %{x}<br>loss %{y:.4f}<br>p=%{z:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            **plotly_layout(
                title={"text": "Loss-distribution spectrogram"},
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=460,
            )
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Row-normalised histogram of replica losses at each recorded "
            "epoch. A sharp horizontal stripe at the bottom = converged."
        )

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
                    x=centres,
                    y=base + dens,
                    mode="lines",
                    line={"color": p["palette"][k % len(p["palette"])], "width": 1.2},
                    fill="tonexty" if k > 0 else None,
                    name=f"epoch {epochs[i]}",
                    showlegend=False,
                )
            )
            fig.add_annotation(
                x=centres.min(),
                y=base + 0.05,
                text=f"ep {epochs[i]}",
                showarrow=False,
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
                yaxis={
                    "showticklabels": False,
                    "showgrid": False,
                    "zeroline": False,
                    "gridcolor": palette()["grid"],
                    "linecolor": palette()["border"],
                },
            )
        )
        st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Top-ranked replicas are drawn in the primary accent, lower-ranked "
            "ones fade toward the secondary accent."
        )


paper_link_footer()
