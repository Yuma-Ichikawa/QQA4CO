"""Visualize page — inspect the last annealing run."""

from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from _common import (  # noqa: E402
    apply_theme,
    empty_state_card,
    hex_to_rgba,
    palette,
    paper_link_footer,
    plotly_layout,
    render_score_card,
    sidebar_brand,
    theme_toggle_in_sidebar,
)
from _solution_viz import render_solution_view  # noqa: E402

from qqa import visualization as viz  # noqa: E402

# Capability detection — see comment in 1_Solve.py. PAResult only exists
# from qqa 0.5.1 onwards; older deployed wheels would crash this page on
# import. We fall back to a sentinel class so ``isinstance`` is False
# everywhere and the PA-specific tabs simply don't render.
try:
    from qqa import PAResult  # noqa: E402
except ImportError:

    class PAResult:  # type: ignore[no-redef]
        """Stand-in used when the deployed qqa is too old to expose PAResult."""


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
is_pa = isinstance(result, PAResult)

# Score card at the top of the page so the headline metric is always visible.
raw = result.best_obj if isinstance(result.best_obj, float) else None
render_score_card(result.score, raw_loss=raw)
if is_pa:
    cF1, cF2, cF3 = st.columns(3)
    cF1.metric(
        "PA  F(β_end) / N",
        f"{result.free_energy_density:.4f}" if result.free_energy_density is not None else "—",
        help="Free energy density estimate from PA's Hukushima–Iba estimator.",
    )
    cF2.metric(
        "PA  ln Z(β_end)",
        f"{result.log_z:.3f}" if result.log_z is not None else "—",
        help="Absolute partition function, anchored at ln Z(0) = N · ln 2.",
    )
    cF3.metric(
        "PA  population R",
        f"{result.final_x.shape[0]}" if result.final_x is not None else "—",
    )

# "Family tree" is now backend-agnostic and always present:
#   * PA → Muller plot of resampling-induced founder takeover
#   * PQQA → hierarchical-clustering dendrogram + per-clade energy evolution
# That answers the user request to have a tree-style view that *adapts*
# to the selected backend rather than hiding behind a "PA-only" label.
base_tabs = [
    "Solution",
    "Dynamics",
    "Best trajectory",
    "Schedule",
    "Parallel population",
    "Solution-space PCA",
    "Diversity",
    "Loss spectrogram",
    "Ridgeline",
    "Replica fate",
    "Family tree",
]
pa_extra_tabs = ["PA: ESS / β", "PA: Free energy", "PA: Equilibrium pop."]
all_tabs = base_tabs + (pa_extra_tabs if is_pa else [])
_tabs = st.tabs(all_tabs)
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
    tab_tree,
) = _tabs[:11]
if is_pa:
    tab_pa_ess, tab_pa_fe, tab_pa_eq = _tabs[11:14]


def _retheme(fig):
    """Reskin a Plotly figure produced by ``viz`` to the current theme."""
    with contextlib.suppress(Exception):
        fig.update_layout(**plotly_layout())
    return fig


with tab_hist:
    try:
        fig = viz.plot_history(result, backend="plotly", show=False)
        st.plotly_chart(_retheme(fig), width="stretch")
    except Exception as e:
        if is_pa:
            # PA's history schema is intentionally smaller than QQA's
            # (no loss_std / penalty_mean). Show the PA-native equivalent.
            import plotly.graph_objects as go  # noqa: PLC0415

            xs = result.history.get("beta", list(range(len(result.history["loss_mean"]))))
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=result.history["loss_mean"],
                    mode="lines+markers",
                    name="loss_mean",
                    line={"color": palette()["palette"][0], "width": 2},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=result.history.get("loss_min", []),
                    mode="lines+markers",
                    name="loss_min",
                    line={"color": palette()["palette"][1], "width": 2, "dash": "dot"},
                )
            )
            fig.update_layout(
                **plotly_layout(
                    title={"text": "PA dynamics — mean / min loss vs β"},
                    xaxis_title="β",
                    yaxis_title="Loss",
                    height=400,
                )
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info(f"History plot unavailable: {e}")

with tab_best:
    try:
        fig = viz.plot_best_trajectory(result, backend="plotly", show=False)
        st.plotly_chart(_retheme(fig), width="stretch")
    except Exception as e:
        st.info(f"Best-trajectory plot unavailable: {e}")

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
        st.info(
            "No x-snapshots recorded — re-run with the default "
            "`PopulationTracker(record_x=True)` to enable this view."
        )
    else:
        try:
            import plotly.graph_objects as go

            # Final-population-only PCA. We deliberately ignore the epoch
            # dimension here — temporal dynamics live in the spectrogram /
            # diversity / replica-fate tabs. This view answers a different
            # question: "where in solution space did my parallel replicas
            # end up?".
            x_final = np.asarray(pop_tracker.x[-1])  # (B, ...)
            B = x_final.shape[0]
            X = x_final.reshape(B, -1).astype(np.float32, copy=False)  # (B, D)
            D = X.shape[1]
            mean = X.mean(axis=0, keepdims=True)
            X0 = X - mean
            ncomp = min(3, D, B)
            try:
                _, sigmas, vt = np.linalg.svd(X0, full_matrices=False)
            except np.linalg.LinAlgError:
                sigmas = np.zeros(ncomp, dtype=np.float32)
                vt = np.eye(D, dtype=np.float32)[:ncomp]
            comps = vt[:ncomp].T  # (D, ncomp)
            proj = X0 @ comps  # (B, ncomp)
            if ncomp < 3:
                proj = np.concatenate([proj, np.zeros((B, 3 - ncomp))], axis=-1)

            # Explained-variance ratio per axis (handles the degenerate
            # σ = 0 case as 0%).
            var_total = float((sigmas**2).sum())
            evr = (sigmas[:3] ** 2 / var_total) if var_total > 0 else np.zeros(3)
            evr = np.pad(evr, (0, max(0, 3 - len(evr))))[:3]

            final_loss = np.asarray(pop_tracker.loss[-1], dtype=np.float64)
            best_idx = int(np.argmin(final_loss))

            p = palette()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=proj[:, 0],
                    y=proj[:, 1],
                    z=proj[:, 2],
                    mode="markers",
                    marker={
                        "size": 5.5,
                        "color": final_loss,
                        "colorscale": "Plasma",
                        "cmin": float(final_loss.min()),
                        "cmax": float(final_loss.max()),
                        "showscale": True,
                        "colorbar": {"title": "loss"},
                        "line": {"color": "white", "width": 0.6},
                        "opacity": 0.95,
                    },
                    text=[
                        f"replica {i}<br>loss = {float(v):.4f}" for i, v in enumerate(final_loss)
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )
            # Highlight the global-best replica with a diamond.
            fig.add_trace(
                go.Scatter3d(
                    x=[float(proj[best_idx, 0])],
                    y=[float(proj[best_idx, 1])],
                    z=[float(proj[best_idx, 2])],
                    mode="markers",
                    marker={
                        "size": 11,
                        "symbol": "diamond",
                        "color": p["accent2"],
                        "line": {"color": "white", "width": 1.5},
                    },
                    name=f"best (replica {best_idx})",
                    hovertemplate=(
                        f"best replica {best_idx}<br>"
                        f"loss = {float(final_loss[best_idx]):.4f}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                **plotly_layout(
                    title={
                        "text": ("Final parallel population in solution space (PCA-3 projection)")
                    },
                    height=620,
                    showlegend=True,
                    legend={"x": 0.02, "y": 0.98},
                )
            )
            fig.update_scenes(
                xaxis_title=f"PC1 ({evr[0] * 100:.1f}%)",
                yaxis_title=f"PC2 ({evr[1] * 100:.1f}%)",
                zaxis_title=f"PC3 ({evr[2] * 100:.1f}%)",
                bgcolor=palette()["bg_card"],
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"PCA-3 of the **final** ``sol_size = {B}`` parallel solutions "
                f"({D}-dimensional, projected via truncated SVD). Marker colour "
                "encodes per-replica final loss; the diamond is the global best. "
                "Drag to rotate. The three axes capture "
                f"{evr[:3].sum() * 100:.1f}% of the population's spread."
            )
            with st.expander("Show 2D projection / epoch trajectory", expanded=False):
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


if is_pa:
    import plotly.graph_objects as go  # noqa: PLC0415

    pa = result
    p = palette()

    with tab_pa_ess:
        if pa.history.get("ess"):
            betas_h = pa.history.get("beta", list(range(len(pa.history["ess"]))))
            ess_h = pa.history["ess"]
            R = pa.final_x.shape[0] if pa.final_x is not None else max(ess_h)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=betas_h,
                    y=ess_h,
                    mode="lines+markers",
                    line={"color": p["palette"][0], "width": 2.4},
                    name="ESS",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[min(betas_h), max(betas_h)],
                    y=[R, R],
                    mode="lines",
                    line={"color": p["muted"], "width": 1, "dash": "dash"},
                    name=f"R = {R}",
                )
            )
            fig.update_layout(
                **plotly_layout(
                    title={"text": "Effective Sample Size vs β"},
                    xaxis_title="β",
                    yaxis_title="ESS",
                    height=380,
                )
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Kish's ESS = (Σw)² / Σw². Drops mean PA had to concentrate "
                "the population on a low-energy subset; persistent collapse "
                "is a sign you need more replicas or a denser β grid."
            )
        else:
            st.info("No ESS history recorded.")

    with tab_pa_fe:
        if pa.history.get("free_energy_density") and pa.history.get("beta"):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=pa.history["beta"],
                    y=pa.history["free_energy_density"],
                    mode="lines+markers",
                    line={"color": p["palette"][3 % len(p["palette"])], "width": 2.4},
                    name="F(β)/N",
                )
            )
            fig.update_layout(
                **plotly_layout(
                    title={"text": "Free-energy density estimate (Hukushima–Iba)"},
                    xaxis_title="β",
                    yaxis_title="F(β) / N",
                    height=380,
                )
            )
            st.plotly_chart(fig, width="stretch")
            # Per-step ln Z increments
            if pa.history.get("log_z"):
                fig2 = go.Figure()
                fig2.add_trace(
                    go.Scatter(
                        x=pa.history["beta"],
                        y=pa.history["log_z"],
                        mode="lines+markers",
                        line={"color": p["palette"][1], "width": 2},
                        name="ln Z(β)",
                    )
                )
                fig2.update_layout(
                    **plotly_layout(
                        title={"text": "Cumulative ln Z(β)"},
                        xaxis_title="β",
                        yaxis_title="ln Z",
                        height=320,
                    )
                )
                st.plotly_chart(fig2, width="stretch")
            st.caption(
                "Free energy is computed from the PA reweighting trail with "
                "the resampling correction implicit (the unnormalised weights "
                "are averaged over the current population at every step)."
            )
        else:
            st.info("No free-energy history available.")

    with tab_pa_eq:
        if pa.final_x is not None and pa.final_loss is not None:
            losses = pa.final_loss.detach().cpu().numpy()
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=losses,
                    nbinsx=min(48, max(8, len(losses) // 8)),
                    marker={"color": p["palette"][2 % len(p["palette"])]},
                    name="P(E)",
                )
            )
            fig.update_layout(
                **plotly_layout(
                    title={"text": "Equilibrium population — energy histogram at β_end"},
                    xaxis_title="Energy",
                    yaxis_title="Count",
                    height=380,
                )
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown(
                f"- mean E = **{float(losses.mean()):.4f}**, "
                f"min E = **{float(losses.min()):.4f}**, "
                f"std E = **{float(losses.std()):.4f}** "
                f"(R = {len(losses)})"
            )
            st.caption(
                "After ``num_temps × sweeps_per_temp`` MCMC steps and "
                "informative resampling, ``final_x`` is approximately a "
                "Boltzmann sample at ``β_end``. Use it to estimate "
                "observables / order parameters."
            )
        else:
            st.info("No equilibrium population stored.")

with tab_tree:
    # Backend-aware family tree.
    #   * PA → resampling-induced founder takeover (Muller plot etc).
    #   * PQQA → hierarchical clustering of the final population (true
    #     dendrogram) + per-clade energy trajectory showing mode formation.
    import plotly.graph_objects as go  # noqa: PLC0415

    p = palette()
    _pa = result if is_pa else None
    if is_pa and _pa is not None and _pa.genealogy is not None:
        import math as _math  # noqa: PLC0415

        pa = _pa

        st.markdown(
            "##### PA family tree — resampling-driven genealogy\n"
            "Each replica's lineage is reconstructed by composing parent "
            "indices step-by-step. Bands appearing / expanding / going "
            "extinct are *clonal sweeps* in real time."
        )

        parents = pa.genealogy["parents"]
        ancestors = pa.genealogy["ancestors"]
        betas_g = pa.genealogy.get("betas", list(range(len(parents) + 1)))

        R = ancestors.shape[0]
        T = len(parents)

        # Compose parent maps to recover ``mat[t, r]`` = founder of slot
        # ``r`` at step ``t``. ``parents[t][i]`` is the slot copied
        # forward into ``i`` at time ``t+1``; chained composition gives
        # each survivor's root founder.
        current = np.arange(R)
        anc_through_time = [current]
        for t in range(T):
            current = current[parents[t].cpu().numpy()]
            anc_through_time.append(current)
        mat = np.stack(anc_through_time, axis=0)  # (T+1, R)
        n_surv = np.array([len(np.unique(mat[t])) for t in range(T + 1)])

        shares = np.zeros((T + 1, R), dtype=float)
        for t in range(T + 1):
            shares[t] = np.bincount(mat[t], minlength=R) / R

        betas_arr = np.asarray(betas_g, dtype=float)
        x_axis = (
            np.concatenate([[0.5 * float(betas_arr.min())], betas_arr])
            if betas_arr.size == T
            else np.arange(T + 1)
        )

        # ----- Muller plot (stacked area founder shares) -----------------
        # Sort founders so dominant ones cluster near the bottom of the
        # stack — strongest "selective sweeps" rise like a wave. Sort by
        # total area under the curve.
        order = np.argsort(shares.sum(axis=0))[::-1]  # large → small
        shares_sorted = shares[:, order]
        colorscale = "Turbo"
        cum = np.zeros_like(x_axis, dtype=float)
        fig_muller = go.Figure()
        for k in range(R):
            share = shares_sorted[:, k]
            if share.max() == 0.0:
                continue
            upper = cum + share
            colour_t = float(order[k]) / max(1.0, R - 1.0)
            rgb = [
                int(255 * v)
                for v in [
                    0.18 + 0.82 * abs(_math.sin(3.0 * colour_t + 0.3)),
                    0.30 + 0.55 * abs(_math.sin(2.0 * colour_t + 1.7)),
                    0.40 + 0.55 * abs(_math.sin(1.5 * colour_t + 3.3)),
                ]
            ]
            rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.92)"
            fig_muller.add_trace(
                go.Scatter(
                    x=np.concatenate([x_axis, x_axis[::-1]]),
                    y=np.concatenate([upper, cum[::-1]]),
                    fill="toself",
                    fillcolor=rgba,
                    line={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    showlegend=False,
                    mode="lines",
                )
            )
            cum = upper

        fig_muller.update_layout(
            **plotly_layout(
                title={
                    "text": (
                        f"PA Muller plot — founder shares vs β "
                        f"(R = {R}; final founders = {int(n_surv[-1])})"
                    )
                },
                xaxis_title="β (anneal progress, log)" if betas_arr.size == T else "step",
                yaxis_title="Population share",
                xaxis={"type": "log"} if betas_arr.size == T else None,
                yaxis={"range": [0, 1], "tickvals": [0, 0.25, 0.5, 0.75, 1.0]},
                height=420,
            )
        )
        st.plotly_chart(fig_muller, width="stretch")

        # ----- Sorted ancestry matrix (heatmap) --------------------------
        sorted_mat = np.sort(mat, axis=1)
        fig_mat = go.Figure(
            go.Heatmap(
                z=sorted_mat.T,
                x=np.arange(T + 1),
                colorscale=colorscale,
                zmin=0,
                zmax=R - 1,
                showscale=True,
                colorbar={"title": "founder id", "thickness": 12, "len": 0.85},
                hovertemplate=("step %{x}<br>sorted-replica %{y}<br>founder %{z}<extra></extra>"),
            )
        )
        fig_mat.update_layout(
            **plotly_layout(
                title={"text": "Sorted ancestry matrix — clades widen / pinch off"},
                xaxis_title="Temperature step",
                yaxis_title="Replica (sorted by founder per step)",
                height=380,
            )
        )
        st.plotly_chart(fig_mat, width="stretch")

        # ----- Survivor curve --------------------------------------------
        fig_n = go.Figure()
        fig_n.add_trace(
            go.Scatter(
                x=x_axis,
                y=n_surv.tolist(),
                mode="lines+markers",
                line={"color": p["palette"][1], "width": 2.4},
                name="distinct surviving founders",
            )
        )
        fig_n.add_trace(
            go.Scatter(
                x=[float(x_axis.min()), float(x_axis.max())],
                y=[R / _math.e, R / _math.e],
                mode="lines",
                line={"color": p["muted"], "width": 1, "dash": "dash"},
                name=f"R/e ≈ {R / _math.e:.0f}",
            )
        )
        fig_n.update_layout(
            **plotly_layout(
                title={
                    "text": "Population collapse — surviving founders vs β "
                    f"(R = {R}; final = {int(n_surv[-1])})"
                },
                xaxis_title="β" if betas_arr.size == T else "step",
                yaxis_title="distinct ancestors",
                xaxis={"type": "log"} if betas_arr.size == T else None,
                height=300,
            )
        )
        st.plotly_chart(fig_n, width="stretch")

        st.caption(
            "**Muller plot** *(top)* shows each founder's lineage as a "
            "coloured band — bands appearing/expanding/going extinct "
            "are clonal sweeps in real time. **Sorted ancestry matrix** "
            "*(middle)* puts the same data on a heatmap with replicas "
            "sorted by founder, so clades read off as horizontal stripes. "
            "**Survivor curve** *(bottom)* tracks distinct founders vs β "
            "with the R/e bottleneck guideline."
        )

    elif (not is_pa) and pop_tracker is not None and pop_tracker.x:
        # PQQA pseudo-genealogy: replicas never resample, but they collapse
        # into a small number of *modes* via the gradient. We expose this
        # collapse with two complementary views:
        #   1. Hierarchical-clustering dendrogram of the final population
        #      (literal tree — answers "which final basins are siblings").
        #   2. Per-clade mean-energy trajectory across epochs (shows when
        #      the clades diverged from a common ancestor distribution).
        st.markdown(
            "##### PQQA family tree — mode-collapse pseudo-genealogy\n"
            "PQQA has no resampling, so we cluster the *final* population "
            "by Hamming distance and trace each cluster's mean energy "
            "back through time. The dendrogram shows which basins are "
            "siblings; the trajectory plot shows when they diverged."
        )

        try:
            from scipy.cluster.hierarchy import dendrogram, fcluster, linkage  # noqa: PLC0415
            from scipy.spatial.distance import pdist  # noqa: PLC0415
        except Exception:  # pragma: no cover
            st.warning(
                "scipy is required for the PQQA dendrogram. Install with `pip install scipy`."
            )
        else:
            x_final = np.asarray(pop_tracker.x[-1])
            x_flat = x_final.reshape(x_final.shape[0], -1).astype(np.float32)
            B = x_flat.shape[0]
            # Hamming distance for binary {0,1}/{-1,+1} states; falls back
            # to Euclidean for floating-point relaxations.
            uniq = np.unique(x_flat)
            metric = "hamming" if uniq.size <= 4 and B >= 2 else "euclidean"
            try:
                dist_vec = pdist(x_flat, metric=metric)
                Z = linkage(dist_vec, method="average")
            except Exception as exc:
                st.info(f"Could not build dendrogram: {exc}")
                Z = None

            if Z is not None:
                # ---- Dendrogram (literal phylogenetic tree) ------------
                # ``no_plot=True`` runs the layout without matplotlib so we
                # can rebuild it in Plotly (matches the rest of the page).
                ddata = dendrogram(Z, no_plot=True, color_threshold=None)
                fig_dend = go.Figure()
                for xs, ys in zip(ddata["icoord"], ddata["dcoord"], strict=True):
                    fig_dend.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            line={"color": p["palette"][0], "width": 1.6},
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                fig_dend.update_layout(
                    **plotly_layout(
                        title={
                            "text": (
                                "PQQA dendrogram — average-linkage hierarchical "
                                f"clustering of the final population (B = {B}, "
                                f"metric = {metric})"
                            )
                        },
                        xaxis_title="Replica (re-ordered by clustering)",
                        yaxis_title=f"Linkage distance ({metric})",
                        xaxis={"showticklabels": False},
                        height=380,
                    )
                )
                st.plotly_chart(fig_dend, width="stretch")

                # ---- Per-clade energy trajectory -----------------------
                # Cut the tree at √B clusters (capped at 8) — small enough
                # to be readable, large enough to capture multi-modality.
                k_clusters = max(2, min(8, int(round(math.sqrt(B)))))
                labels = fcluster(Z, t=k_clusters, criterion="maxclust")
                # Map labels to dense 0..k-1 ordered by cluster size.
                _, counts = np.unique(labels, return_counts=True)
                order_by_size = np.argsort(-counts)  # largest first
                relabel = {old + 1: new for new, old in enumerate(order_by_size)}
                clade = np.array([relabel[int(lbl)] for lbl in labels])

                epochs_pqqa = np.asarray(pop_tracker.epochs)
                loss_mat = np.stack(pop_tracker.loss, axis=0)  # (T, B)
                fig_clade = go.Figure()
                for k in range(k_clusters):
                    mask = clade == k
                    if not mask.any():
                        continue
                    mean_loss = loss_mat[:, mask].mean(axis=1)
                    band_lo = loss_mat[:, mask].min(axis=1)
                    band_hi = loss_mat[:, mask].max(axis=1)
                    colour = p["palette"][k % len(p["palette"])]
                    fig_clade.add_trace(
                        go.Scatter(
                            x=np.concatenate([epochs_pqqa, epochs_pqqa[::-1]]),
                            y=np.concatenate([band_hi, band_lo[::-1]]),
                            fill="toself",
                            fillcolor=hex_to_rgba(colour, 0.12),
                            line={"color": "rgba(0,0,0,0)"},
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    fig_clade.add_trace(
                        go.Scatter(
                            x=epochs_pqqa,
                            y=mean_loss,
                            mode="lines",
                            name=f"clade {k} (n={int(mask.sum())})",
                            line={"color": colour, "width": 2},
                        )
                    )
                fig_clade.update_layout(
                    **plotly_layout(
                        title={
                            "text": (
                                f"Per-clade energy trajectory — {k_clusters} "
                                f"final modes traced back through training"
                            )
                        },
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        height=380,
                        legend={"orientation": "h", "y": -0.18},
                    )
                )
                st.plotly_chart(fig_clade, width="stretch")

                st.caption(
                    "**Dendrogram** *(top)* clusters the final population "
                    "by Hamming/Euclidean distance — short branches are "
                    "near-duplicate solutions, long branches are distinct "
                    "basins. **Per-clade energy trajectory** *(bottom)* "
                    "follows each final clade's mean loss back through "
                    "epochs, with the shaded band giving min–max spread. "
                    "Bifurcating bands mark when the population committed "
                    "to different basins."
                )

    else:
        if is_pa:
            st.info(
                "Family tree unavailable — re-run from the **Solve** page "
                "with the PA backend (genealogy is recorded by default in "
                "this UI). PA's API also accepts ``record_genealogy=True``."
            )
        else:
            st.info(
                "Family tree unavailable — run a problem from the "
                "**Solve** page first; this view needs per-epoch "
                "population snapshots (PQQA records them automatically)."
            )


paper_link_footer()
