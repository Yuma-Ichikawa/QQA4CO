"""Solve page — run QQA with live progress, metrics, and a parallel-search view."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from _common import (  # noqa: E402
    apply_theme,
    build_problem,
    get_theme,
    hex_to_rgba,
    palette,
    paper_link_footer,
    plotly_layout,
    render_score_card,
    sidebar_brand,
    theme_toggle_in_sidebar,
)
from _solution_viz import render_solution_view  # noqa: E402

import qqa  # noqa: E402
from qqa.callbacks import Callback, CallbackState, PopulationTracker  # noqa: E402

st.set_page_config(page_title="Solve — QQA", page_icon="⚛️", layout="wide")
sidebar_brand()
theme_toggle_in_sidebar()
apply_theme()
st.title("Solve")

if "problem_config" not in st.session_state:
    st.warning("Define a problem on the Home page first.")
    st.stop()
cfg = st.session_state["problem_config"]
st.caption(
    f"problem: **{cfg['kind']}** | size: **{cfg['size']}** | "
    f"device: **{cfg['device']}** | seed: **{cfg['seed']}**"
)

# Hyper-parameter presets — each tuple is
# (sol_size, epochs, learning_rate, temp, min_bg, max_bg, curve_rate,
#  div_param, update_every).
_PRESETS = {
    "🏃  Fast smoke": (32, 200, 1.0, 0.0, -2.0, 0.1, 2, 0.0, 10),
    "🎯  Default": (64, 1000, 1.0, 0.0, -2.0, 0.1, 2, 0.0, 20),
    "🔬  Thorough": (128, 3000, 0.7, 0.05, -3.0, 0.2, 4, 0.0, 50),
}


def _apply_preset(name: str) -> None:
    """Write preset values into ``st.session_state`` so the widgets pick
    them up on the next rerun."""
    keys = (
        "sol_size",
        "epochs",
        "learning_rate",
        "temp",
        "min_bg",
        "max_bg",
        "curve_rate",
        "div_param",
        "update_every",
    )
    for k, v in zip(keys, _PRESETS[name], strict=True):
        st.session_state[k] = v


with st.sidebar:
    st.header("2 · QQA hyper-parameters")
    preset_name = st.radio(
        "Preset",
        list(_PRESETS),
        index=1,
        horizontal=False,
        help="Quickly seed every slider below. You can still tweak any value.",
    )
    if st.button("Apply preset", width="stretch"):
        _apply_preset(preset_name)
        st.rerun()

    with st.expander("Population & schedule", expanded=True):
        sol_size = st.slider(
            "sol_size",
            4,
            400,
            st.session_state.get("sol_size", 64),
            key="sol_size",
            help="Number of parallel replicas annealed in lockstep.",
        )
        epochs = st.slider(
            "epochs",
            100,
            5000,
            st.session_state.get("epochs", 1000),
            step=100,
            key="epochs",
            help="Total annealing iterations.",
        )
        curve_rate = st.selectbox(
            "curve rate",
            (2, 4, 6),
            index=(2, 4, 6).index(st.session_state.get("curve_rate", 2)),
            key="curve_rate",
            help="Steepness of the bias schedule (higher = more abrupt).",
        )

    with st.expander("Optimiser", expanded=False):
        learning_rate = st.slider(
            "learning rate",
            0.05,
            3.0,
            st.session_state.get("learning_rate", 1.0),
            0.05,
            key="learning_rate",
            help="Adam step size for the relaxed variables.",
        )
        temp = st.slider(
            "Langevin temperature",
            0.0,
            1.0,
            st.session_state.get("temp", 0.0),
            0.01,
            key="temp",
            help="Magnitude of the stochastic noise injected each step (0 = deterministic).",
        )

    with st.expander("Cooling / diversity", expanded=False):
        min_bg = st.slider(
            "min bg",
            -5.0,
            0.0,
            st.session_state.get("min_bg", -2.0),
            0.1,
            key="min_bg",
            help="Initial bias-gain (smooth, exploratory).",
        )
        max_bg = st.slider(
            "max bg",
            0.0,
            2.0,
            st.session_state.get("max_bg", 0.1),
            0.1,
            key="max_bg",
            help="Final bias-gain (sharp, near-discrete).",
        )
        div_param = st.slider(
            "div_param",
            0.0,
            1.0,
            st.session_state.get("div_param", 0.0),
            0.01,
            key="div_param",
            help="Repulsion strength between replicas (0 = independent runs).",
        )

    with st.expander("Display", expanded=False):
        update_every = st.slider(
            "UI update every (epochs)",
            1,
            200,
            st.session_state.get("update_every", 20),
            key="update_every",
            help="Lower = smoother animation but slower wall-clock; higher = faster.",
        )


class StreamlitCallback(Callback):
    """Stream QQA progress + a live parallel-population panel to Streamlit."""

    def __init__(
        self,
        progress_bar,
        metrics_holder,
        chart_holder,
        pop_holder,
        diversity_holder,
        sol_size: int,
        update_every: int = 20,
    ):
        self.progress_bar = progress_bar
        self.metrics_holder = metrics_holder
        self.chart_holder = chart_holder
        self.pop_holder = pop_holder
        self.diversity_holder = diversity_holder
        self.sol_size = sol_size
        self.update_every = max(1, int(update_every))
        self.epochs: list[int] = []
        self.mean_loss: list[float] = []
        self.best: list[float] = []
        self.pop: list[np.ndarray] = []
        self.std_loss: list[float] = []
        self._start = time.time()

    def on_epoch_end(self, state: CallbackState) -> None:
        epoch = state.epoch
        total = state.num_epochs
        self.epochs.append(epoch)
        losses = state.losses.detach().cpu().numpy()
        self.mean_loss.append(float(losses.mean()))
        self.std_loss.append(float(losses.std()))
        bo = state.best_obj
        self.best.append(float(np.asarray(bo).mean()) if hasattr(bo, "mean") else float(bo))
        self.pop.append(losses)

        if epoch % self.update_every != 0 and epoch != total - 1:
            return
        self.progress_bar.progress(min(1.0, (epoch + 1) / total))
        elapsed = time.time() - self._start
        with self.metrics_holder.container():
            r1c1, r1c2, r1c3 = st.columns(3)
            r1c1.metric("epoch", f"{epoch + 1} / {total}")
            r1c2.metric("best", f"{self.best[-1]:.4g}")
            r1c3.metric("mean", f"{self.mean_loss[-1]:.4g}")
            r2c1, r2c2, r2c3 = st.columns(3)
            r2c1.metric("σ (replicas)", f"{self.std_loss[-1]:.4g}")
            r2c2.metric("bg", f"{state.bg:.3f}")
            r2c3.metric("elapsed", f"{elapsed:.1f}s")

        p = palette()
        theme = get_theme()

        # --- Dynamics plot: best vs mean, with ±σ band ------------------
        std_arr = np.asarray(self.std_loss)
        mean_arr = np.asarray(self.mean_loss)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.epochs + self.epochs[::-1],
                y=np.concatenate([mean_arr + std_arr, (mean_arr - std_arr)[::-1]]).tolist(),
                fill="toself",
                fillcolor=hex_to_rgba(p["palette"][0], 0.13 if theme == "light" else 0.20),
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=False,
                name="±σ",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.epochs,
                y=self.mean_loss,
                mode="lines",
                name="mean / replica",
                line={"color": p["palette"][0], "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.epochs,
                y=self.best,
                mode="lines",
                name="best / replica",
                line={"color": p["palette"][1], "width": 3, "dash": "solid"},
            )
        )
        fig.update_layout(
            **plotly_layout(
                height=320,
                title={"text": "Annealing dynamics (per-replica loss)"},
                xaxis_title="Epoch",
                yaxis_title="Loss (per replica)",
                legend={"x": 0.01, "y": 0.02, "bgcolor": "rgba(255,255,255,0.6)"},
            )
        )
        # IMPORTANT: do NOT pass ``key=`` here. ``st.empty().plotly_chart`` is
        # invoked many times within one script run (once per ``update_every``
        # epochs); a stable key would collide with itself and trip
        # StreamlitDuplicateElementKey. ``theme=None`` skips Streamlit's
        # per-call theme-injection step (the figure already carries our
        # palette via ``plotly_layout``) which makes the redraw cheaper and
        # noticeably reduces flash on the live charts.
        self.chart_holder.plotly_chart(
            fig, width="stretch", theme=None, config={"displayModeBar": False}
        )

        # --- Population heatmap: replicas sorted by best-so-far --------
        pop = np.stack(self.pop, axis=1)  # (sol_size, T)
        # Use the replica's minimum over the whole history as the sort key,
        # so replicas that converged are packed near the top.
        order = np.argsort(pop.min(axis=1))
        pop_sorted = pop[order]
        best_traj = pop.min(axis=0)
        colorscale = "Viridis" if theme == "dark" else "Cividis"
        pop_fig = go.Figure()
        pop_fig.add_trace(
            go.Heatmap(
                z=pop_sorted,
                x=self.epochs,
                colorscale=colorscale,
                colorbar={"title": "loss", "thickness": 12, "len": 0.8},
                zsmooth="best",
                hovertemplate="epoch=%{x}<br>rank=%{y}<br>loss=%{z:.4f}<extra></extra>",
            )
        )
        pop_fig.add_trace(
            go.Scatter(
                x=self.epochs,
                y=best_traj,
                mode="lines",
                name="best-of-batch",
                yaxis="y2",
                line={"color": p["palette"][1], "width": 2.2},
            )
        )
        pop_fig.update_layout(
            **plotly_layout(
                height=340,
                title={"text": "Parallel population — loss over time"},
                xaxis_title="Epoch",
                yaxis_title="Replica (sorted by best-so-far)",
                yaxis2={
                    "overlaying": "y",
                    "side": "right",
                    "showgrid": False,
                    "title": "best loss",
                },
                legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.6)"},
            )
        )
        self.pop_holder.plotly_chart(
            pop_fig, width="stretch", theme=None, config={"displayModeBar": False}
        )

        # --- Diversity curve: std across replicas vs epoch --------------
        div_fig = go.Figure()
        div_fig.add_trace(
            go.Scatter(
                x=self.epochs,
                y=self.std_loss,
                mode="lines",
                fill="tozeroy",
                line={"color": p["palette"][2], "width": 2},
                fillcolor=hex_to_rgba(p["palette"][2], 0.20 if theme == "light" else 0.33),
                name="loss σ",
            )
        )
        div_fig.update_layout(
            **plotly_layout(
                height=200,
                title={"text": "Population diversity (std of per-replica loss)"},
                xaxis_title="Epoch",
                yaxis_title="σ",
                showlegend=False,
            )
        )
        self.diversity_holder.plotly_chart(
            div_fig, width="stretch", theme=None, config={"displayModeBar": False}
        )


run = st.button("▶  Run QQA", type="primary")
if run:
    try:
        problem = build_problem(cfg)
    except Exception as e:
        st.error(f"Could not build problem: {e}")
        st.stop()

    progress = st.progress(0.0)
    metrics = st.empty()
    score_holder = st.empty()
    c_left, c_right = st.columns([1, 1])
    chart = c_left.empty()
    pop_holder = c_right.empty()
    diversity_holder = st.empty()

    cb = StreamlitCallback(
        progress,
        metrics,
        chart,
        pop_holder,
        diversity_holder,
        sol_size=sol_size,
        update_every=update_every,
    )
    pop_tracker = PopulationTracker(stride=max(1, update_every), record_x=True)

    try:
        result = qqa.anneal(
            problem,
            sol_size=sol_size,
            learning_rate=learning_rate,
            temp=temp,
            min_bg=min_bg,
            max_bg=max_bg,
            curve_rate=curve_rate,
            div_param=div_param,
            num_epochs=epochs,
            device=cfg["device"],
            callbacks=[cb, pop_tracker],
            verbose=False,
        )
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

    # Clear the "still running" progress bar and display the polished result.
    progress.empty()
    raw = (
        result.best_obj
        if isinstance(result.best_obj, float)
        else float(__import__("numpy").asarray(result.best_obj).mean())
    )
    with score_holder.container():
        render_score_card(result.score, raw_loss=raw)
    st.session_state.setdefault("results", []).append(
        {
            "cfg": dict(cfg),
            "hp": {
                "sol_size": sol_size,
                "learning_rate": learning_rate,
                "temp": temp,
                "min_bg": min_bg,
                "max_bg": max_bg,
                "curve_rate": curve_rate,
                "div_param": div_param,
                "num_epochs": epochs,
            },
            "result": result,
        }
    )
    st.session_state["last_result"] = result
    st.session_state["last_problem"] = problem
    st.session_state["last_pop_tracker"] = pop_tracker

    # Professional, problem-specific solution visualisation.
    st.markdown("### Solution")
    st.caption(
        "Problem-aware view of the best configuration QQA found. "
        "Structure, constraint satisfaction, and summary metrics are shown."
    )
    render_solution_view(problem, result, cfg)

    st.info("Open **Visualize** for deeper inspection of this run (history, PCA, ridgeline).")


paper_link_footer()
