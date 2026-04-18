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
    palette,
    plotly_layout,
    render_score_card,
    theme_toggle_in_sidebar,
)

import qqa  # noqa: E402
from qqa.callbacks import Callback, CallbackState, PopulationTracker  # noqa: E402

st.set_page_config(page_title="Solve — QQA", page_icon="⚛️", layout="wide")
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

with st.sidebar:
    st.header("2 · QQA hyper-parameters")
    sol_size = st.slider("sol_size (parallel population)", 4, 400, 64)
    epochs = st.slider("epochs", 100, 5000, 1000, step=100)
    learning_rate = st.slider("learning rate", 0.05, 3.0, 1.0, 0.05)
    temp = st.slider("Langevin temperature", 0.0, 1.0, 0.0, 0.01)
    min_bg = st.slider("min bg", -5.0, 0.0, -2.0, 0.1)
    max_bg = st.slider("max bg", 0.0, 2.0, 0.1, 0.1)
    curve_rate = st.selectbox("curve rate", (2, 4, 6), index=0)
    div_param = st.slider("div_param", 0.0, 1.0, 0.0, 0.01)
    update_every = st.slider("UI update every (epochs)", 1, 200, 20)


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
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("epoch", f"{epoch + 1} / {total}")
            c2.metric("best / replica", f"{self.best[-1]:.4f}")
            c3.metric("mean / replica", f"{self.mean_loss[-1]:.4f}")
            c4.metric("σ across replicas", f"{self.std_loss[-1]:.4f}")
            c5.metric("bg", f"{state.bg:.3f}")
            c6.metric("elapsed", f"{elapsed:.1f}s")

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
                fillcolor=p["palette"][0] + ("22" if theme == "light" else "33"),
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
        self.chart_holder.plotly_chart(fig, width='stretch')

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
                    "overlaying": "y", "side": "right", "showgrid": False,
                    "title": "best loss",
                },
                legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.6)"},
            )
        )
        self.pop_holder.plotly_chart(pop_fig, width='stretch')

        # --- Diversity curve: std across replicas vs epoch --------------
        div_fig = go.Figure()
        div_fig.add_trace(
            go.Scatter(
                x=self.epochs,
                y=self.std_loss,
                mode="lines",
                fill="tozeroy",
                line={"color": p["palette"][2], "width": 2},
                fillcolor=p["palette"][2] + ("33" if theme == "light" else "55"),
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
        self.diversity_holder.plotly_chart(div_fig, width='stretch')


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
        progress, metrics, chart, pop_holder, diversity_holder,
        sol_size=sol_size, update_every=update_every,
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
    raw = result.best_obj if isinstance(result.best_obj, float) else float(
        __import__("numpy").asarray(result.best_obj).mean()
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
    st.info("Open **Visualize** for deeper inspection of this run.")
