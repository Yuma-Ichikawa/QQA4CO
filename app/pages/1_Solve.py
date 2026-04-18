"""Solve page — run QQA with live progress, metrics, and a parallel-search view."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from _common import apply_theme, build_problem  # noqa: E402

import qqa  # noqa: E402
from qqa.callbacks import Callback, CallbackState, PopulationTracker  # noqa: E402

st.set_page_config(page_title="Solve — QQA", page_icon="⚛️", layout="wide")
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
    """Stream QQA progress and a live population panel to Streamlit widgets."""

    def __init__(
        self,
        progress_bar,
        metrics_holder,
        chart_holder,
        pop_holder,
        update_every: int = 20,
    ):
        self.progress_bar = progress_bar
        self.metrics_holder = metrics_holder
        self.chart_holder = chart_holder
        self.pop_holder = pop_holder
        self.update_every = max(1, int(update_every))
        self.epochs: list[int] = []
        self.mean_loss: list[float] = []
        self.best: list[float] = []
        self.pop: list[np.ndarray] = []
        self._start = time.time()

    def on_epoch_end(self, state: CallbackState) -> None:
        epoch = state.epoch
        total = state.num_epochs
        self.epochs.append(epoch)
        losses = state.losses.detach().cpu().numpy()
        self.mean_loss.append(float(losses.mean()))
        bo = state.best_obj
        self.best.append(float(np.asarray(bo).mean()) if hasattr(bo, "mean") else float(bo))
        self.pop.append(losses)

        if epoch % self.update_every != 0 and epoch != total - 1:
            return
        self.progress_bar.progress(min(1.0, (epoch + 1) / total))
        elapsed = time.time() - self._start
        with self.metrics_holder.container():
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("epoch", f"{epoch + 1} / {total}")
            c2.metric("best obj", f"{self.best[-1]:.4f}")
            c3.metric("mean loss", f"{self.mean_loss[-1]:.4f}")
            c4.metric("loss σ", f"{float(losses.std()):.4f}")
            c5.metric("elapsed", f"{elapsed:.1f}s")

        # --- Dynamics plot (mean + best) ---------------------------------
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.epochs,
                y=self.mean_loss,
                mode="lines",
                name="mean loss",
                line={"color": "#60a5fa"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.epochs,
                y=self.best,
                mode="lines",
                name="best",
                line={"color": "#f472b6", "width": 3},
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=340,
            title={"text": "Live annealing dynamics", "x": 0.5},
            xaxis_title="Epoch",
            yaxis_title="Objective",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend={"x": 0.01, "y": 0.99},
        )
        self.chart_holder.plotly_chart(fig, use_container_width=True)

        # --- Parallel-population heatmap ---------------------------------
        #
        # Rows = replicas (sorted by current-epoch loss). Columns = snapshot
        # epochs. Colour = loss. White curve = best-of-batch.
        pop = np.stack(self.pop, axis=1)  # (sol_size, T)
        order = np.argsort(pop[:, -1])
        pop_sorted = pop[order]
        best_traj = pop.min(axis=0)
        pop_fig = go.Figure()
        pop_fig.add_trace(
            go.Heatmap(
                z=pop_sorted,
                x=self.epochs,
                colorscale="Viridis",
                colorbar={"title": "loss"},
                zsmooth="best",
            )
        )
        pop_fig.add_trace(
            go.Scatter(
                x=self.epochs,
                y=best_traj,
                mode="lines",
                name="best-of-batch",
                yaxis="y2",
                line={"color": "#f8fafc", "width": 2.5},
            )
        )
        pop_fig.update_layout(
            template="plotly_dark",
            height=360,
            title={"text": "Parallel population (sol_size replicas)", "x": 0.5},
            xaxis_title="Epoch",
            yaxis_title="Replica (sorted)",
            yaxis2={
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "title": "best loss",
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend={"x": 0.01, "y": 0.99},
        )
        self.pop_holder.plotly_chart(pop_fig, use_container_width=True)


run = st.button("▶  Run QQA", type="primary")
if run:
    try:
        problem = build_problem(cfg)
    except Exception as e:
        st.error(f"Could not build problem: {e}")
        st.stop()

    progress = st.progress(0.0)
    metrics = st.empty()
    c_left, c_right = st.columns(2)
    chart = c_left.empty()
    pop_holder = c_right.empty()

    cb = StreamlitCallback(progress, metrics, chart, pop_holder, update_every=update_every)
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

    st.success(f"Done. Best objective: {result.best_obj}")
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
