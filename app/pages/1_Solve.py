"""Solve page — run PQQA or Population Annealing with live progress.

Two backends are exposed via a sidebar radio:

* **PQQA** (default) — Parallel Quasi-Quantum Annealing, the gradient-based
  CRA-PI-GNN family that ships with the library.
* **PA** — Hukushima–Iba Population Annealing with resampling, equilibrium
  sample dump, free-energy estimation, and genealogy recording. The
  recorded genealogy + ``final_x`` populate the new "PA: …" tabs on the
  Visualize page so you can inspect resampling collapse and the family
  tree of the final population.
"""

from __future__ import annotations

import inspect
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
    render_config_chips,
    render_score_card,
    sidebar_brand,
    theme_toggle_in_sidebar,
)
from _solution_viz import render_solution_view  # noqa: E402

import qqa  # noqa: E402
from qqa.callbacks import Callback, CallbackState, PopulationTracker  # noqa: E402

# Capability detection — the deployed ``qqa`` wheel is occasionally older
# than the app source (e.g. Streamlit Community Cloud caches the previous
# build, or pip resolves ``qqa`` from PyPI when ``.`` install fails). We
# probe instead of relying on the symbol being present, so the page never
# crashes with a cryptic ``module 'qqa' has no attribute …`` error.
PA_AVAILABLE = hasattr(qqa, "population_annealing") and hasattr(qqa, "PAResult")
QQA_VERSION = getattr(qqa, "__version__", "unknown")

st.set_page_config(page_title="Solve — QQA", page_icon="⚛️", layout="wide")
sidebar_brand()
theme_toggle_in_sidebar()
apply_theme()
st.title("Solve")

if "problem_config" not in st.session_state:
    from _common import empty_state_card  # noqa: PLC0415

    empty_state_card(
        title="No problem selected",
        body=(
            "Pick a built-in problem (or paste your own) on the "
            "<b>Home</b> page first. The annealer reads its "
            "configuration from there."
        ),
        cta_label="Open Home",
        cta_page="streamlit_app.py",
    )
    st.stop()
cfg = st.session_state["problem_config"]
render_config_chips(cfg)

with st.sidebar:
    st.header("2 · Backend")
    _backend_options = ["PQQA"] + (["PA (Population Annealing)"] if PA_AVAILABLE else [])
    backend = st.radio(
        "Solver backend",
        _backend_options,
        index=0,
        help=(
            "PQQA = Parallel Quasi-Quantum Annealing (gradient-based, the "
            "library's headline solver). PA = Hukushima–Iba Population "
            "Annealing with resampling — also produces equilibrium samples "
            "and a free-energy estimate."
        ),
        key="solve_backend",
    )
    if not PA_AVAILABLE:
        st.warning(
            f"PA backend not available in this build (qqa = `{QQA_VERSION}`). "
            "PA needs `qqa.population_annealing`, added in 0.5.1+. "
            "Re-deploy with the latest source (`pip install -e .` from the "
            "repo root) or pull a newer wheel.",
            icon=":material/upgrade:",
        )

# Hyper-parameter presets — backend-aware so the sidebar can scale knobs
# to whatever solver is selected. Each backend owns its own preset
# dict + apply function so adding a new solver later is a one-stop edit.
#
# PQQA: (sol_size, epochs, learning_rate, temp, min_bg, max_bg, curve_rate,
#        div_param, update_every).
_PQQA_PRESETS = {
    "🏃  Fast smoke": (32, 200, 1.0, 0.0, -2.0, 0.1, 2, 0.0, 10),
    "🎯  Default": (64, 1000, 1.0, 0.0, -2.0, 0.1, 2, 0.0, 20),
    "🔬  Thorough": (128, 3000, 0.7, 0.05, -3.0, 0.2, 4, 0.0, 50),
}
# PA: (sol_size, num_temps, sweeps_per_temp, beta_start, beta_end,
#      beta_schedule, resample, update_every_steps).
_PA_PRESETS = {
    "🏃  Fast smoke": (32, 30, 4, 0.1, 6.0, "geometric", "systematic", 1),
    "🎯  Default": (128, 100, 10, 0.1, 10.0, "geometric", "systematic", 5),
    "🔬  Thorough": (256, 200, 20, 0.05, 20.0, "geometric", "systematic", 10),
}


def _apply_pqqa_preset(name: str) -> None:
    """Seed PQQA-specific session_state keys from a preset tuple."""
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
    for k, v in zip(keys, _PQQA_PRESETS[name], strict=True):
        st.session_state[k] = v


def _apply_pa_preset(name: str) -> None:
    """Seed PA-specific session_state keys from a preset tuple."""
    keys = (
        "pa_sol_size",
        "pa_num_temps",
        "pa_sweeps_per_temp",
        "pa_beta_start",
        "pa_beta_end",
        "pa_beta_schedule",
        "pa_resample",
        "pa_update_every",
    )
    for k, v in zip(keys, _PA_PRESETS[name], strict=True):
        st.session_state[k] = v


# -----------------------------------------------------------------------
# Backend-conditional sidebar.
#
# We render *only* the controls that affect the selected backend. PQQA's
# learning_rate / Langevin temp / cooling schedule have no PA analogue, and
# PA's R / num_temps / β grid have no PQQA analogue, so showing both at
# once made the sidebar long and the "inactive controls" surface area
# noisy (cf. the screenshot regression that motivated this refactor).
# -----------------------------------------------------------------------
if backend == "PQQA":
    with st.sidebar:
        st.header("3 · PQQA hyper-parameters")
        st.caption("Parallel Quasi-Quantum Annealing — gradient-based solver.")
        preset_name = st.radio(
            "Preset",
            list(_PQQA_PRESETS),
            index=1,
            horizontal=False,
            help="Quickly seed every PQQA slider. You can still tweak any value.",
            key="pqqa_preset",
        )
        if st.button("Apply preset", width="stretch", key="apply_pqqa_preset_btn"):
            _apply_pqqa_preset(preset_name)
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
                help="Magnitude of stochastic noise per step (0 = deterministic).",
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

        with st.expander("Post-processing & warm-start", expanded=False):
            polish = st.toggle(
                "Greedy 1-flip polish (recommended)",
                value=st.session_state.get("polish", True),
                key="polish",
                help=(
                    "Run a deterministic single-bit local search on the "
                    "annealer's winner. Skipped on non-QUBO problems."
                ),
            )
            # Warm-start only makes sense when the problem exposes a
            # NetworkX graph the BFS heuristic can read. We surface the
            # toggle for any graph problem and gate the call at runtime.
            warm_start = st.toggle(
                "BFS 2-color warm-start (graph QUBOs only)",
                value=st.session_state.get("warm_start", False),
                key="warm_start",
                help="Seed each replica with the BFS-tree 2-coloring.",
            )

        with st.expander("Display", expanded=False):
            update_every = st.slider(
                "UI update every (epochs)",
                1,
                200,
                st.session_state.get("update_every", 20),
                key="update_every",
                help="Lower = smoother animation but slower wall-clock.",
            )
else:
    # PA mode: render *only* PA's knobs. The PQQA-only variables below
    # are bound to safe defaults so the rest of this file (which still
    # references them inside the `if backend == "PQQA"` run block) stays
    # syntactically valid.
    sol_size = st.session_state.get("sol_size", 64)
    epochs = st.session_state.get("epochs", 1000)
    curve_rate = st.session_state.get("curve_rate", 2)
    learning_rate = st.session_state.get("learning_rate", 1.0)
    temp = st.session_state.get("temp", 0.0)
    min_bg = st.session_state.get("min_bg", -2.0)
    max_bg = st.session_state.get("max_bg", 0.1)
    div_param = st.session_state.get("div_param", 0.0)
    polish = st.session_state.get("polish", True)
    warm_start = st.session_state.get("warm_start", False)
    update_every = st.session_state.get("update_every", 20)

    with st.sidebar:
        st.header("3 · PA hyper-parameters")
        st.caption(
            "Hukushima–Iba Population Annealing with resampling. Returns "
            "an equilibrium population, free-energy estimate, and a full "
            "resampling genealogy."
        )
        pa_preset_name = st.radio(
            "Preset",
            list(_PA_PRESETS),
            index=1,
            horizontal=False,
            help="Seed every PA slider in one click.",
            key="pa_preset",
        )
        if st.button("Apply preset", width="stretch", key="apply_pa_preset_btn"):
            _apply_pa_preset(pa_preset_name)
            st.rerun()

        with st.expander("Population & schedule", expanded=True):
            pa_sol_size = st.slider(
                "PA population (R)",
                4,
                512,
                st.session_state.get("pa_sol_size", 128),
                key="pa_sol_size",
                help="Number of replicas carried through resampling.",
            )
            pa_num_temps = st.slider(
                "num_temps (T)",
                10,
                500,
                st.session_state.get("pa_num_temps", 100),
                step=10,
                key="pa_num_temps",
                help="Number of inverse-temperature steps.",
            )
            pa_sweeps_per_temp = st.slider(
                "sweeps_per_temp (K)",
                1,
                50,
                st.session_state.get("pa_sweeps_per_temp", 10),
                key="pa_sweeps_per_temp",
                help="MCMC sweeps after each resampling step.",
            )

        with st.expander("β schedule", expanded=False):
            pa_beta_start = st.slider(
                "β_start",
                0.01,
                2.0,
                st.session_state.get("pa_beta_start", 0.1),
                step=0.01,
                key="pa_beta_start",
                help="Initial inverse temperature.",
            )
            pa_beta_end = st.slider(
                "β_end",
                0.5,
                100.0,
                st.session_state.get("pa_beta_end", 10.0),
                step=0.5,
                key="pa_beta_end",
                help="Final inverse temperature (= 1/T_min).",
            )
            pa_beta_schedule = st.selectbox(
                "β schedule",
                ["geometric", "linear"],
                index=["geometric", "linear"].index(
                    st.session_state.get("pa_beta_schedule", "geometric")
                ),
                key="pa_beta_schedule",
                help="Geometric (recommended for spin-glass) or linear.",
            )

        with st.expander("Resampling", expanded=False):
            pa_resample = st.selectbox(
                "Resampling rule",
                ["systematic", "multinomial"],
                index=["systematic", "multinomial"].index(
                    st.session_state.get("pa_resample", "systematic")
                ),
                key="pa_resample",
                help=(
                    "Systematic = low-variance (Doucet & Johansen, 2008). "
                    "Multinomial = textbook baseline (higher variance)."
                ),
            )

        with st.expander("Post-processing", expanded=False):
            pa_polish = st.checkbox(
                "Greedy 1-flip polish",
                value=st.session_state.get("pa_polish", True),
                key="pa_polish",
                help=(
                    "Run ``qqa.polish.greedy_one_flip`` on PA's best replica "
                    "after the last β step — identical post-processing to "
                    "PQQA, guarantees a 1-flip local optimum on QUBO problems. "
                    "Disable to see the raw MCMC best."
                ),
            )

        with st.expander("Display", expanded=False):
            pa_update_every = st.slider(
                "UI update every (steps)",
                1,
                25,
                st.session_state.get("pa_update_every", 5),
                key="pa_update_every",
                help="Lower = smoother live charts but slower wall-clock.",
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
        # Best *relaxed* loss of the current epoch (= losses.min()).  Same
        # units as ``mean_loss``, so they share a y-axis cleanly.  Earlier
        # versions plotted ``state.best_obj`` (the running min of the
        # *discrete* loss), which for problems like SK / number-partitioning
        # / 3-SAT lives on a vastly different scale than the relaxed loss.
        # On a shared axis the discrete value visually pinned to 0 and the
        # line looked broken — see ``tasks/lessons.md``.
        self.best_relaxed: list[float] = []
        # Running min of the discrete loss — the same number ``AnnealResult``
        # ultimately returns. Surfaced as a metric card, not a chart line.
        self.best_disc: list[float] = []
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
        self.best_relaxed.append(float(losses.min()))
        bo = state.best_obj
        self.best_disc.append(float(np.asarray(bo).mean()) if hasattr(bo, "mean") else float(bo))
        self.pop.append(losses)

        if epoch % self.update_every != 0 and epoch != total - 1:
            return
        self.progress_bar.progress(min(1.0, (epoch + 1) / total))
        elapsed = time.time() - self._start
        with self.metrics_holder.container():
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric("epoch", f"{epoch + 1} / {total}")
            r1c2.metric(
                "best (discrete)",
                f"{self.best_disc[-1]:.4g}",
                help="Running min of loss_fn evaluated on the projected discrete sample — "
                "this is the value AnnealResult.best_obj will report.",
            )
            r1c3.metric(
                "best (relaxed, this epoch)",
                f"{self.best_relaxed[-1]:.4g}",
                help="min over replicas of the relaxed loss at this epoch — same scale as mean.",
            )
            r1c4.metric("mean", f"{self.mean_loss[-1]:.4g}")
            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            r2c1.metric("σ (replicas)", f"{self.std_loss[-1]:.4g}")
            r2c2.metric("bg", f"{state.bg:.3f}")
            r2c3.metric("elapsed", f"{elapsed:.1f}s")
            r2c4.metric("update_every", f"{self.update_every}")

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
                y=self.best_relaxed,
                mode="lines",
                name="best of epoch (relaxed)",
                line={"color": p["palette"][1], "width": 2.4, "dash": "solid"},
                hovertemplate="epoch=%{x}<br>best (relaxed)=%{y:.4f}<extra></extra>",
            )
        )
        # Annotate the running discrete best in the corner so users can see
        # the integer-scale value alongside the relaxed curves without
        # having to pin two wildly-different scales onto the same axis.
        disc_text = f"best (discrete) = {self.best_disc[-1]:.4g}"
        fig.update_layout(
            **plotly_layout(
                height=320,
                title={"text": "Annealing dynamics (per-replica loss)"},
                xaxis_title="Epoch",
                yaxis_title="Loss (per replica)",
                legend={
                    "x": 0.01,
                    "y": 0.02,
                    "bgcolor": "rgba(255,255,255,0.6)"
                    if theme == "light"
                    else "rgba(15,23,42,0.6)",
                },
                annotations=[
                    {
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.99,
                        "y": 0.98,
                        "xanchor": "right",
                        "yanchor": "top",
                        "showarrow": False,
                        "text": disc_text,
                        "font": {"size": 12, "color": p["muted"]},
                        "bgcolor": "rgba(255,255,255,0.55)"
                        if theme == "light"
                        else "rgba(15,23,42,0.55)",
                        "bordercolor": p["border"],
                        "borderwidth": 1,
                        "borderpad": 4,
                    }
                ],
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


# Compact "active hyper-params" chip row, so the user can see what is
# about to run without scrolling the sidebar.
if backend == "PQQA":
    render_config_chips(
        cfg,
        extras={
            "sol_size": sol_size,
            "epochs": epochs,
            "lr": f"{learning_rate:.2g}",
            "T": f"{temp:.2g}",
            "polish": "on" if polish else "off",
            "warm-start": "on" if warm_start else "off",
        },
    )
else:
    render_config_chips(
        cfg,
        extras={
            "backend": "PA",
            "R": pa_sol_size,
            "T": pa_num_temps,
            "K": pa_sweeps_per_temp,
            "β": f"{pa_beta_start:.2g}→{pa_beta_end:.2g}",
            "resample": pa_resample,
        },
    )

# ---------------------------------------------------------------------------
# PA capability probe — build the problem once and ask the chain-backend
# validator whether PA can actually sample it. Gives users a friendly,
# actionable warning BEFORE they click Run instead of the opaque
# ``NotImplementedError: ... uses a CategoricalRelaxation`` stacktrace they
# used to get on TSP / QAP / Coloring / NQueens / BGP (categorical) and on
# explicitly selected structured binary formulations. The probe is cheap — it only imports
# the validator and the built problem, never starts MCMC.
# ---------------------------------------------------------------------------
pa_capability_error: str | None = None
if backend != "PQQA":
    try:
        _probe_problem = build_problem(cfg)
    except Exception as _e:  # noqa: BLE001
        pa_capability_error = f"Could not build problem: {_e}"
    else:
        try:
            from qqa.sa import _validate_chain_problem as _pa_validate  # noqa: PLC0415

            _pa_validate(_probe_problem)
        except NotImplementedError as _e:
            pa_capability_error = str(_e)
        except Exception:  # noqa: BLE001
            # Other TypeErrors are surfaced by the actual run; don't block.
            pa_capability_error = None

    if pa_capability_error:
        st.warning(
            "**Population Annealing is not available for this problem.** "
            f"{pa_capability_error}  \n\n"
            "Switch **Backend → PQQA** in the sidebar — PQQA handles "
            "categorical and permutation models (TSP / QAP / Coloring / "
            "NQueens) natively. Or pick a QUBO / Ising / spin "
            "family on **Home** (MIS, MaxCut, MaxClique, VertexCover, "
            "GraphBisection, MDS, Knapsack, NumberPartition, Ising, SK, EA, "
            "RFIM, p-spin, Hopfield, Perceptron) to keep PA.",
            icon=":material/warning:",
        )

run_label = "▶  Run QQA" if backend == "PQQA" else "▶  Run Population Annealing"
run = st.button(
    run_label,
    type="primary",
    width="stretch",
    disabled=(backend != "PQQA" and pa_capability_error is not None),
)
if run and backend == "PQQA":
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

    # Build the warm-start seed when requested AND the problem exposes a
    # graph attribute. Falls back silently otherwise so the toggle never
    # crashes a non-graph run.
    initial_state = None
    if warm_start:
        graph = getattr(problem, "nx_graph", None) or getattr(problem, "graph", None)
        if graph is not None:
            try:
                initial_state = qqa.warmstart.bfs_2color(graph).to(cfg["device"])
                st.caption(
                    f"warm-started {sol_size} replicas from BFS 2-coloring "
                    f"({initial_state.shape[0]} bits)."
                )
            except Exception as exc:
                # Don't fail the whole run if the heuristic chokes on an
                # unusual graph; just log and fall back to the random init.
                st.warning(f"warm-start unavailable: {exc}")

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
            initial_state=initial_state,
            polish=polish,
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
        else float(np.asarray(result.best_obj).mean())
    )
    # The callback tracks the *un-polished* running best (it fires inside
    # the annealing loop, before greedy_one_flip runs). Inject it into
    # the score dict so render_score_card can show a "polish improved
    # by Δ" badge when polish actually moved the needle. We pass it via
    # the dict (instead of a kwarg) so an older deployed _common.py
    # cannot raise ``TypeError: got an unexpected keyword argument``.
    import contextlib  # noqa: PLC0415

    score_payload = dict(result.score) if isinstance(result.score, dict) else {}
    if polish and result.polished_sol is not None and cb.best_disc:
        with contextlib.suppress(TypeError, ValueError):
            score_payload["pre_polish_loss"] = float(cb.best_disc[-1])
    with score_holder.container():
        render_score_card(score_payload, raw_loss=raw)
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
                "polish": polish,
                "warm_start": warm_start,
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


# =============================================================================
# Population Annealing (PA) backend
# =============================================================================
if run and backend != "PQQA":
    if not PA_AVAILABLE:
        st.error(
            f"Population Annealing is not exposed by the installed `qqa` "
            f"(version `{QQA_VERSION}`). The deployed wheel is older than this "
            "Streamlit app — re-deploy with the latest source so "
            "`qqa.population_annealing` is importable."
        )
        st.stop()
    try:
        problem = build_problem(cfg)
    except Exception as e:
        st.error(f"Could not build problem: {e}")
        st.stop()

    progress = st.progress(0.0)
    metrics = st.empty()
    score_holder = st.empty()
    chart = st.empty()
    free_energy_holder = st.empty()

    pa_loss_mean: list[float] = []
    pa_best: list[float] = []
    pa_ess: list[float] = []
    pa_steps: list[int] = []

    pa_total_steps = int(pa_num_temps)
    pa_update_every = max(1, int(st.session_state.get("pa_update_every", 5)))
    pa_t0 = time.time()

    def _pa_callback(step: int, mean_loss: float, best_obj: float, ess: float) -> None:
        pa_steps.append(step)
        pa_loss_mean.append(float(mean_loss))
        pa_best.append(float(best_obj))
        pa_ess.append(float(ess))
        if step % pa_update_every != 0 and step != pa_total_steps - 1:
            return
        progress.progress(min(1.0, (step + 1) / pa_total_steps))
        elapsed = time.time() - pa_t0
        with metrics.container():
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("step", f"{step + 1} / {pa_total_steps}")
            r2.metric("best", f"{best_obj:.4g}")
            r3.metric("mean", f"{mean_loss:.4g}")
            r4.metric("ESS", f"{ess:.1f} / {int(pa_sol_size)}")
            r5, r6, r7, r8 = st.columns(4)
            r5.metric("elapsed", f"{elapsed:.1f}s")
            r6.metric("R", f"{int(pa_sol_size)}")
            r7.metric("K", f"{int(pa_sweeps_per_temp)}")
            r8.metric("backend", "PA")
        # Live convergence plot. Two stacked sub-plots so the loss and ESS
        # curves do not fight for the same y-axis (the screenshot regression
        # had the legend, "best so far" line and "ESS" line all overlapping
        # in the corner because they shared the plot area).
        p = palette()
        from plotly.subplots import make_subplots  # noqa: PLC0415

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.06,
            subplot_titles=("Loss vs temperature step", "Effective Sample Size"),
        )
        fig.add_trace(
            go.Scatter(
                x=pa_steps,
                y=pa_loss_mean,
                mode="lines",
                name="mean / replica",
                line={"color": p["palette"][0], "width": 2},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=pa_steps,
                y=pa_best,
                mode="lines",
                name="best so far",
                line={"color": p["palette"][1], "width": 2.4},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=pa_steps,
                y=pa_ess,
                mode="lines",
                name="ESS",
                line={"color": p["palette"][2], "width": 1.8},
                fill="tozeroy",
                fillcolor=hex_to_rgba(p["palette"][2], 0.18),
            ),
            row=2,
            col=1,
        )
        # ESS ceiling = R guideline (Kish: ESS ∈ [1, R]).
        fig.add_trace(
            go.Scatter(
                x=[pa_steps[0] if pa_steps else 0, pa_steps[-1] if pa_steps else 1],
                y=[int(pa_sol_size), int(pa_sol_size)],
                mode="lines",
                name=f"R = {int(pa_sol_size)}",
                line={"color": p["muted"], "width": 1, "dash": "dash"},
                showlegend=True,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Temperature step", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="ESS", row=2, col=1, range=[0, int(pa_sol_size) * 1.05])
        fig.update_layout(
            **plotly_layout(
                height=460,
                legend={
                    "orientation": "h",
                    "x": 0.0,
                    "y": 1.10,
                    "xanchor": "left",
                    "yanchor": "bottom",
                },
                margin={"t": 70, "b": 50, "l": 60, "r": 40},
            )
        )
        chart.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

    try:
        pa_kwargs = dict(
            problem=problem,
            sol_size=int(pa_sol_size),
            num_temps=int(pa_num_temps),
            sweeps_per_temp=int(pa_sweeps_per_temp),
            beta_schedule=pa_beta_schedule,
            beta_start=float(pa_beta_start),
            beta_end=float(pa_beta_end),
            resample=pa_resample,
            device=cfg["device"],
            record_genealogy=True,
            verbose=False,
            callback=_pa_callback,
        )
        # Guard against older ``qqa`` wheels that pre-date the PA ``polish``
        # knob (added alongside this UI control). The wheel check is cheap
        # and keeps the app usable against any 0.5.1+ release; newer wheels
        # get the QQA-symmetric post-processing by default.
        if "polish" in inspect.signature(qqa.population_annealing).parameters:
            pa_kwargs["polish"] = bool(pa_polish)
        pa_result = qqa.population_annealing(**pa_kwargs)
    except NotImplementedError as e:
        # Specific to "PA does not support this problem class" — give the
        # user a clear next step instead of a generic traceback string.
        st.error(
            f"**Population Annealing cannot solve this problem.**\n\n{e}\n\n"
            "Switch to **Backend → PQQA** in the sidebar; PQQA handles all "
            "problem families this UI exposes."
        )
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"PA run failed: {e}")
        st.stop()

    progress.empty()
    raw = float(pa_result.best_obj)
    with score_holder.container():
        render_score_card(pa_result.score, raw_loss=raw)

    # If the post-run polish strictly improved the MCMC's best, tell the
    # user: otherwise the live chart ends at ``mcmc_best`` while the score
    # card shows the strictly-better ``polished best_obj`` and the gap
    # looks like a reporting bug.
    polished = getattr(pa_result, "polished_sol", None)
    if polished is not None and pa_best:
        mcmc_best = float(pa_best[-1])
        if raw < mcmc_best - 1e-9:
            st.caption(
                f"Post-run greedy 1-flip polish improved the MCMC best "
                f"from ``{mcmc_best:.4g}`` to ``{raw:.4g}`` — the score "
                "card reflects the polished value (PAResult.best_sol "
                "is automatically overwritten when polish strictly wins)."
            )

    # Free-energy density curve
    if pa_result.history.get("beta") and pa_result.history.get("free_energy_density"):
        p = palette()
        fig_f = go.Figure()
        fig_f.add_trace(
            go.Scatter(
                x=pa_result.history["beta"],
                y=pa_result.history["free_energy_density"],
                mode="lines+markers",
                line={"color": p["palette"][3 % len(p["palette"])], "width": 2.4},
                name="F(β)/N",
            )
        )
        fig_f.update_layout(
            **plotly_layout(
                height=320,
                title={"text": "Free-energy density estimate (Hukushima–Iba)"},
                xaxis_title="β (inverse temperature)",
                yaxis_title="F(β) / N",
                xaxis={"type": "log"} if pa_beta_schedule == "geometric" else None,
            )
        )
        free_energy_holder.plotly_chart(
            fig_f, width="stretch", theme=None, config={"displayModeBar": False}
        )
        st.caption(
            f"PA estimate at β={float(pa_beta_end):.3g}: "
            f"**F/N = {pa_result.free_energy_density:.4f}**, "
            f"ln Z = {pa_result.log_z:.3f}. The estimator uses the "
            "average unnormalised reweighting factor at every annealing "
            "step, so the resampling correction is implicit."
        )

    st.session_state["last_result"] = pa_result
    st.session_state["last_problem"] = problem
    st.session_state["last_pop_tracker"] = None  # PA stores its own population
    st.session_state["last_pa_result"] = pa_result

    st.markdown("### Solution")
    st.caption(
        "Problem-aware view of the best configuration PA found. The "
        "**Visualize** page shows the full equilibrium population, ESS "
        "history, free-energy curve, and a resampling family tree."
    )
    render_solution_view(problem, pa_result, cfg)
    st.info(
        "Open **Visualize** to see the equilibrium population, ESS over β, "
        "the free-energy density curve, and the resampling family tree."
    )


paper_link_footer()
