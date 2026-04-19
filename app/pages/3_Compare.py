"""Compare page — two modes:

1. **QQA hyper-parameter sweep** — grid over ``min_bg``/``max_bg``/``div_param``.
2. **PQQA vs SA shootout** — head-to-head against the SA baseline on the
   currently-selected problem, designed to make the speed gap obvious.
"""

from __future__ import annotations

import contextlib
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
from _common import (  # noqa: E402
    apply_theme,
    build_problem,
    paper_link_footer,
    plotly_layout,
    sidebar_brand,
    theme_toggle_in_sidebar,
)

import qqa  # noqa: E402
from qqa import visualization as viz  # noqa: E402
from qqa.relaxation import CategoricalRelaxation  # noqa: E402


def _retheme(fig):
    with contextlib.suppress(Exception):
        fig.update_layout(**plotly_layout())
    return fig


def _scalar_best(obj) -> float:
    """Return a Python float regardless of whether ``best_obj`` is a list."""
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, list):
        return float(min(obj))
    return float(obj)


st.set_page_config(page_title="Compare — QQA", page_icon="⚛️", layout="wide")
sidebar_brand()
theme_toggle_in_sidebar()
apply_theme()
st.title("Compare")

if "problem_config" not in st.session_state:
    from _common import empty_state_card  # noqa: PLC0415

    empty_state_card(
        title="No problem selected",
        body=(
            "Pick or define a problem on the <b>Home</b> page first. "
            "Compare runs a small hyper-parameter sweep (or a head-to-head "
            "PQQA vs SA shootout) on the currently-selected problem."
        ),
        cta_label="Open Home",
        cta_page="streamlit_app.py",
    )
    st.stop()
cfg = st.session_state["problem_config"]

with st.sidebar:
    mode = st.radio(
        "Compare mode",
        ["QQA hyper-parameter sweep", "PQQA vs SA shootout"],
        index=0,
        help=(
            "Sweep: grid-search QQA hyper-parameters on the current problem. "
            "Shootout: race PQQA against the SA baseline at a matched compute "
            "budget — useful for showing how much the relaxation buys you."
        ),
    )


# =============================================================================
# Mode 1 — QQA hyper-parameter sweep (existing)
# =============================================================================
if mode == "QQA hyper-parameter sweep":
    with st.sidebar:
        st.header("3. Sweep configuration")
        epochs = st.slider("epochs per trial", 100, 3000, 500, step=100)
        sol_size = st.slider("sol_size", 4, 200, 32)
        min_bgs = st.multiselect(
            "min_bg grid", [-5.0, -3.0, -2.0, -1.0, -0.5], default=[-3.0, -2.0, -1.0]
        )
        max_bgs = st.multiselect("max_bg grid", [0.0, 0.1, 0.5, 1.0], default=[0.0, 0.1])
        div_params = st.multiselect("div_param grid", [0.0, 0.1, 0.3], default=[0.0, 0.1])
        run = st.button("▶ Run sweep", type="primary")

    if run:
        if not min_bgs or not max_bgs or not div_params:
            st.error("All three grid lists must have at least one value.")
            st.stop()
        trials = list(itertools.product(min_bgs, max_bgs, div_params))
        st.info(f"Running {len(trials)} trials…")
        rows: list[dict] = []
        results: list = []
        labels: list[str] = []
        prog = st.progress(0.0)
        for i, (mb, Mb, dp) in enumerate(trials):
            problem = build_problem(cfg)
            r = qqa.anneal(
                problem,
                sol_size=sol_size,
                min_bg=float(mb),
                max_bg=float(Mb),
                div_param=float(dp),
                num_epochs=epochs,
                device=cfg["device"],
                verbose=False,
            )
            rows.append(
                {
                    "min_bg": float(mb),
                    "max_bg": float(Mb),
                    "div_param": float(dp),
                    "best_obj": _scalar_best(r.best_obj),
                    "runtime": float(r.runtime),
                }
            )
            results.append(r)
            labels.append(f"bg=[{mb},{Mb}] dp={dp}")
            prog.progress((i + 1) / len(trials))

        st.success("Sweep complete.")
        st.subheader("Results table")
        st.dataframe(rows, width="stretch")

        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            fig = viz.plot_parallel_coordinates(
                df, objective="best_obj", backend="plotly", show=False
            )
            st.subheader("Parallel coordinates")
            st.plotly_chart(_retheme(fig), width="stretch")
        except Exception as e:
            st.info(f"Parallel-coordinates unavailable: {e}")

        st.subheader("Run comparison")
        fig2 = viz.plot_run_comparison(results, labels=labels, backend="plotly", show=False)
        st.plotly_chart(_retheme(fig2), width="stretch")


# =============================================================================
# Mode 2 — PQQA vs SA shootout (new)
# =============================================================================
elif mode == "PQQA vs SA shootout":
    # SA only supports binary / spin relaxations; refuse upfront on Categorical.
    probe_problem = build_problem(cfg)
    sa_supported = not isinstance(getattr(probe_problem, "relaxation", None), CategoricalRelaxation)

    st.markdown(
        "Race **Parallel QQA** against a textbook **Simulated Annealing** "
        "baseline on the same problem. Both run on the device you picked on "
        "Home, with a comparable compute budget — the convergence plot below "
        "exposes the wall-clock gap directly."
    )

    if not sa_supported:
        st.warning(
            "SA is not supported for this problem (categorical relaxation). "
            "Switch the problem on Home to a QUBO / Ising / spin family "
            "(e.g. MIS, Max-Cut, SK) to use the shootout."
        )

    with st.sidebar:
        st.header("3. Shootout configuration")
        st.caption("PQQA")
        pqqa_epochs = st.slider("PQQA epochs", 100, 5000, 1000, step=100, key="pqqa_epochs")
        pqqa_sol_size = st.slider("PQQA sol_size", 4, 256, 64, key="pqqa_sol")
        pqqa_min_bg = st.slider("PQQA min_bg", -10.0, 0.0, -2.0, step=0.5, key="pqqa_minbg")
        pqqa_max_bg = st.slider("PQQA max_bg", -1.0, 2.0, 0.1, step=0.1, key="pqqa_maxbg")

        st.caption("SA baseline")
        sa_sweeps = st.slider("SA num_sweeps", 100, 10_000, 1000, step=100, key="sa_sweeps")
        sa_chains = st.slider("SA chains (sol_size)", 4, 256, 64, key="sa_chains")
        sa_beta_start = st.slider("SA β_start", 0.01, 1.0, 0.1, step=0.01, key="sa_b0")
        sa_beta_end = st.slider("SA β_end", 1.0, 100.0, 10.0, step=1.0, key="sa_b1")
        sa_schedule = st.selectbox("SA schedule", ["geometric", "linear"], key="sa_sched")

        run_shootout = st.button("▶ Run shootout", type="primary", disabled=not sa_supported)

    if sa_supported and run_shootout:
        with st.spinner("Running PQQA…"):
            problem_pqqa = build_problem(cfg)
            res_pqqa = qqa.anneal(
                problem_pqqa,
                sol_size=int(pqqa_sol_size),
                num_epochs=int(pqqa_epochs),
                min_bg=float(pqqa_min_bg),
                max_bg=float(pqqa_max_bg),
                device=cfg["device"],
                verbose=False,
            )

        with st.spinner("Running SA baseline…"):
            problem_sa = build_problem(cfg)
            res_sa = qqa.simulated_annealing(
                problem_sa,
                sol_size=int(sa_chains),
                num_sweeps=int(sa_sweeps),
                beta_start=float(sa_beta_start),
                beta_end=float(sa_beta_end),
                beta_schedule=sa_schedule,
                device=cfg["device"],
                verbose=False,
            )

        pqqa_best = _scalar_best(res_pqqa.best_obj)
        sa_best = _scalar_best(res_sa.best_obj)

        # Side-by-side summary cards.
        col_p, col_s = st.columns(2)
        col_p.metric(
            "PQQA best_obj",
            f"{pqqa_best:.4f}",
            delta=f"runtime {res_pqqa.runtime:.2f} s",
            delta_color="off",
        )
        col_s.metric(
            "SA best_obj",
            f"{sa_best:.4f}",
            delta=f"runtime {res_sa.runtime:.2f} s",
            delta_color="off",
        )

        # "How much faster did PQQA reach SA's best?" by walking the SA history.
        sa_history = res_sa.history.get("best_obj", []) or [sa_best]
        # Time per SA sweep (rough), so we can map sweep index -> wall-clock.
        sa_dt = float(res_sa.runtime) / max(1, len(sa_history))
        # Find the first sweep at which SA reached the PQQA best.
        sa_reach_idx = next(
            (i for i, v in enumerate(sa_history) if float(v) <= pqqa_best),
            None,
        )

        if sa_reach_idx is None:
            st.warning(
                f"SA never matched PQQA's best ({pqqa_best:.4f}). "
                f"SA stalled at {sa_best:.4f} after {res_sa.runtime:.2f}s. "
                "Increase `SA num_sweeps` or `β_end` to give SA more room."
            )
        else:
            sa_time_to_pqqa = sa_dt * (sa_reach_idx + 1)
            speedup = sa_time_to_pqqa / max(res_pqqa.runtime, 1e-6)
            st.success(
                f"PQQA reached `best_obj = {pqqa_best:.4f}` in "
                f"**{res_pqqa.runtime:.2f}s**, while SA needed "
                f"~**{sa_time_to_pqqa:.2f}s** to match it "
                f"(speedup ≈ **{speedup:.1f}×**)."
            )

        # Convergence overlay: x = epoch / sweep index, y = best_obj.
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            pqqa_hist = res_pqqa.history.get("best_obj", []) or [pqqa_best]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(pqqa_hist))),
                    y=[float(v) for v in pqqa_hist],
                    name="PQQA",
                    mode="lines",
                    line={"width": 3},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sa_history))),
                    y=[float(v) for v in sa_history],
                    name="SA",
                    mode="lines",
                    line={"width": 3, "dash": "dash"},
                )
            )
            fig.update_layout(
                title="Best-objective trajectory",
                xaxis_title="iteration (PQQA epoch / SA sweep)",
                yaxis_title="best_obj (lower is better)",
                **plotly_layout(),
            )
            st.subheader("Convergence")
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.info(f"Plot unavailable: {e}")

        # Final results table for export.
        st.subheader("Summary table")
        st.dataframe(
            [
                {
                    "backend": "PQQA",
                    "best_obj": pqqa_best,
                    "runtime_s": float(res_pqqa.runtime),
                    "iterations": int(pqqa_epochs),
                    "sol_size": int(pqqa_sol_size),
                },
                {
                    "backend": "SA",
                    "best_obj": sa_best,
                    "runtime_s": float(res_sa.runtime),
                    "iterations": int(sa_sweeps),
                    "sol_size": int(sa_chains),
                },
            ],
            width="stretch",
        )


paper_link_footer()
