"""Compare page — two modes:

1. **QQA hyper-parameter sweep** — grid over ``min_bg``/``max_bg``/``div_param``.
2. **PQQA vs SA shootout** — head-to-head against the SA baseline on the
   currently-selected problem, designed to make the speed gap obvious.
"""

from __future__ import annotations

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
from _common import (
    retheme_plotly as _retheme,
)

import qqa  # noqa: E402
from qqa import visualization as viz  # noqa: E402
from qqa.relaxation import CategoricalRelaxation  # noqa: E402


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
        ["QQA hyper-parameter sweep", "PQQA vs SA vs PA shootout"],
        index=0,
        help=(
            "Sweep: grid-search QQA hyper-parameters on the current problem. "
            "Shootout: race PQQA against the SA and Population-Annealing "
            "baselines at a matched compute budget."
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
# Mode 2 — PQQA vs SA vs PA shootout
# =============================================================================
elif mode == "PQQA vs SA vs PA shootout":
    # SA / PA only support binary / spin relaxations; refuse on Categorical.
    probe_problem = build_problem(cfg)
    chain_supported = not isinstance(
        getattr(probe_problem, "relaxation", None), CategoricalRelaxation
    )

    st.markdown(
        "Race **Parallel QQA** against two textbook MCMC baselines — "
        "**Simulated Annealing (SA)** and **Population Annealing (PA, with "
        "resampling)** — on the same problem. Total compute is matched: "
        "PA's ``num_temps × sweeps_per_temp`` ≈ SA's ``num_sweeps``."
    )

    if not chain_supported:
        st.warning(
            "SA / PA do not support categorical-relaxation problems. "
            "Pick a QUBO / Ising / spin family (e.g. MIS, Max-Cut, SK) on "
            "**Home** to use the shootout."
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

        st.caption("Population Annealing baseline")
        pa_temps = st.slider("PA num_temps", 10, 500, 100, step=10, key="pa_temps")
        pa_sweeps = st.slider("PA sweeps_per_temp", 1, 50, 10, key="pa_sweeps")
        pa_chains = st.slider("PA population (sol_size)", 4, 256, 64, key="pa_chains")
        pa_beta_start = st.slider("PA β_start", 0.01, 1.0, 0.1, step=0.01, key="pa_b0")
        pa_beta_end = st.slider("PA β_end", 1.0, 100.0, 10.0, step=1.0, key="pa_b1")
        pa_resample = st.selectbox("PA resample", ["systematic", "multinomial"], key="pa_resample")

        run_shootout = st.button("▶ Run shootout", type="primary", disabled=not chain_supported)

    if chain_supported and run_shootout:
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

        with st.spinner("Running Population Annealing baseline…"):
            problem_pa = build_problem(cfg)
            res_pa = qqa.population_annealing(
                problem_pa,
                sol_size=int(pa_chains),
                num_temps=int(pa_temps),
                sweeps_per_temp=int(pa_sweeps),
                beta_start=float(pa_beta_start),
                beta_end=float(pa_beta_end),
                resample=pa_resample,
                device=cfg["device"],
                verbose=False,
            )

        pqqa_best = _scalar_best(res_pqqa.best_obj)
        sa_best = _scalar_best(res_sa.best_obj)
        pa_best = _scalar_best(res_pa.best_obj)

        col_p, col_s, col_pa = st.columns(3)
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
        col_pa.metric(
            "PA best_obj",
            f"{pa_best:.4f}",
            delta=f"runtime {res_pa.runtime:.2f} s",
            delta_color="off",
        )

        # "How much faster did PQQA reach the best baseline's best?"
        baseline_best = min(sa_best, pa_best)
        baseline_label = "SA" if sa_best <= pa_best else "PA"
        baseline_res = res_sa if sa_best <= pa_best else res_pa
        baseline_history = baseline_res.history.get("best_obj", []) or [baseline_best]
        baseline_dt = float(baseline_res.runtime) / max(1, len(baseline_history))
        baseline_reach = next(
            (i for i, v in enumerate(baseline_history) if float(v) <= pqqa_best),
            None,
        )

        if baseline_reach is None:
            st.warning(
                f"Neither SA nor PA matched PQQA's best ({pqqa_best:.4f}). "
                f"Best baseline = {baseline_label} at {baseline_best:.4f} "
                "after the chosen budget. Increase the baseline's compute or "
                "tune β to give it more room."
            )
        else:
            baseline_t = baseline_dt * (baseline_reach + 1)
            speedup = baseline_t / max(res_pqqa.runtime, 1e-6)
            st.success(
                f"PQQA reached `best_obj = {pqqa_best:.4f}` in "
                f"**{res_pqqa.runtime:.2f}s**, while {baseline_label} "
                f"(the stronger baseline) needed ~**{baseline_t:.2f}s** to "
                f"match it (speedup ≈ **{speedup:.1f}×**)."
            )

        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            for name, hist, fallback, dash in (
                ("PQQA", res_pqqa.history.get("best_obj", []), pqqa_best, None),
                ("SA", res_sa.history.get("best_obj", []), sa_best, "dash"),
                ("PA", res_pa.history.get("best_obj", []), pa_best, "dot"),
            ):
                hist = hist or [fallback]
                line = {"width": 3}
                if dash:
                    line["dash"] = dash
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(hist))),
                        y=[float(v) for v in hist],
                        name=name,
                        mode="lines",
                        line=line,
                    )
                )
            fig.update_layout(
                title="Best-objective trajectory",
                xaxis_title="iteration (PQQA epoch / SA sweep / PA temperature step)",
                yaxis_title="best_obj (lower is better)",
                **plotly_layout(),
            )
            st.subheader("Convergence")
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.info(f"Plot unavailable: {e}")

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
                {
                    "backend": "PA",
                    "best_obj": pa_best,
                    "runtime_s": float(res_pa.runtime),
                    "iterations": int(pa_temps) * int(pa_sweeps),
                    "sol_size": int(pa_chains),
                },
            ],
            width="stretch",
        )


paper_link_footer()
