"""Compare page — run a hyper-parameter sweep and inspect results."""

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


def _retheme(fig):
    with contextlib.suppress(Exception):
        fig.update_layout(**plotly_layout())
    return fig


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
            "Compare runs a small hyper-parameter sweep across the "
            "currently-selected problem, then ranks the results."
        ),
        cta_label="Open Home",
        cta_page="streamlit_app.py",
    )
    st.stop()
cfg = st.session_state["problem_config"]

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
                "best_obj": float(r.best_obj)
                if not hasattr(r.best_obj, "tolist")
                else float(r.best_obj.mean()),
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
        fig = viz.plot_parallel_coordinates(df, objective="best_obj", backend="plotly", show=False)
        st.subheader("Parallel coordinates")
        st.plotly_chart(_retheme(fig), width="stretch")
    except Exception as e:
        st.info(f"Parallel-coordinates unavailable: {e}")

    st.subheader("Run comparison")
    fig2 = viz.plot_run_comparison(results, labels=labels, backend="plotly", show=False)
    st.plotly_chart(_retheme(fig2), width="stretch")


paper_link_footer()
