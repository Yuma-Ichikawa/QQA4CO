"""QQA Streamlit dashboard — Home page.

Launch with ``uv run qqa gui`` or::

    uv run streamlit run app/streamlit_app.py

The dashboard is a 4-page app:

* Home (this file) — pick a built-in or custom problem, set parameters, preview.
* Solve — run QQA with live progress and a live parallel-population view.
* Visualize — inspect history, best trajectory, and the best solution.
* Compare — run a small sweep and inspect parallel-coordinates.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
from _common import (  # noqa: E402
    DEFAULT_CUSTOM_SNIPPET,
    apply_theme,
    build_problem,
    preview_problem,
)

import qqa  # noqa: E402

st.set_page_config(
    page_title="QQA dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

st.title("Quasi-Quantum Annealing")
st.markdown(
    '<div class="qqa-card">'
    "Define a combinatorial or spin-glass problem, plug in your own loss "
    "function, tune QQA, and explore the results interactively."
    "</div>",
    unsafe_allow_html=True,
)
st.write("")


# ---------------------------------------------------------------------------
# Sidebar — problem definition
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Problem definition")

    use_custom = st.toggle(
        "Use custom problem",
        value=False,
        help="Plug in your own loss_fn directly from this UI.",
    )

    extra: dict = {}
    if use_custom:
        problem_kind = "custom"
        extra["variable_kind"] = st.selectbox(
            "Variable kind",
            ("spin", "binary", "categorical"),
            index=0,
            help=("'spin' ∈ {-1,+1}, 'binary' ∈ {0,1}, 'categorical' one-hot over K choices."),
        )
        extra["num_vars"] = int(
            st.number_input("Number of variables N", min_value=2, max_value=4096, value=32)
        )
        if extra["variable_kind"] == "categorical":
            extra["num_category"] = int(st.number_input("K (categories)", 2, 32, 3))
        extra["name"] = st.text_input("Problem name", value="custom")
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=0)
        size = extra["num_vars"]
        st.markdown("Edit the snippet in the main panel →")
    else:
        problem_kind = st.selectbox(
            "Problem family",
            (
                "mis",
                "maxcut",
                "maxclique",
                "coloring",
                "ising1d",
                "ea",
                "sk",
                "perceptron",
                "hopfield",
            ),
            index=0,
            format_func=lambda s: {
                "mis": "Maximum Independent Set",
                "maxcut": "Max-Cut",
                "maxclique": "Max Clique",
                "coloring": "Graph coloring",
                "ising1d": "1D Ising model",
                "ea": "Edwards–Anderson spin glass",
                "sk": "Sherrington–Kirkpatrick spin glass",
                "perceptron": "Binary perceptron (teacher-student)",
                "hopfield": "Hopfield memory",
            }[s],
        )

        size = st.number_input("Problem size (N or L)", min_value=4, max_value=400, value=32)
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=0)

        if problem_kind in {"mis", "maxcut", "maxclique"}:
            extra["graph_d"] = st.slider("Random-regular degree d", 2, 8, 3)
        if problem_kind == "coloring":
            extra["num_category"] = st.slider("Number of colours K", 2, 6, 3)
            extra["graph_d"] = st.slider("Random-regular degree d", 2, 8, 3)
        if problem_kind == "ea":
            extra["dim"] = st.selectbox("Lattice dim", (2, 3), index=1)
        if problem_kind == "perceptron":
            extra["alpha"] = st.slider("Loading α = M/N", 0.1, 1.5, 0.5, 0.1)
        if problem_kind == "hopfield":
            extra["patterns"] = st.slider("Stored patterns P", 1, 20, 3)

    device = st.selectbox("Device", ("cpu", "cuda"), index=0)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if use_custom:
    st.subheader("Custom loss editor")
    st.markdown(
        "Define a function named `loss_fn(x)` that maps a batched configuration "
        "tensor to a `(B,)` loss vector. The namespace already has `torch` and "
        "`np` (numpy) imported. Any constants (couplings, patterns, ...) you "
        "declare at module scope are captured by closure."
    )
    source = st.text_area(
        "Snippet",
        value=st.session_state.get("custom_source", DEFAULT_CUSTOM_SNIPPET),
        height=320,
        key="custom_source_editor",
    )
    st.session_state["custom_source"] = source
    extra["source"] = source

st.session_state["problem_config"] = {
    "kind": problem_kind,
    "size": int(size),
    "seed": int(seed),
    "device": device,
    "extra": extra,
}

cfg = st.session_state["problem_config"]

left, right = st.columns([1, 2])
with left:
    st.subheader("Summary")
    st.code(
        f"problem: {cfg['kind']}\n"
        f"size   : {cfg['size']}\n"
        f"device : {cfg['device']}\n"
        f"seed   : {cfg['seed']}\n"
        f"extra  : {cfg['extra']}",
        language="text",
    )
    st.info(
        "Open **Solve** in the left sidebar to run QQA on this problem.",
        icon="➡️",
    )

with right:
    st.subheader("Preview")
    try:
        problem = build_problem(cfg)
        preview_problem(problem, cfg)
    except Exception as e:
        st.error(f"Could not build problem: {e}")

st.markdown("---")
st.caption(f"QQA v{qqa.__version__} · PyTorch + Streamlit · gradient-based parallel annealer")
