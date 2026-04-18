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

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
from _common import (  # noqa: E402
    DEFAULT_CUSTOM_SNIPPET,
    apply_theme,
    build_problem,
    preview_problem,
    theme_toggle_in_sidebar,
)

import qqa  # noqa: E402

# The custom-problem editor runs user-supplied Python via ``exec``. Disable
# it on public deployments by setting ``QQA_ALLOW_CUSTOM=0`` (this is the
# default on Streamlit Community Cloud / Hugging Face Spaces). Set it to
# ``1`` to re-enable on a trusted machine.
ALLOW_CUSTOM = os.getenv("QQA_ALLOW_CUSTOM", "0") == "1"

st.set_page_config(
    page_title="QQA dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme_toggle_in_sidebar()
apply_theme()

st.title("Quasi-Quantum Annealing")
st.markdown(
    '<div class="qqa-card">'
    "A unified gradient-based solver for combinatorial optimisation and "
    "spin-glass models. Pick a problem from the catalog, plug in your own "
    "loss, or tweak the relaxation and watch the parallel population "
    "anneal in real time."
    "</div>",
    unsafe_allow_html=True,
)
st.write("")


# ---------------------------------------------------------------------------
# Sidebar — problem definition
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Problem definition")

    if ALLOW_CUSTOM:
        use_custom = st.toggle(
            "Use custom problem",
            value=False,
            help="Plug in your own loss_fn directly from this UI.",
        )
    else:
        use_custom = False
        st.caption(
            "Custom-loss editor is disabled on this deployment. "
            "Set `QQA_ALLOW_CUSTOM=1` to enable it on a trusted machine."
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
        _FAMILIES = {
            "Graph (binary QUBO)": [
                ("mis", "Maximum Independent Set"),
                ("maxcut", "Max-Cut"),
                ("maxclique", "Max Clique"),
                ("vertex_cover", "Vertex Cover"),
                ("graph_bisection", "Graph bisection"),
            ],
            "Categorical / assignment": [
                ("coloring", "Graph coloring"),
                ("tsp", "Travelling Salesman (TSP)"),
                ("qap", "Quadratic Assignment (QAP)"),
                ("nqueens", "N-Queens"),
            ],
            "Classic CO": [
                ("knapsack", "0/1 Knapsack"),
                ("number_partition", "Number partitioning"),
                ("maxsat3", "MaxSAT (random 3-SAT)"),
            ],
            "Physics / spin": [
                ("ising1d", "1D Ising model"),
                ("ea", "Edwards–Anderson spin glass"),
                ("sk", "Sherrington–Kirkpatrick spin glass"),
                ("perceptron", "Binary perceptron"),
                ("hopfield", "Hopfield memory"),
            ],
        }
        _ALL_OPTS = [(k, label) for group in _FAMILIES.values() for (k, label) in group]
        _LABELS = {k: label for k, label in _ALL_OPTS}

        family = st.selectbox("Problem family", list(_FAMILIES.keys()), index=0)
        problem_kind = st.selectbox(
            "Problem",
            [k for k, _ in _FAMILIES[family]],
            format_func=lambda s: _LABELS[s],
        )

        size_default = {"tsp": 10, "qap": 8, "nqueens": 8}.get(problem_kind, 32)
        size_max = {"tsp": 20, "qap": 14, "nqueens": 14}.get(problem_kind, 400)
        size_label = {
            "tsp": "Cities N",
            "qap": "Facilities N",
            "nqueens": "Board size N",
            "maxsat3": "Variables N",
            "knapsack": "Items N",
            "number_partition": "Values N",
            "ea": "Lattice side L",
        }.get(problem_kind, "Problem size (N)")
        size = st.number_input(size_label, min_value=4, max_value=size_max, value=size_default)
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=0)

        # Per-problem auxiliary controls.
        if problem_kind in {"mis", "maxcut", "maxclique", "vertex_cover", "graph_bisection"}:
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
        if problem_kind == "knapsack":
            extra["capacity_ratio"] = st.slider("Capacity / Σwᵢ", 0.1, 0.9, 0.5, 0.05)
        if problem_kind == "number_partition":
            extra["max_value"] = st.slider("Max value", 10, 500, 100, step=10)
        if problem_kind == "graph_bisection":
            extra["balance_penalty"] = st.slider("Balance penalty", 0.5, 5.0, 2.0, 0.1)
        if problem_kind == "maxsat3":
            extra["ratio"] = st.slider("Clause ratio M/N", 1.0, 6.0, 3.0, 0.1)
        if problem_kind == "tsp":
            extra["column_penalty"] = st.slider("Column penalty λ", 1.0, 10.0, 3.0, 0.5)
        if problem_kind == "qap":
            extra["column_penalty"] = st.slider("Column penalty λ", 1.0, 30.0, 10.0, 0.5)

    # Only offer devices that are actually available on the host. On public
    # deployments (Streamlit Cloud / Hugging Face Spaces) this is always
    # ``cpu``; on a local workstation with a GPU the ``cuda`` option appears.
    import torch  # noqa: PLC0415 - lazy import to keep startup snappy

    device_options = ["cpu"]
    if torch.cuda.is_available():
        device_options.append("cuda")
    device = st.selectbox(
        "Device",
        device_options,
        index=0,
        help=(
            "The annealing loop runs inside this Streamlit process. "
            "`cuda` appears only when a GPU is visible to PyTorch."
        ),
    )


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
