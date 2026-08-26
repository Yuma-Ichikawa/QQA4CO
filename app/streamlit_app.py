"""QQA Streamlit dashboard — Home page.

Launch with ``uv run qqa gui`` or::

    uv run streamlit run app/streamlit_app.py

The dashboard is a 5-page app:

* Home (this file) — pick a built-in or custom problem, set parameters, preview.
* Solve — run QQA with live progress and a live parallel-population view.
* Visualize — inspect history, best trajectory, and the best solution.
* Compare — run a small sweep and inspect parallel-coordinates.
* Universal — mixed, Pareto, black-box, and TeX/SCIP workflows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
from _common import (  # noqa: E402
    CUSTOM_EXAMPLES,
    DEFAULT_CUSTOM_SNIPPET,
    apply_theme,
    build_problem,
    hero_badges,
    paper_link_footer,
    preview_problem,
    sidebar_brand,
    theme_toggle_in_sidebar,
)

import qqa  # noqa: E402

# The custom-problem editor runs user-supplied Python via ``exec`` inside
# the Streamlit process. Keep it disabled unless the operator explicitly
# enables it on a trusted local deployment; a browser user must never gain
# code execution on a shared/public server by default.
ALLOW_CUSTOM = os.getenv("QQA_ALLOW_CUSTOM", "0") == "1"

st.set_page_config(
    page_title="QQA dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)
sidebar_brand()
theme_toggle_in_sidebar()
apply_theme()

st.title("Quasi-Quantum Annealing")
hero_badges()
st.markdown(
    '<div class="qqa-card">'
    "A unified gradient-based solver for combinatorial optimisation and "
    "spin-glass models. Pick a problem from the catalogue, plug in your "
    "own loss, or tweak the relaxation, then click <b>Solve</b> to watch "
    "the parallel population anneal in real time."
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
            help=(
                "Plug in your own loss_fn directly from this UI. The code "
                "runs in this Streamlit process — only paste code you trust."
            ),
            key="use_custom",
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
        # The form values are session-state-keyed so the "Load template"
        # button on the main panel can write the right defaults here on rerun.
        kind_options = ("spin", "binary", "categorical")
        kind_default = st.session_state.get("custom_kind", "spin")
        extra["variable_kind"] = st.selectbox(
            "Variable kind",
            kind_options,
            index=kind_options.index(kind_default) if kind_default in kind_options else 0,
            key="custom_kind",
            help=("'spin' ∈ {-1,+1}, 'binary' ∈ {0,1}, 'categorical' one-hot over K choices."),
        )
        extra["num_vars"] = int(
            st.number_input(
                "Number of variables N",
                min_value=2,
                max_value=4096,
                value=int(st.session_state.get("custom_num_vars", 32)),
                key="custom_num_vars",
            )
        )
        if extra["variable_kind"] == "categorical":
            extra["num_category"] = int(
                st.number_input(
                    "K (categories)",
                    2,
                    32,
                    int(st.session_state.get("custom_num_category", 3)),
                    key="custom_num_category",
                )
            )
        extra["name"] = st.text_input(
            "Problem name", value=st.session_state.get("custom_name", "custom"), key="custom_name"
        )
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=0)
        size = extra["num_vars"]
        st.success("Edit & validate the snippet in the main panel →", icon="📝")
    else:
        # Catalog. Each entry is (kind, label, required qqa attribute).
        # The ``required`` field lets us prune problems whose underlying
        # class is not present in the installed qqa module — for example
        # when a Streamlit Cloud build is still on the previous release
        # while the README catalog already advertises the new ones.
        # Without the prune, the user would pick "RFIM" from the menu,
        # hit Run, and see ``module 'qqa' has no attribute ...`` with no
        # way to recover. With the prune, the option simply doesn't
        # appear until the deployment catches up.
        import qqa as _qqa  # noqa: PLC0415

        _CATALOG: dict[str, list[tuple[str, str, str]]] = {
            "Graph (binary QUBO)": [
                ("mis", "Maximum Independent Set", "MaximumIndependentSet"),
                ("maxcut", "Max-Cut", "MaxCut"),
                ("maxclique", "Max Clique", "MaxClique"),
                ("vertex_cover", "Vertex Cover", "VertexCover"),
                ("graph_bisection", "Graph bisection", "GraphBisection"),
                ("min_dominating_set", "Minimum Dominating Set", "MinimumDominatingSet"),
            ],
            "Categorical / assignment": [
                ("coloring", "Graph coloring", "Coloring"),
                ("bgp", "Balanced graph partition (K-way)", "BalancedGraphPartition"),
                ("tsp", "Travelling Salesman (TSP)", "TSP"),
                ("qap", "Quadratic Assignment (QAP)", "QAP"),
                ("nqueens", "N-Queens", "NQueens"),
            ],
            "Classic CO": [
                ("knapsack", "0/1 Knapsack", "Knapsack"),
                ("number_partition", "Number partitioning", "NumberPartitioning"),
                ("maxsat3", "MaxSAT (random 3-SAT)", "MaxSAT3"),
            ],
            "Physics / spin": [
                ("ising1d", "1D Ising model", "Ising1D"),
                ("ea", "Edwards–Anderson spin glass", "EdwardsAnderson"),
                ("sk", "Sherrington–Kirkpatrick spin glass", "SherringtonKirkpatrick"),
                ("pspin", "p-spin glass (dense, p ≥ 2)", "PSpinGlass"),
                ("rfim", "Random Field Ising Model (RFIM)", "RandomFieldIsing"),
                ("perceptron", "Binary perceptron", "BinaryPerceptron"),
                ("hopfield", "Hopfield memory", "HopfieldMemory"),
            ],
        }

        _FAMILIES: dict[str, list[tuple[str, str]]] = {}
        _MISSING: list[str] = []
        for fam, entries in _CATALOG.items():
            keep = [(k, lab) for (k, lab, attr) in entries if hasattr(_qqa, attr)]
            if keep:
                _FAMILIES[fam] = keep
            _MISSING.extend(f"{lab}" for (_, lab, attr) in entries if not hasattr(_qqa, attr))

        _ALL_OPTS = [(k, label) for group in _FAMILIES.values() for (k, label) in group]
        _LABELS = {k: label for k, label in _ALL_OPTS}

        family = st.selectbox("Problem family", list(_FAMILIES.keys()), index=0)
        problem_kind = st.selectbox(
            "Problem",
            [k for k, _ in _FAMILIES[family]],
            format_func=lambda s: _LABELS[s],
        )
        if _MISSING:
            # Loud, non-dismissable banner — a quiet ``st.caption`` was
            # easy to miss and users repeatedly asked "where did p-spin /
            # RFIM go?". Explicit > implicit.
            st.warning(
                f"⚠️  Your installed `qqa=={_qqa.__version__}` is missing "
                f"{len(_MISSING)} problem(s) advertised by this UI: "
                + ", ".join(f"**{m}**" for m in _MISSING)
                + ". Run `pip install -U qqa` (or redeploy this app) to "
                "enable them.",
                icon="📦",
            )

        size_default = {"tsp": 8, "qap": 8, "nqueens": 8, "ea": 6, "rfim": 8, "pspin": 16}.get(
            problem_kind, 32
        )
        size_max = {"tsp": 20, "qap": 14, "nqueens": 14, "ea": 16, "rfim": 20, "pspin": 32}.get(
            problem_kind, 400
        )
        size_label = {
            "tsp": "Cities N",
            "qap": "Facilities N",
            "nqueens": "Board size N",
            "maxsat3": "Variables N",
            "knapsack": "Items N",
            "number_partition": "Values N",
            "ea": "Lattice side L",
            "rfim": "Lattice side L",
            "pspin": "Spins N",
        }.get(problem_kind, "Problem size (N)")
        size = st.number_input(size_label, min_value=4, max_value=size_max, value=size_default)
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=0)

        # Per-problem auxiliary controls.
        if problem_kind in {
            "mis",
            "maxcut",
            "maxclique",
            "vertex_cover",
            "graph_bisection",
            "min_dominating_set",
        }:
            extra["graph_d"] = st.slider("Random-regular degree d", 2, 8, 3)
        if problem_kind == "coloring":
            extra["num_category"] = st.slider("Number of colours K", 2, 6, 3)
            extra["graph_d"] = st.slider("Random-regular degree d", 2, 8, 3)
        if problem_kind == "bgp":
            extra["num_category"] = st.slider("Number of partitions K", 2, 8, 3)
            extra["graph_d"] = st.slider("Random-regular degree d", 2, 8, 3)
            extra["balance_penalty"] = st.slider(
                "Balance penalty", 0.0001, 0.01, 0.0005, 0.0001, format="%.4f"
            )
        if problem_kind == "ea":
            extra["dim"] = st.selectbox("Lattice dim", (2, 3), index=1)
        if problem_kind == "rfim":
            extra["dim"] = st.selectbox("Lattice dim", (1, 2, 3), index=1)
            extra["coupling_J"] = st.slider("Ferromagnetic coupling J", 0.1, 5.0, 1.0, 0.1)
            extra["h_std"] = st.slider("Random-field σ_h", 0.0, 5.0, 1.0, 0.1)
        if problem_kind == "pspin":
            extra["p_order"] = st.slider("Interaction order p", 2, 5, 3)
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
            st.caption(
                "TSP uses a Sinkhorn doubly-stochastic relaxation by default, "
                "followed by explicit Hungarian repair and 2-opt refinement. "
                "The binary penalty formulation remains available as an "
                "explicit compatibility option."
            )
            extra["relaxation"] = st.selectbox(
                "TSP relaxation",
                ("sinkhorn", "binary"),
                index=0,
                format_func=lambda value: (
                    "Sinkhorn (recommended)" if value == "sinkhorn" else "Binary penalty (opt-in)"
                ),
                help=(
                    "Sinkhorn preserves the doubly-stochastic assignment structure. "
                    "Binary penalty is retained for controlled compatibility studies."
                ),
            )
            if extra["relaxation"] == "binary":
                sync = st.toggle(
                    "Sync λ_r = λ_c",
                    value=True,
                    key="tsp_sync_lambda",
                    help="Tie the row and column penalty sliders together.",
                )
                if sync:
                    lam = st.slider(
                        "Penalty weight λ",
                        0.5,
                        20.0,
                        5.0,
                        0.5,
                        help=("Shared weight for both (Σ_i x[t,i] − 1)² and (Σ_t x[t,i] − 1)²."),
                    )
                    extra["row_penalty"] = lam
                    extra["col_penalty"] = lam
                else:
                    extra["row_penalty"] = st.slider(
                        "Row penalty λ_r (each position holds 1 city)",
                        0.5,
                        20.0,
                        5.0,
                        0.5,
                        help="Weight on (Σ_i x[t,i] − 1)² summed over positions t.",
                    )
                    extra["col_penalty"] = st.slider(
                        "Col penalty λ_c (each city visited exactly once)",
                        0.5,
                        20.0,
                        5.0,
                        0.5,
                        help="Weight on (Σ_t x[t,i] − 1)² summed over cities i.",
                    )
        if problem_kind == "qap":
            extra["column_penalty"] = st.slider(
                "Column penalty λ",
                1.0,
                30.0,
                10.0,
                0.5,
                help="Weight on the assignment-uniqueness penalty.",
            )

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
    with st.expander("How does this work? (3 steps)", expanded=False):
        st.markdown(
            """
1. **Pick a template** from the dropdown — it auto-fills the sidebar form
   (variable kind, N, K) and replaces the editor with a working snippet.
2. **Edit `loss_fn(x)`** to your problem.  Anything you declare at module
   scope (couplings, patterns, matrices, ...) is captured by closure and
   only runs once per build.
3. **Hit ✓ Validate** to evaluate the loss on a single random batch and
   confirm the shape contract before running the full anneal.
            """
        )
        st.code(
            "def loss_fn(x: torch.Tensor) -> torch.Tensor:\n"
            "    # x.shape =\n"
            "    #   (B, N)         for variable_kind in {'spin','binary'}\n"
            "    #   (B, N, K)      for variable_kind == 'categorical'\n"
            "    # return a tensor of shape (B,) — the per-replica loss.\n"
            "    ...",
            language="python",
        )
    st.warning(
        "The snippet below is executed via Python `exec` inside this "
        "Streamlit process. Only paste code you trust; the editor is "
        "intentionally exposed for research use, not for hosting "
        "untrusted user code.",
        icon="⚠️",
    )

    example_keys = list(CUSTOM_EXAMPLES)
    col_l, col_r = st.columns([2, 1])
    with col_l:
        example_choice = st.selectbox(
            "Template",
            example_keys,
            index=example_keys.index(st.session_state.get("custom_example", example_keys[0]))
            if st.session_state.get("custom_example") in example_keys
            else 0,
            key="custom_example",
            help="Replace the editor with a fully-working, ready-to-anneal snippet.",
        )
    with col_r:
        st.write("")
        if st.button("📥  Load template", width="stretch"):
            meta = CUSTOM_EXAMPLES[example_choice]
            st.session_state["custom_source"] = meta["source"]
            st.session_state["custom_kind"] = meta["kind"]
            st.session_state["custom_num_vars"] = int(meta["num_vars"])
            if meta["kind"] == "categorical" and meta.get("num_category"):
                st.session_state["custom_num_category"] = int(meta["num_category"])
            st.session_state["custom_name"] = (
                example_choice.split("·")[-1].strip().lower().replace(" ", "-")
            )
            st.rerun()
    st.caption(f"📌 *{CUSTOM_EXAMPLES[example_choice]['description']}*")

    source = st.text_area(
        "Snippet (`loss_fn` definition + module-level constants)",
        value=st.session_state.get("custom_source", DEFAULT_CUSTOM_SNIPPET),
        height=340,
        key="custom_source_editor",
    )
    st.session_state["custom_source"] = source
    extra["source"] = source

    # The Validate button below the editor short-circuits the run.  Even
    # without it the Preview panel re-evaluates on every rerun, but giving
    # users an explicit "I want to test it now" affordance reduces the
    # cognitive load of "why is the page reloading every keystroke".
    validate = st.button(
        "✓  Validate snippet",
        type="primary",
        help="Build the problem and run loss_fn on a single random batch.",
    )
    if validate:
        st.session_state["_custom_validate_requested"] = True

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
    # For built-in problems we always preview — it's cheap.  For custom
    # problems we wait for an explicit "Validate" click to avoid running
    # arbitrary user code on every keystroke (the text_area triggers a
    # rerun on every change).  After the first successful validate, we
    # keep previewing on every rerun so it feels responsive.
    should_preview = cfg["kind"] != "custom" or st.session_state.get(
        "_custom_validate_requested", False
    )
    if should_preview:
        try:
            problem = build_problem(cfg)
            preview_problem(problem, cfg)
        except Exception as e:
            st.error(f"Could not build problem: {e}")
    else:
        st.info(
            "Click **✓ Validate snippet** below the editor to test your "
            "`loss_fn` on a random batch before running the full anneal.",
            icon="✏️",
        )

st.markdown("---")
st.caption(f"QQA v{qqa.__version__} · PyTorch + Streamlit · gradient-based parallel annealer")

paper_link_footer()
