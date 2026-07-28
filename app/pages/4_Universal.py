"""Universal optimisation studio for mixed, Pareto, black-box, and TeX models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from _common import (  # noqa: E402
    apply_theme,
    paper_link_footer,
    retheme_plotly,
    sidebar_brand,
    theme_toggle_in_sidebar,
)

import qqa  # noqa: E402

_PRODUCTION_SPEC = {
    "name": "regional-production-plan",
    "variables": [
        {"name": "open_a", "kind": "binary", "lower": 0, "upper": 1, "size": 1},
        {"name": "open_b", "kind": "binary", "lower": 0, "upper": 1, "size": 1},
        {"name": "lots_a", "kind": "integer", "lower": 0, "upper": 12, "size": 1},
        {"name": "lots_b", "kind": "integer", "lower": 0, "upper": 10, "size": 1},
        {"name": "overtime", "kind": "real", "lower": 0, "upper": 16, "size": 1},
    ],
    "objectives": [
        {
            "name": "weekly_cost",
            "direction": "min",
            "expression": (
                "1400*open_a + 1100*open_b + 460*lots_a + 510*lots_b + 38*square(overtime)"
            ),
            "unit": "USD",
        }
    ],
    "constraints": [
        {
            "name": "demand",
            "expression": "8*lots_a + 7*lots_b + overtime",
            "sense": ">=",
            "rhs": 105,
            "weight": 1000,
            "scale": 105,
            "tolerance": 0.05,
        },
        {
            "name": "plant_a_link",
            "expression": "lots_a - 12*open_a",
            "sense": "<=",
            "rhs": 0,
            "weight": 500,
            "scale": 12,
            "tolerance": 0.01,
        },
        {
            "name": "plant_b_link",
            "expression": "lots_b - 10*open_b",
            "sense": "<=",
            "rhs": 0,
            "weight": 500,
            "scale": 10,
            "tolerance": 0.01,
        },
    ],
    "notes": "Two plants, integer production lots, continuous overtime, and activation links.",
}


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _json_download(label: str, payload: dict, filename: str) -> None:
    st.download_button(
        label,
        json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


st.set_page_config(page_title="Universal Studio — QQA", page_icon="🌐", layout="wide")
sidebar_brand()
theme_toggle_in_sidebar()
apply_theme()

st.title("Universal Optimization Studio")
st.markdown(
    """
<div class="qqa-card">
Build and solve practical models without leaving the dashboard:
<b>binary + integer + real planning</b>, a GPU-parallel
<b>Pareto front</b>, constrained <b>black-box experiments</b>, and an
auditable <b>TeX → QQA → SCIP</b> workflow.
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Universal runtime")
    detected = _device()
    device = st.selectbox(
        "Compute device",
        ("cpu", "cuda") if detected == "cuda" else ("cpu",),
        index=1 if detected == "cuda" else 0,
    )
    seed = int(st.number_input("Seed", min_value=0, max_value=100_000, value=7))
    st.caption(f"Detected accelerator: **{detected}**")

mixed_tab, pareto_tab, blackbox_tab, tex_tab = st.tabs(
    ["⚡ Mixed planning", "◎ Pareto studio", "◇ Black-box lab", "∑ TeX model"]
)

with mixed_tab:
    st.subheader("Microgrid unit commitment & dispatch")
    st.caption(
        "Four generators, on/off decisions, continuous dispatch, integer storage, "
        "demand response, unit coupling, demand balance, and spinning reserve."
    )
    left, right = st.columns([1, 2])
    with left:
        mixed_population = st.slider("Parallel plans", 32, 1024, 256, 32, key="u_mixed_pop")
        mixed_epochs = st.slider("Iterations", 100, 3000, 900, 100, key="u_mixed_epochs")
        run_mixed = st.button(
            "Optimize dispatch",
            type="primary",
            use_container_width=True,
            key="u_run_mixed",
        )
    if run_mixed:
        qqa.fix_seed(seed)
        model = qqa.build_microgrid_dispatch()
        with st.spinner("Exploring mixed dispatch plans…"):
            result = model.solve(
                sol_size=mixed_population,
                num_epochs=mixed_epochs,
                device=device,
                verbose=False,
            )
        st.session_state["universal_mixed"] = (model, result)
    if "universal_mixed" in st.session_state:
        model, result = st.session_state["universal_mixed"]
        with right:
            a, b, c = st.columns(3)
            a.metric("Operating cost", f"${result.score['value']:,.0f}/h")
            b.metric("Feasible", "Yes" if result.score["feasible"] else "No")
            b_data = result.score["extra"]["variables"]
            c.metric("Storage units", f"{b_data['storage_units']:.0f}")
            rows = [{"variable": name, "value": value} for name, value in b_data.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            _json_download("Download result", result.score, "microgrid-dispatch.json")
    else:
        with right:
            st.info("Choose a population and click **Optimize dispatch**.")

with pareto_tab:
    st.subheader("Cost × emissions × resilience in one parallel run")
    st.caption(
        "Each GPU replica follows a different reference direction; the live archive "
        "keeps feasible nondominated plans and recommends a scale-invariant knee."
    )
    controls, chart = st.columns([1, 3])
    with controls:
        pareto_population = st.slider("Reference directions", 64, 2048, 512, 64)
        pareto_epochs = st.slider("Pareto iterations", 100, 3000, 1000, 100)
        run_pareto = st.button(
            "Generate Pareto front",
            type="primary",
            use_container_width=True,
        )
    if run_pareto:
        model = qqa.build_microgrid_pareto()
        with st.spinner("Building the nondominated archive…"):
            result = model.solve_pareto(
                sol_size=pareto_population,
                num_epochs=pareto_epochs,
                device=device,
                seed=seed,
                verbose=False,
            )
        st.session_state["universal_pareto"] = (model, result)
    if "universal_pareto" in st.session_state:
        model, result = st.session_state["universal_pareto"]
        knee = result.select()
        with controls:
            st.metric("Pareto plans", len(result.solutions))
            st.metric("Recommended knee", knee)
            st.dataframe(
                pd.DataFrame(
                    {
                        "objective": result.objective_names,
                        "value": result.objectives[knee].detach().cpu().tolist(),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                "Download front CSV",
                result.to_frame(model).to_csv(index=False),
                file_name="microgrid-pareto.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with chart:
            figure = qqa.plot_pareto(result, show=False, title="Microgrid Pareto front")
            st.plotly_chart(retheme_plotly(figure), use_container_width=True)
    else:
        with chart:
            st.info("Click **Generate Pareto front** to explore the trade-off surface.")

with blackbox_tab:
    st.subheader("Expensive process simulator tuning")
    st.caption(
        "Catalyst choice, reactor count, temperature, residence time, and recycle "
        "are tuned under yield, heat-capacity, and staffing constraints."
    )
    controls, chart = st.columns([1, 3])
    with controls:
        budget = st.slider("Experiment budget", 24, 256, 96, 8)
        batch_size = st.slider("Parallel batch", 1, 16, 8)
        workers = st.slider("Evaluation workers", 1, 16, 4)
        run_blackbox = st.button(
            "Run virtual experiments",
            type="primary",
            use_container_width=True,
        )
    if run_blackbox:
        model = qqa.build_process_blackbox()
        with st.spinner("Fitting surrogate and scheduling experiments…"):
            result = model.solve(
                budget=budget,
                batch_size=batch_size,
                workers=workers,
                device=device,
                seed=seed,
                verbose=False,
            )
        st.session_state["universal_blackbox"] = (model, result)
    if "universal_blackbox" in st.session_state:
        model, result = st.session_state["universal_blackbox"]
        with controls:
            st.metric("Best hourly profit", f"${result.best_value:,.0f}")
            st.metric("Feasible", "Yes" if result.feasible else "No")
            st.json(result.best_point)
            st.download_button(
                "Download experiments CSV",
                result.to_frame(model).to_csv(index=False),
                file_name="process-experiments.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with chart:
            figure = qqa.plot_blackbox(result, show=False, title="Process tuning")
            st.plotly_chart(retheme_plotly(figure), use_container_width=True)
    else:
        with chart:
            st.info("Click **Run virtual experiments** to start budget-aware optimisation.")

with tex_tab:
    st.subheader("Auditable TeX / JSON modelling")
    st.caption(
        "Review the generated model before solving. API credentials stay in memory "
        "for this run and are never embedded in the model or downloads."
    )
    source_mode = st.radio(
        "Input",
        ("Audited JSON", "TeX via compatible API"),
        horizontal=True,
    )
    if source_mode == "Audited JSON":
        model_source = st.text_area(
            "Validated model JSON",
            json.dumps(_PRODUCTION_SPEC, ensure_ascii=False, indent=2),
            height=360,
        )
        if st.button("Validate audited model", use_container_width=True):
            try:
                spec = qqa.ModelSpec.from_json(model_source)
            except (TypeError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state["universal_spec"] = spec
                st.success("Safe grammar, domains, bounds, and constraints validated.")
    else:
        tex_source = st.text_area(
            "TeX optimisation model",
            (
                r"\min\ 1400y_A+1100y_B+460n_A+510n_B+38h^2"
                "\n"
                r"\mathrm{s.t.}\ 8n_A+7n_B+h\ge105,\ "
                r"n_A\le12y_A,\ n_B\le10y_B,\ "
                r"y_A,y_B\in\{0,1\},\ n_A\in\{0,\ldots,12\},\ "
                r"n_B\in\{0,\ldots,10\},\ 0\le h\le16."
            ),
            height=170,
        )
        api_key = st.text_input("API key", type="password")
        api_base = st.text_input("OpenAI-compatible base URL", value=qqa.tex.DEFAULT_BASE_URL)
        model_name = st.text_input("Model", value=qqa.tex.DEFAULT_MODEL)
        insecure = st.checkbox("Trusted private gateway uses a non-standard certificate")
        if st.button("Translate & validate TeX", use_container_width=True):
            if not api_key:
                st.error("Enter an API key for this in-memory request.")
            else:
                try:
                    client = qqa.OpenAICompatibleClient(
                        api_key=api_key,
                        base_url=api_base,
                        model=model_name,
                        verify_ssl=not insecure,
                    )
                    spec = qqa.compile_tex(tex_source, client=client)
                except Exception as exc:  # noqa: BLE001 - user-facing API boundary
                    st.error(str(exc))
                else:
                    st.session_state["universal_spec"] = spec
                    st.success("Translation validated. Review the model below before solving.")

    spec = st.session_state.get("universal_spec")
    if spec is not None:
        st.code(spec.to_json(), language="json")
        left, right = st.columns(2)
        with left:
            if len(spec.objectives) == 1:
                use_scip = st.toggle(
                    "SCIP proof phase",
                    value=False,
                    help="Requires pip install qqa[scip].",
                )
            else:
                use_scip = False
                st.info("Multi-objective models use the parallel Pareto solver.")
        with right:
            solve_model = st.button(
                "Solve reviewed model", type="primary", use_container_width=True
            )
        if solve_model:
            problem = qqa.problem_from_spec(spec)
            with st.spinner("Solving reviewed model…"):
                if isinstance(problem, qqa.MultiObjectiveProblem):
                    result = problem.solve_pareto(
                        sol_size=256,
                        num_epochs=1000,
                        device=device,
                        seed=seed,
                        verbose=False,
                    )
                elif use_scip:
                    result = qqa.solve_spec_scip(
                        spec,
                        qqa_kwargs={
                            "sol_size": 256,
                            "num_epochs": 1000,
                            "device": device,
                            "verbose": False,
                        },
                        time_limit=60,
                    )
                else:
                    result = problem.solve(
                        sol_size=256,
                        num_epochs=1000,
                        device=device,
                        verbose=False,
                    )
            if isinstance(result, qqa.ParetoResult):
                knee = result.select()
                st.success(
                    f"Found {len(result.solutions):,} nondominated plans · recommended knee #{knee}"
                )
                st.dataframe(
                    result.to_frame(problem),
                    hide_index=True,
                    use_container_width=True,
                )
                figure = qqa.plot_pareto(
                    result,
                    show=False,
                    title=f"{spec.name} Pareto front",
                )
                st.plotly_chart(retheme_plotly(figure), use_container_width=True)
                st.download_button(
                    "Download Pareto front",
                    result.to_frame(problem).to_csv(index=False),
                    file_name=f"{spec.name}-pareto.csv",
                    mime="text/csv",
                )
            else:
                st.success(
                    f"Objective: {result.score.get('value', result.best_obj):,.6g} · "
                    f"Feasible: {result.score.get('feasible', True)}"
                )
                st.json(result.score)
        st.download_button(
            "Download audited model",
            spec.to_json(),
            file_name=f"{spec.name}.json",
            mime="application/json",
        )

paper_link_footer()
