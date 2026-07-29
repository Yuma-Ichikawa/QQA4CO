"""Universal optimisation studio for mixed, Pareto, black-box, and TeX models."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
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


def _solve_reviewed_model(spec, *, use_scip: bool, device: str, seed: int):
    """Run one reviewed spec with a single UI error boundary."""
    qqa.fix_seed(seed)
    problem = qqa.problem_from_spec(spec)
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
    return problem, result


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

ask_tab, mixed_tab, pareto_tab, blackbox_tab, tex_tab = st.tabs(
    ["✦ Ask QQA", "⚡ Mixed planning", "◎ Pareto studio", "◇ Black-box lab", "∑ TeX model"]
)

with ask_tab:
    st.subheader("Describe the decision. QQA builds and runs the workflow.")
    st.caption(
        "Write the variables, bounds, goals, and constraints in ordinary language. "
        "The generated model is validated locally before QQA, QQA+SCIP, Pareto, "
        "or black-box optimisation can run."
    )
    request = st.text_area(
        "What should QQA optimise?",
        (
            "Plan weekly production at two plants. Decide whether each plant opens "
            "(binary), production lots from 0 to 12 and 0 to 10 (integers), and "
            "overtime from 0 to 16 hours (real). Minimize 1400*open_a + 1100*open_b "
            "+ 460*lots_a + 510*lots_b + 38*overtime^2. Meet demand "
            "8*lots_a + 7*lots_b + overtime >= 105 and production must be zero when "
            "its plant is closed."
        ),
        height=190,
        key="ask_request",
    )
    settings, output = st.columns([1, 3])
    with settings:
        workflow_label = st.selectbox(
            "Workflow",
            ("Auto", "QQA", "QQA + SCIP", "Parallel Pareto", "Black-box"),
            help=(
                "Auto uses objective count, black-box intent, and local SCIP availability. "
                "The routing decision is always shown before the result."
            ),
        )
        workflow = {
            "Auto": "auto",
            "QQA": "qqa",
            "QQA + SCIP": "qqa-scip",
            "Parallel Pareto": "pareto",
            "Black-box": "blackbox",
        }[workflow_label]
        with st.expander("Model API", expanded=False):
            configured_key = bool(os.getenv("QQA_LLM_API_KEY") or os.getenv("QQA_LLM_API_KEY"))
            api_key = st.text_input(
                "API key",
                type="password",
                help=(
                    "Used only for this translation request. It is never included in "
                    "models, downloads, logs, or reports."
                ),
                key="ask_api_key",
            )
            if configured_key and not api_key:
                st.caption("Using the QQA LLM API key from the server environment.")
            api_base = st.text_input(
                "Compatible base URL",
                value=qqa.tex.DEFAULT_BASE_URL,
                key="ask_api_base",
            )
            model_name = st.text_input(
                "Model",
                value=qqa.tex.DEFAULT_MODEL,
                key="ask_model",
            )
            insecure = st.checkbox(
                "Trusted private gateway uses a non-standard certificate",
                key="ask_insecure",
            )
        with st.expander("Compute budget", expanded=False):
            ask_population = st.slider("Parallel plans", 32, 2048, 256, 32, key="ask_population")
            ask_epochs = st.slider("QQA iterations", 50, 4000, 1200, 50, key="ask_epochs")
            ask_budget = st.slider("Black-box evaluations", 16, 512, 96, 8, key="ask_budget")
            ask_batch = st.slider("Black-box parallel batch", 1, 32, 8, key="ask_batch")
        plan_only = st.button("Build reviewed plan", use_container_width=True)
        plan_and_solve = st.button(
            "Plan & solve",
            type="primary",
            use_container_width=True,
        )

    request_digest = hashlib.sha256(
        json.dumps(
            {
                "request": request,
                "workflow": workflow,
                "api_base": api_base,
                "model": model_name,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if (
        st.session_state.get("ask_plan_digest") is not None
        and st.session_state["ask_plan_digest"] != request_digest
    ):
        st.session_state.pop("ask_plan", None)
        st.session_state.pop("ask_answer", None)
        st.session_state.pop("ask_plan_digest", None)

    if plan_only or plan_and_solve:
        try:
            client = qqa.OpenAICompatibleClient(
                api_key=api_key or None,
                base_url=api_base,
                model=model_name,
                verify_ssl=not insecure,
            )
            with st.spinner("Compiling and validating the mathematical model…"):
                plan = qqa.compile_natural_language(
                    request,
                    client=client,
                    solver=workflow,
                )
            st.session_state["ask_plan"] = plan
            st.session_state["ask_plan_digest"] = request_digest
            st.session_state.pop("ask_answer", None)
            if plan_and_solve:
                with st.spinner(f"Running {plan.selected_solver}…"):
                    answer = qqa.execute_plan(
                        plan,
                        device=device,
                        seed=seed,
                        sol_size=ask_population,
                        num_epochs=ask_epochs,
                        budget=ask_budget,
                        batch_size=ask_batch,
                        workers=min(ask_batch, 8),
                        verbose=False,
                    )
                st.session_state["ask_answer"] = answer
        except Exception as exc:  # noqa: BLE001 - user-facing API/solver boundary
            st.error(str(exc))

    plan = st.session_state.get("ask_plan")
    answer = st.session_state.get("ask_answer")
    with output:
        if plan is None:
            st.info("Describe a decision and click **Plan & solve**.")
        else:
            route_a, route_b, route_c = st.columns(3)
            route_a.metric("Selected workflow", plan.selected_solver)
            route_b.metric("Decision variables", plan.variable_count)
            route_c.metric("Objectives", len(plan.spec.objectives))
            st.success(plan.rationale)
            for warning in plan.warnings:
                st.warning(warning)
            with st.expander("Review audited model", expanded=answer is None):
                st.code(plan.spec.to_json(), language="json")
                st.download_button(
                    "Download plan JSON",
                    json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
                    file_name=f"{plan.spec.name}-plan.json",
                    mime="application/json",
                )
            if answer is None:
                st.caption("The plan is validated. Click **Plan & solve** when ready.")
            elif isinstance(answer.result, qqa.ParetoResult):
                result = answer.result
                knee = result.select()
                st.success(
                    f"Found {len(result.solutions):,} nondominated plans; "
                    f"recommended compromise #{knee}."
                )
                st.dataframe(
                    result.to_frame(answer.problem),
                    hide_index=True,
                    use_container_width=True,
                )
                st.plotly_chart(
                    retheme_plotly(
                        qqa.plot_pareto(
                            result,
                            show=False,
                            title=f"{plan.spec.name} Pareto front",
                        )
                    ),
                    use_container_width=True,
                )
            elif isinstance(answer.result, qqa.BlackBoxResult):
                result = answer.result
                summary_a, summary_b, summary_c = st.columns(3)
                summary_a.metric("Best objective", f"{result.best_value:,.6g}")
                summary_b.metric("Feasible", "Yes" if result.feasible else "No")
                summary_c.metric("Evaluations", result.evaluations)
                st.json(result.best_point)
                st.plotly_chart(
                    retheme_plotly(
                        qqa.plot_blackbox(
                            result,
                            show=False,
                            title=f"{plan.spec.name} experiments",
                        )
                    ),
                    use_container_width=True,
                )
            else:
                result = answer.result
                value = result.score.get(
                    "value",
                    getattr(result, "objective_value", result.best_obj),
                )
                summary_a, summary_b, summary_c = st.columns(3)
                summary_a.metric("Objective", f"{value:,.6g}")
                summary_b.metric(
                    "Feasible",
                    "Yes" if result.score.get("feasible", True) else "No",
                )
                summary_c.metric("Runtime", f"{result.runtime:.2f} s")
                if hasattr(result, "scip_status"):
                    st.caption(
                        f"SCIP status: **{result.scip_status}** · "
                        f"gap: **{result.gap}** · dual bound: **{result.dual_bound}**"
                    )
                st.json(result.score)

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
    st.subheader("A complete trade-off surface in one parallel run")
    st.caption(
        "Each GPU replica follows a different reference direction; the live archive "
        "keeps feasible nondominated plans and recommends a scale-invariant knee."
    )
    controls, chart = st.columns([1, 3])
    with controls:
        pareto_application = st.selectbox(
            "Application",
            ("Microgrid planning", "Portfolio allocation"),
        )
        pareto_population = st.slider("Reference directions", 64, 2048, 512, 64)
        pareto_epochs = st.slider("Pareto iterations", 100, 3000, 1000, 100)
        run_pareto = st.button(
            "Generate Pareto front",
            type="primary",
            use_container_width=True,
        )
    if run_pareto:
        model = (
            qqa.build_microgrid_pareto()
            if pareto_application == "Microgrid planning"
            else qqa.build_portfolio_pareto()
        )
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
                file_name=f"{model.name}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with chart:
            front_view, diagnostics_view = st.tabs(["Trade-off surface", "Search diagnostics"])
            with front_view:
                figure = qqa.plot_pareto(
                    result,
                    show=False,
                    title=f"{model.name} Pareto front",
                )
                st.plotly_chart(retheme_plotly(figure), use_container_width=True)
            with diagnostics_view:
                diagnostics = qqa.plot_pareto_diagnostics(
                    result,
                    show=False,
                    title=f"{model.name} feasibility and archive health",
                )
                st.plotly_chart(retheme_plotly(diagnostics), use_container_width=True)
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
        spec_digest = hashlib.sha256(("json\0" + model_source).encode()).hexdigest()
        if st.button("Validate audited model", use_container_width=True):
            try:
                spec = qqa.ModelSpec.from_json(model_source)
            except (TypeError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state["universal_spec"] = spec
                st.session_state["universal_spec_digest"] = spec_digest
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
        spec_digest = hashlib.sha256(
            ("tex\0" + tex_source + "\0" + api_base + "\0" + model_name).encode()
        ).hexdigest()
        if st.button("Translate & validate TeX", use_container_width=True):
            configured_key = bool(os.getenv("QQA_LLM_API_KEY") or os.getenv("QQA_LLM_API_KEY"))
            if not api_key and not configured_key:
                st.error("Enter an API key for this in-memory request.")
            else:
                try:
                    client = qqa.OpenAICompatibleClient(
                        api_key=api_key or None,
                        base_url=api_base,
                        model=model_name,
                        verify_ssl=not insecure,
                    )
                    spec = qqa.compile_tex(tex_source, client=client)
                except Exception as exc:  # noqa: BLE001 - user-facing API boundary
                    st.error(str(exc))
                else:
                    st.session_state["universal_spec"] = spec
                    st.session_state["universal_spec_digest"] = spec_digest
                    st.success("Translation validated. Review the model below before solving.")

    if (
        st.session_state.get("universal_spec_digest") is not None
        and st.session_state["universal_spec_digest"] != spec_digest
    ):
        st.session_state.pop("universal_spec", None)
        st.session_state.pop("universal_spec_digest", None)
    spec = st.session_state.get("universal_spec")
    if spec is not None:
        st.code(spec.to_json(), language="json")
        left, right = st.columns(2)
        with left:
            if len(spec.objectives) == 1:
                scip_ready = importlib.util.find_spec("pyscipopt") is not None
                use_scip = st.toggle(
                    "SCIP proof phase",
                    value=False,
                    disabled=not scip_ready,
                    help=(
                        "QQA supplies diverse starts and SCIP refines/certifies the model. "
                        "Install qqa[scip] to enable this phase."
                    ),
                )
                if not scip_ready:
                    st.caption("SCIP is not installed; using QQA.")
            else:
                use_scip = False
                st.info("Multi-objective models use the parallel Pareto solver.")
        with right:
            solve_model = st.button(
                "Solve reviewed model", type="primary", use_container_width=True
            )
        if solve_model:
            try:
                with st.spinner("Solving reviewed model…"):
                    problem, result = _solve_reviewed_model(
                        spec,
                        use_scip=use_scip,
                        device=device,
                        seed=seed,
                    )
            except Exception as exc:  # noqa: BLE001 - user-facing solver boundary
                st.error(str(exc))
                st.stop()
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
                diagnostics = qqa.plot_pareto_diagnostics(
                    result,
                    show=False,
                    title=f"{spec.name} search diagnostics",
                )
                with st.expander("Feasibility and archive diagnostics"):
                    st.plotly_chart(retheme_plotly(diagnostics), use_container_width=True)
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
