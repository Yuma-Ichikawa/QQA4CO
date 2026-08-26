# GUI

The Streamlit dashboard exposes the same problem, solver, planning, and
visualisation APIs as Python and the CLI.

```bash
pip install "qqa[gui]"
qqa gui
# or, from a source checkout:
streamlit run app/streamlit_app.py
```

`qqa gui --port 8505 --headless` is convenient on a remote machine. The CLI
loads the app bundled in an installed wheel and falls back to `app/` in a
source or editable install.

## Pages

### Home

Pick from the built-in graph, assignment, permutation, physics, and
statistical-learning problems. Controls change with the selected family and a
domain-aware preview is rendered before solving.

The arbitrary-Python editor executes trusted local code and is therefore
disabled by default. Enable it only on a machine you control:

```bash
QQA_ALLOW_CUSTOM=1 qqa gui
```

Do not enable this flag on a public or shared deployment.

### Solve

Choose PQQA, Population Annealing, CRA-PI-GNN, or CPRA where compatible, then
set backend-specific controls. The page streams progress, loss, diversity, and
parallel-population diagnostics and renders the final solution in a
problem-aware view. Unsupported backend/problem combinations are blocked
before a run.

### Visualize

Inspect the stored run through backend-aware tabs. PQQA views include schedule,
parallel population, solution-space PCA, diversity, loss spectrogram, ridgeline,
replica fate, and family tree. Population Annealing exposes its own ESS,
free-energy, equilibrium, thermodynamic, lineage, and ancestry diagnostics
instead of empty PQQA panels.

### Compare

Run parameter sweeps and inspect them with parallel coordinates and overlaid
best-objective histories. The solver shootout compares PQQA with the available
sampling baselines under matched budgets.

### Universal

Universal Studio covers the major bounded optimisation classes supported by
QQA: binary, integer, real, mixed, constrained, multi-objective, and
black-box. Its tabs are:

| Tab | Workflow |
|---|---|
| **Ask QQA** | Natural language → audited `ModelSpec` → explained automatic route → result |
| **Mixed planning** | Practical binary/integer/real microgrid dispatch |
| **Pareto studio** | One-run cost/emissions/resilience or portfolio front |
| **Black-box lab** | Explicitly defined constrained process evaluator |
| **TeX model** | Reviewed TeX or JSON model with optional SCIP proof phase |

## Ask QQA

Describe variables, finite bounds, objectives, units, and constraints in the
text box. For example:

> Choose integer production lots in [0, 12] and real overtime in [0, 16].
> Minimize cost while meeting demand of at least 105.

Choose **Auto** unless a specific workflow is required. The model service
receives a dedicated system prompt separately from the untrusted description.
The browser never runs generated Python: the response must pass strict schema,
safe-expression, resource-quota, scalar-shape, and finite-value checks.

Two actions make the review boundary explicit:

- **Build reviewed plan** translates and validates without solving.
- **Plan & solve** additionally runs the selected local workflow.

The result panel shows the selected solver, local routing rationale, warnings,
audited JSON, and a downloadable plan before the numerical result. Multiple
objectives route to parallel Pareto QQA. Compatible single-objective models
remain on pure QQA unless **QQA + SCIP** is explicitly selected.

If the request explicitly states a safe objective formula, the black-box route
validates it and evaluates it point by point without gradients. Natural
language cannot recreate an opaque simulator, external service, or physical
experiment whose formula was not supplied. Use the Black-box lab's packaged
evaluator or bind the real callable/service adapter with
`qqa.BlackBoxProblem`; QQA must not invent the missing evaluator. Inspect model
`notes` and correct material assumptions before solving.

## Model API credentials

For local use, set credentials in the process environment before launching:

```bash
export QQA_LLM_API_KEY='…'
export QQA_LLM_BASE_URL='https://your-compatible-endpoint'
export QQA_LLM_MODEL='your-model'
qqa gui
```

QQA embeds no provider-specific endpoint or model. Never commit `.env` or
`.streamlit/secrets.toml`; both are ignored by this repository. A key entered
in the password widget is used for the translation request and is not included
in the model, downloads, reports, or logs. TLS verification is enabled by
default; the insecure private-gateway option should be limited to a trusted
development network.

On a public deployment, do not expose an unrestricted operator-funded model
key or enable arbitrary custom Python. Add authentication, request quotas, and
an endpoint allowlist appropriate to the deployment.

## Programmatic access

The reusable interfaces are framework-independent:

```python
import qqa

plan = qqa.compile_natural_language(request, solver="auto")
answer = qqa.execute_plan(plan, device="auto", seed=0)
```

For a custom dashboard, subclass `qqa.callbacks.Callback` and update your own
widgets in its lifecycle hooks. `StreamlitCallback` in
`app/pages/1_Solve.py` is an application implementation detail and should not
be imported as a library module.

See the
[natural-language optimisation Colab](https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/12_natural_language_optimization_colab.ipynb)
for the same reviewed workflow in a notebook.
