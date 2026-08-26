# Universal optimisation

QQA exposes one coherent surface for the major bounded optimisation classes:
single or multiple objectives; binary, integer, real, and mixed variables;
differentiable or black-box functions; and constrained or unconstrained
models. Single-objective QUBO and safe mixed nonlinear models can additionally
be handed to SCIP for exact refinement and certification.

## One description, one reviewed plan

The Python API, CLI, and Streamlit app expose the same ordinary-language
workflow:

```bash
export QQA_LLM_API_KEY='your-key'
export QQA_LLM_BASE_URL='https://api.example.com'
export QQA_LLM_MODEL='your-model-id'

qqa ask \
  "Choose integer units in [0,20] and real overtime in [0,8]. \
   Minimize 3*units + square(overtime), with 4*units + overtime >= 45." \
  --solver auto --device auto
```

```python
import qqa

request = """
Choose binary open decisions, integer production in [0, 20], and real reserve
in [0, 10]. Minimize operating cost while satisfying demand >= 100.
"""

# One-call translation, routing, and execution.
answer = qqa.ask(request, solver="auto", device="auto", seed=0)
print(answer.plan.selected_solver)
print(answer.plan.rationale)

# Or stop at the review boundary.
plan = qqa.compile_natural_language(request, solver="auto")
print(plan.spec.to_json())
reviewed_answer = qqa.execute_plan(plan, device="auto", seed=0)
```

`qqa.MODEL_SYSTEM_PROMPT` is sent separately from the user's text. It defines
the allowed declarative grammar, treats input as untrusted data, forbids
generated executable code, and preserves multiple objectives instead of
inventing a weighted sum. The returned JSON is still only a proposal: local
validation checks exact fields, variable domains and bounds, expression syntax,
resource quotas, scalar output shape, and finite values before a solver can run.
Review `plan.spec.notes` and the full model when the request leaves any business
assumption open.

With `solver="auto"`, trusted local code—not the LLM—selects the route:

| Validated request | Route |
|---|---|
| One bounded symbolic objective | QQA, or QQA→SCIP when the compatible optional backend is available |
| Two or more symbolic objectives | One-run parallel Pareto QQA |
| Explicit safe formula to evaluate without gradients | Pointwise `BlackBoxProblem` adapter and budget-aware black-box optimiser |
| Objective available only through a simulator/API/experiment | Bind that real evaluator explicitly with `BlackBoxProblem` |

This unified surface covers all major bounded classes supported by QQA:
binary, integer, real, mixed, constrained, multi-objective, and black-box.
For the natural-language black-box route, the user must state a safe objective
formula. QQA validates it and adapts it to pointwise, no-gradient evaluation;
the optimiser then treats the values as opaque. This does not mean a language
model can recreate a simulator, service, laboratory experiment, or proprietary
scoring function from prose. When no formula is supplied, QQA must not invent
one: bind the actual callable or service adapter explicitly with
`BlackBoxProblem`. Review `plan.spec.notes` and stop at plan review whenever an
assumption changes the intended mathematics.

The dashboard exposes this workflow in **Universal → Ask QQA**. It displays the
selected route and rationale, keeps the audited JSON downloadable, and offers a
separate **Build reviewed plan** action before **Plan & solve**.

## QQA × SCIP

Install the optional exact backend:

```bash
pip install "qqa[scip]"
```

```python
import networkx as nx
import qqa
from qqa.hybrid import solve_qqa_scip

problem = qqa.MaxCut(nx.random_regular_graph(3, 100, seed=0))
result = solve_qqa_scip(
    problem,
    qqa_kwargs={"sol_size": 512, "num_epochs": 2000, "device": "cuda"},
    time_limit=120,
    max_warm_starts=64,
)
print(result.best_obj, result.scip_status, result.gap, result.dual_bound)
```

QQA first searches many basins in parallel. The projected population is
deduplicated, ranked, and installed as multiple SCIP primal starts. SCIP then
solves the identical binary quadratic objective
\(x^\mathsf{T}Qx\), can improve the incumbent, and returns a dual bound and
relative gap. The final objective is therefore never worse than the QQA
incumbent. `proven_optimal` is true only when SCIP reports `optimal`.

`solve_qqa_scip` accepts `QUBOProblem` models. Safe TeX/JSON models have a
second exact route, `solve_spec_scip`, which preserves binary, bounded integer,
bounded real, nonlinear objective, and nonlinear constraints:

```python
from pathlib import Path
from qqa.hybrid import solve_spec_scip

spec = qqa.ModelSpec.from_json(Path("audited-model.json").read_text())
result = solve_spec_scip(
    spec,
    qqa_kwargs={"sol_size": 512, "num_epochs": 1500, "device": "cuda"},
    time_limit=120,
)
print(result.objective_value, result.scip_status, result.gap)
```

The safe arithmetic grammar compiles directly to PySCIPOpt expressions; no
generated Python is evaluated. Multiple QQA population members are installed
as primal starts before SCIP's proof phase.

## One-run parallel Pareto front

```python
model = qqa.MultiObjectiveProblem(
    [
        qqa.Binary("open", size=4),
        qqa.Integer("production", 0, 20, size=4),
        qqa.Real("reserve", 0, 10),
    ],
    [
        qqa.Objective(cost, "cost", direction="min", unit="kUSD"),
        qqa.Objective(resilience, "resilience", direction="max"),
        qqa.Objective(emissions, "CO2", direction="min", unit="t"),
    ],
    constraints=[demand, linking],
)
result = model.solve_pareto(sol_size=1024, device="cuda")
figure = qqa.plot_pareto(result)  # 2-D, 3-D, or parallel coordinates
diagnostics = qqa.plot_pareto_diagnostics(result)
```

Every replica receives a different low-discrepancy reference direction.
Augmented Tchebycheff scalarisation can reach non-convex front regions that a
weighted sum can miss. Feasible nondominated projections are accumulated
throughout the run, deduplicated, and thinned by crowding distance only when
the archive limit is reached.

`result.objectives` and `result.solutions` have aligned rows.
`result.named_solutions(model)` restores the variable names, and
`result.to_frame(problem)` includes objectives and named decision variables.
`result.select()` returns a scale-invariant knee, weighted selection is
available with `result.select([w1, ...])`, and two-objective fronts expose
exact `result.hypervolume(reference_point)`.

Dominance comparisons are chunked, avoiding a full quadratic temporary tensor
on large GPU archives. Feasibility uses the projected
Powell–Hestenes–Rockafellar augmented Lagrangian: equality multipliers are
signed, inequality multipliers are non-negative, every reference-direction
replica owns its KKT multiplier vector, and penalty growth measures only
residual beyond each declared tolerance. Stagnating weak replicas are
split between archive-centred and global restarts while objective-axis anchors
and the nondominated archive are retained.

`plot_pareto` highlights the scale-invariant knee and labels each objective's
direction. `plot_pareto_diagnostics` exposes archive growth, feasible
population share, normalised violation, penalty \(\rho\), and restart epochs.

A ready-to-run cardinality-constrained portfolio demonstrates a genuinely
mixed, non-convex three-objective surface:

```python
portfolio = qqa.build_portfolio_pareto()
front = portfolio.solve_pareto(sol_size=1024, device="auto")
index = front.select(weights=[0.45, 0.35, 0.20])
print(portfolio.score_summary(front.solutions[index]))
qqa.plot_pareto(front)
```

## Mixed-variable black-box optimisation

Use this path when gradients are unavailable or each evaluation launches a
simulation, service request, or physical experiment:

```python
problem = qqa.BlackBoxProblem(
    [
        qqa.Binary("feature", size=5),
        qqa.Integer("workers", 1, 32),
        qqa.Real("threshold", 0.0, 1.0),
    ],
    expensive_objective,  # receives one named Python dict
    constraints=[qqa.BlackBoxConstraint(memory_use, sense="<=", rhs=64, name="memory")],
)
result = problem.solve(
    budget=200,
    initial_points=24,
    batch_size=8,
    workers=8,
    device="cuda",
)
qqa.plot_blackbox(result)
```

The optimiser combines a numerically regularised RBF surrogate, expected
improvement (or lower confidence bounds), a separate log-violation surrogate
for every constraint, joint probability of feasibility, global Sobol coverage,
local trust-region search, adaptive expansion/shrinkage, duplicate
suppression, and greedy batch diversification. Multi-output constraints share
one Cholesky factorisation. `budget` counts actual objective calls. `workers`
evaluates independent batch points concurrently; surrogate linear algebra and
candidate scoring can run on CUDA. Use `resume_from=previous_result` to extend
a campaign without repeating expensive evaluations. `max_model_points` bounds
cubic kernel cost on long runs.

## TeX to a safe solver model

Credentials are read only from the process environment:

```bash
export QQA_LLM_API_KEY='your-key'
export QQA_LLM_BASE_URL='https://your-openai-compatible-gateway'
export QQA_LLM_MODEL='your-model'

qqa tex --file regional-production.tex \
  --solver auto --device auto \
  --output-model audited-model.json \
  --output-result solution.json --report result.html
```

Use `--dry-run` to stop after translation and local validation. A reviewed
model can later be solved without an API call or key:

```bash
qqa tex --spec audited-model.json --device cuda
```

`--solver auto` keeps single-objective models on pure QQA regardless of which
optional packages happen to be installed. Use `--solver scip` to opt in to the
QQA→SCIP proof path. Multi-objective models automatically use the one-run
Pareto solver. `--show-model` prints the reviewed intermediate representation.

The translator requests a JSON schema through the Responses API and falls
back to prompt-enforced JSON for compatible gateways without Structured
Outputs. Messages-style endpoints are available with
`--api-style messages`. `--insecure` is explicit and should only be used for
a trusted private development gateway with a non-standard certificate.

LLM output is never executed. QQA validates exact keys, domains, bounds,
directions, constraint scales and finite constants, then interprets a small
arithmetic AST. Imports, attributes, comprehensions, strings, indirect calls,
and unknown functions are rejected. No credential is written into model JSON,
reports, notebooks, or error messages.
QQA embeds no provider-specific endpoint or model. Configure
`QQA_LLM_API_KEY`, `QQA_LLM_BASE_URL`, and `QQA_LLM_MODEL` in the environment
or pass the non-secret endpoint and model through the corresponding API/CLI
options.

## Reproducible examples

The packaged examples run without copying model code:

```bash
qqa example run microgrid-dispatch --output-dir results/dispatch
qqa example run microgrid-pareto --device cuda --output-dir results/pareto
qqa example run portfolio-pareto --device cuda --output-dir results/portfolio
qqa example run process-blackbox --device cuda --output-dir results/process
```

The
[universal optimisation Colab notebook](https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/10_universal_optimization_colab.ipynb)
covers the API surface. The
[real-world optimisation studio](https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/11_real_world_optimization_studio.ipynb)
uses microgrid, portfolio, process, and production-planning models end to end.
The
[natural-language optimisation notebook](https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/12_natural_language_optimization_colab.ipynb)
demonstrates the system-prompt boundary, reviewed plans, automatic routes,
`qqa.ask`, `qqa ask`, and the Ask QQA tab.
Keep large validation outputs outside the package tree; the repository's own
GPU and live-API validation is performed under `works/`.
