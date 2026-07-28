# Universal optimisation

QQA exposes one coherent surface for the major bounded optimisation classes:
single or multiple objectives; binary, integer, real, and mixed variables;
differentiable or black-box functions; and constrained or unconstrained
models. QUBOs can additionally be handed to SCIP for exact refinement and
certification.

## QQA × SCIP

Install the optional exact backend:

```bash
pip install "qqa[scip]"
```

```python
import networkx as nx
import qqa

problem = qqa.MaxCut(nx.random_regular_graph(3, 100, seed=0))
result = qqa.solve_qqa_scip(
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

SCIP refinement currently accepts `QUBOProblem` models. General
`MixedProblem` models remain available to QQA, including nonlinear
differentiable objectives and constraints.

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
```

Every replica receives a different low-discrepancy reference direction.
Augmented Tchebycheff scalarisation can reach non-convex front regions that a
weighted sum can miss. Feasible nondominated projections are accumulated
throughout the run, deduplicated, and thinned by crowding distance only when
the archive limit is reached.

`result.objectives` and `result.solutions` have aligned rows.
`result.named_solutions(model)` restores the variable names, and
`result.to_frame()` creates an export-ready table.

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

The optimiser combines a numerically regularised RBF surrogate, uncertainty
aware lower confidence bounds, global Sobol coverage, local trust-region
search, adaptive expansion/shrinkage, constraint surrogates, duplicate
suppression, and greedy batch diversification. `budget` counts actual
objective calls. `workers` evaluates independent batch points concurrently;
surrogate linear algebra can run on CUDA.

## TeX to a safe solver model

Credentials are read only from the process environment:

```bash
export QQA_LLM_API_KEY='your-key'
export QQA_LLM_BASE_URL='https://your-openai-compatible-gateway'
export QQA_LLM_MODEL='your-model'

qqa tex '\min_{x\in[-5,5],\,n\in\{0,\ldots,6\}} (x-2)^2+(n-3)^2' \
  --device cuda --output-model audited-model.json --report result.html
```

Use `--dry-run` to stop after translation and local validation. A reviewed
model can later be solved without an API call or key:

```bash
qqa tex --spec audited-model.json --device cuda
```

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

## Reproducible examples

The
[universal optimisation Colab notebook](https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/10_universal_optimization_colab.ipynb)
covers all four workflows. Keep large validation outputs outside the package
tree; the repository's own GPU and live-API validation is performed under
`works/`.
