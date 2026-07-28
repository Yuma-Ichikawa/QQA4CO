# Universal optimisation

QQA exposes one coherent surface for the major bounded optimisation classes:
single or multiple objectives; binary, integer, real, and mixed variables;
differentiable or black-box functions; and constrained or unconstrained
models. Single-objective QUBO and safe mixed nonlinear models can additionally
be handed to SCIP for exact refinement and certification.

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

`solve_qqa_scip` accepts `QUBOProblem` models. Safe TeX/JSON models have a
second exact route, `solve_spec_scip`, which preserves binary, bounded integer,
bounded real, nonlinear objective, and nonlinear constraints:

```python
from pathlib import Path

spec = qqa.ModelSpec.from_json(Path("audited-model.json").read_text())
result = qqa.solve_spec_scip(
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
on large GPU archives. Feasibility uses an adaptive augmented Lagrangian, and
stagnating weak replicas are restarted while the nondominated archive is
retained.

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
improvement (or lower confidence bounds), probability of feasibility, global
Sobol coverage, local trust-region search, adaptive expansion/shrinkage,
duplicate suppression, and greedy batch diversification. `budget` counts
actual objective calls. `workers` evaluates independent batch points
concurrently; surrogate linear algebra and candidate scoring can run on CUDA.
Use `resume_from=previous_result` to extend a campaign without repeating
expensive evaluations. `max_model_points` bounds cubic kernel cost on long
runs.

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

`--solver auto` selects the QQA→SCIP proof path for a single-objective model
when `qqa[scip]` is installed, and otherwise uses QQA. `--solver qqa` is
always available; multi-objective models automatically use the one-run Pareto
solver. `--show-model` prints the reviewed intermediate representation.

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

The packaged examples run without copying model code:

```bash
qqa example run microgrid-dispatch --output-dir results/dispatch
qqa example run microgrid-pareto --device cuda --output-dir results/pareto
qqa example run process-blackbox --device cuda --output-dir results/process
```

The
[universal optimisation Colab notebook](https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/10_universal_optimization_colab.ipynb)
covers the API surface. The
[real-world optimisation studio](https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/11_real_world_optimization_studio.ipynb)
uses the microgrid, process, and production-planning models end to end. Keep
large validation outputs outside the package tree; the repository's own GPU
and live-API validation is performed under `works/`.
