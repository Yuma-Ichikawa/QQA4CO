# QQA4CO

[![CI](https://github.com/Yuma-Ichikawa/QQA4CO/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuma-Ichikawa/QQA4CO/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-MkDocs-blue)](https://yuma-ichikawa.github.io/QQA4CO/)
[![PyPI](https://img.shields.io/pypi/v/qqa)](https://pypi.org/project/qqa/)
[![Python](https://img.shields.io/pypi/pyversions/qqa)](https://pypi.org/project/qqa/)

QQA4CO is a GPU-first primal-search and hybrid optimisation runtime. Quasi-Quantum
Annealing (QQA) generates diverse candidates, structure-aware repair and local
search refine them, and optional mathematical solvers certify them when requested.

The default `qqa.solve(...)` route is pure QQA. Exact solvers, GNNs, dashboard
dependencies, and public benchmark parsers are explicit extras.

## Install

Core CPU/GPU solver:

```bash
pip install qqa
```

Common optional installations:

```bash
pip install "qqa[gui]"                 # Streamlit studio
pip install "qqa[benchmark]"           # MIPLIB/QPLIB + SCIP
pip install "qqa[highs]"               # HiGHS LP/MIP adapter
pip install "qqa[cpsat]"               # OR-Tools CP-SAT adapter
pip install "qqa[pignn]"               # experimental CRA/CPRA GNN backends
pip install "qqa[dev]"                 # tests, lint, typing, docs tooling
```

PyTorch chooses CPU or CUDA according to the installed wheel. Follow the
[GPU setup guide](https://yuma-ichikawa.github.io/QQA4CO/how-to/gpu/) when a
specific CUDA build is required.

## Quickstart

```python
import networkx as nx
import qqa

graph = nx.random_regular_graph(d=3, n=100, seed=0)
problem = qqa.MaximumIndependentSet(graph, penalty=2.0)

result = qqa.solve(
    problem,
    profile="balanced",
    device="auto",
    seed=0,
)

print(result.status.value)
print(result.best_obj)      # original mathematical objective
print(result.feasible)
print(result.runtime)
```

The high-level result does not hide transformations:

```python
print(result.objective_value)
print(result.internal_energy)
print(result.merit_value)
print(result.raw_solution)
print(result.repaired_solution)
print(result.violations.maximum_violation)
print(result.plan.explain())
```

Legacy research calls such as `qqa.anneal`, `qqa.simulated_annealing`,
`qqa.population_annealing`, and `qqa.discrete_langevin` remain available.
New integrations should prefer `solve`, `plan`, and `inspect`.

## Inspect and plan before solving

```python
inspection = qqa.inspect(problem)
print(inspection.to_dict())

plan = qqa.plan(problem, profile="quality", device="cuda")
print(plan.explain())
```

The planner reports domains, sparse factors, connected components, selected
engine, refinements, certification route, VRAM estimate, replica count, and the
reason for each choice. It does not execute the solver.

## Profiles

```python
qqa.solve(problem, profile="fast")
qqa.solve(problem, profile="balanced")
qqa.solve(problem, profile="quality")
qqa.solve(problem, profile="reproducible", seed=7)
```

Additional profiles:

| Profile | Intended use |
| --- | --- |
| `certify` | QQA incumbent generation followed by an optional exact backend |
| `diverse` | retain a larger candidate population |
| `pareto` | multi-objective workflows |

Advanced configuration is strict. Unknown or misspelled options raise an error:

```python
config = qqa.SolverConfig.for_profile(
    "quality",
    replicas=512,
    epochs=4000,
    schedule="reheat",
    compile_core=True,
    exact_backend="none",
)
result = qqa.solve(problem, config=config)
```

## Model inputs

The common factor-based `ModelIR` represents binary, spin, integer, real,
categorical, and permutation blocks. Native factors include:

- linear, sparse quadratic, and higher-order terms;
- clauses, cardinality, all-different, assignment, Potts, and table terms;
- indicators, SOS1/SOS2, piecewise-linear and logical relations;
- precedence, no-overlap, cumulative resource, flow, matching, and subtour terms;
- scenario mean/worst-case/CVaR and chance-constraint aggregation.

Supported file routes:

| Format | Route |
| --- | --- |
| MPS / LP | sparse algebraic model; optional exact completion |
| QPLIB | official parser extra, sparse linear/quadratic model |
| JSON ModelIR | dependency-light canonical model |
| OPB | linear pseudo-Boolean model |
| DIMACS CNF / WCNF | native clause factors |
| QUBO text | sparse binary quadratic model |
| Ising edge list | sparse spin model |

```bash
qqa inspect model.mps
qqa plan model.mps --profile quality --device auto
qqa solve model.mps --profile balanced --device auto
```

## MIPLIB and QPLIB

Install the benchmark extra once:

```bash
pip install "qqa[benchmark]"
```

Fetch and inspect one public instance:

```bash
qqa benchmark fetch miplib --instance air05 --output benchmarks/miplib
qqa benchmark inspect benchmarks/miplib/air05.mps.gz

qqa benchmark fetch qplib --instance QPLIB_0031 --output benchmarks/qplib
qqa benchmark inspect benchmarks/qplib/QPLIB_0031.qplib
```

Run QQA-centred SG-CQQA or a paired baseline comparison:

```bash
qqa benchmark run INSTANCE --solver sg-cqqa --time-limit 60 --device auto

qqa benchmark compare INSTANCE \
  --solvers scip-aggressive sg-cqqa \
  --seeds 0 1 2 \
  --time-limit 60 \
  --output results.json
```

The campaign runner uses equal wall-clock budgets, paired seeds, checkpointing,
direction-aware objectives, feasibility, bounds, gaps, time to first feasible,
and primal-integral metrics. See the
[MIPLIB/QPLIB guide](https://yuma-ichikawa.github.io/QQA4CO/miplib-qplib/)
and the transparent [published result summary](https://yuma-ichikawa.github.io/QQA4CO/benchmark-results/).

## Optional exact completion

Pure QQA remains the default:

```python
result = qqa.solve(model, profile="balanced", exact_backend="none")
```

Certification is explicit:

```python
result = qqa.solve(model, profile="certify", exact_backend="scip", budget=60)
print(result.best_bound, result.relative_gap, result.proven_optimal)
```

SCIP supports the broadest mixed nonlinear route. HiGHS is available for sparse
linear LP/MIP models, and CP-SAT for bounded integral linear models with integral
coefficients. Unsupported semantics fail clearly; adapters never silently round,
drop, or reinterpret constraints.

## Structured optimisation

```python
problem = qqa.TSP(N=30, seed=0)  # Sinkhorn permutation relaxation
result = qqa.solve(problem, profile="quality")

print(result.raw_solution)       # untouched optimiser output
print(result.repaired_solution)  # Hungarian projection + 2-opt, when changed
```

Mixed-variable models use scaled constraints, per-constraint augmented Lagrangian
multipliers, feasibility-first archives, and explicit repair. Sparse graph QUBOs
use edge factors and incremental one-flip local search instead of an `N × N`
matrix in the hot path.

## Black-box optimisation

```python
problem = qqa.BlackBoxProblem(
    [qqa.Real("x", -2.0, 2.0), qqa.Integer("n", 0, 8)],
    lambda point: (point["x"] - 0.4) ** 2 + (point["n"] - 3) ** 2,
)

result = problem.solve(
    budget=100,
    workers=4,
    trust_regions=3,
    evaluation_database="evaluations.sqlite",
)
```

The optional evaluation database caches encoded points and records
pending/running/completed/failed/timed-out/cancelled states. Large campaigns can
select random-Fourier-feature surrogates; discrete batch acquisition can opt into
QQA with `acquisition_optimizer="qqa"`.

## Feature status

| Feature | Status |
| --- | --- |
| Sparse factor QQA and local search | Stable |
| Dense QUBO compatibility view | Deprecated for large models |
| Stable `solve` / `plan` / `inspect` contract | Stable |
| Mixed-variable QQA and ModelIR presolve | Beta |
| MPS, QPLIB, OPB, DIMACS, QUBO, Ising inputs | Beta |
| SCIP hybrid and exact certification | Beta, optional |
| HiGHS and CP-SAT adapters | Beta, optional |
| Black-box trust-region solver | Beta |
| Multi-objective and uncertainty extensions | Beta |
| Multi-device replica islands | Experimental, opt-in |
| CRA/CPRA graph GNN backend | Experimental, opt-in |
| Natural-language / TeX model compilation | Experimental |
| cuOpt bridge | Experimental, version-dependent |

## Documentation

- [Quickstart](https://yuma-ichikawa.github.io/QQA4CO/quickstart/)
- [Architecture](https://yuma-ichikawa.github.io/QQA4CO/explanation/architecture/)
- [Algorithm](https://yuma-ichikawa.github.io/QQA4CO/explanation/algorithm/)
- [Problem catalog](https://yuma-ichikawa.github.io/QQA4CO/problems/)
- [Mixed optimisation](https://yuma-ichikawa.github.io/QQA4CO/mixed-optimization/)
- [Benchmark protocol](https://yuma-ichikawa.github.io/QQA4CO/how-to/benchmark/)
- [CLI reference](https://yuma-ichikawa.github.io/QQA4CO/reference/cli/)
- [Python API](https://yuma-ichikawa.github.io/QQA4CO/api/)
- [Migration guide](https://yuma-ichikawa.github.io/QQA4CO/migration/)

## Development

```bash
git clone https://github.com/Yuma-Ichikawa/QQA4CO.git
cd QQA4CO
pip install -e ".[dev]"
ruff check src tests scripts app
ruff format --check src tests scripts app
pytest -q
mkdocs build --clean --strict
```

Large datasets and generated campaign trajectories are not source files. Keep only
tiny smoke instances, checksums, licenses, and compact summaries in the repository;
use the documented fetch commands and release/CI artifacts for full data.

## Citation

```bibtex
@inproceedings{ichikawa2025optimization,
  title={Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling},
  author={Ichikawa, Yuma and Arai, Yamato},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=9EfBeXaXf0}
}
```

QQA4CO is distributed under the BSD 3-Clause License.
