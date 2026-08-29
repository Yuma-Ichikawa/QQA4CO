# QQA4CO

[![CI](https://github.com/Yuma-Ichikawa/QQA4CO/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuma-Ichikawa/QQA4CO/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-MkDocs-blue)](https://yuma-ichikawa.github.io/QQA4CO/)
[![PyPI](https://img.shields.io/pypi/v/qqa)](https://pypi.org/project/qqa/)
[![Python](https://img.shields.io/pypi/pyversions/qqa)](https://pypi.org/project/qqa/)

<p align="center">
  <a href="https://parallelquasiquantum4co.streamlit.app/">
    <img src="https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/docs/assets/qqa-runtime-hero.png" alt="Parallel QQA replicas flowing through a sparse factor graph to verified discrete solutions" width="900">
  </a>
</p>

## Try QQA in your browser

[![Launch the live QQA Studio](https://img.shields.io/badge/Launch-Live_QQA_Studio-6C5CE7?style=for-the-badge&logo=streamlit&logoColor=white)](https://parallelquasiquantum4co.streamlit.app/)

The hosted Streamlit studio needs no local installation. Build a problem, run
QQA, inspect convergence and feasibility, and compare supported methods from a
browser. For a local studio, install `qqa[gui]` and run `qqa gui`.

QQA4CO is a GPU-first primal-search and hybrid optimisation runtime. Quasi-Quantum
Annealing (QQA) generates diverse candidates, structure-aware repair and local
search refine them, and optional mathematical solvers certify them when requested.

The default `qqa.solve(...)` route is pure QQA. Exact solvers, GNNs, dashboard
dependencies, and public benchmark parsers are explicit extras.

## Install

Core CPU/GPU solver:

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade qqa
```

Common optional installations:

```bash
python -m pip install --upgrade "qqa[gui]"       # Streamlit studio
python -m pip install --upgrade "qqa[benchmark]" # MIPLIB/QPLIB + SCIP
python -m pip install --upgrade "qqa[highs]"     # HiGHS LP/MIP adapter
python -m pip install --upgrade "qqa[cpsat]"     # OR-Tools CP-SAT adapter
python -m pip install --upgrade "qqa[service]"   # schema-only FastAPI job service
python -m pip install --upgrade "qqa[pignn]"     # experimental CRA/CPRA GNN
python -m pip install --upgrade "qqa[triton]"    # optional fused CUDA kernels
python -m pip install --upgrade "qqa[dev]"       # tests, lint, typing, docs
```

QQA4CO supports CPython 3.10--3.14. PyPy is not currently supported because the
solver depends on PyTorch, whose official distributions target CPython.

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
    goal="best",
    budget="30s",
    device="auto",
    seed=0,
)

print(result.status.value)
print(result.best_obj)  # original mathematical objective
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

The one-call goal can be `best`, `feasible`, `prove`, `diverse`, or `pareto`.
Durations accept `ms`, `s`, `m`, and `h`. `prove` enables an exact route but
still returns a proof status only when the backend certifies the original
model.

## Inspect and plan before solving

```python
inspection = qqa.inspect(problem)
print(inspection.to_dict())

doctor = qqa.doctor(problem, replicas=128)
print(doctor.explain())

plan = qqa.plan(problem, profile="quality", device="cuda")
print(plan.explain())
```

The planner reports domains, sparse factors, connected components, selected
engine, refinements, certification route, VRAM estimate, replica count, and the
reason for each choice. It does not execute the solver.

The Model Doctor additionally checks bounds, factor capabilities, scaling,
curvature, presolve contradictions, decomposition, and route/proof support.
Missing real or integer bounds are never silently replaced by a guessed box.

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

## Checkpoint, events, and portable packages

Long QQA runs can be resumed without pickle payloads:

```python
qqa.solve(
    problem,
    budget="10m",
    checkpoint_path="run.qqacp",
    checkpoint_interval=100,
)
result = qqa.solve(problem, budget="20m", resume_from="run.qqacp")
```

The atomic checkpoint verifies model/config fingerprints, tensor checksums,
optimizer and RNG state, schedule state, incumbent, population, and historical
archive. Paths stay outside result provenance. `SolveResult.events` uses the
versioned schema-v2 stream, and `qqa.runtime.export_result_package` writes a
checksum-protected, environment-neutral exchange bundle.

Python source and pickle inputs are trusted-local features and are denied by
default. Use `trusted=True` in Python, `--allow-unsafe-python` in the CLI, or
`QQA_ALLOW_CUSTOM=1` for a single-user local GUI. Do not enable custom code in
a shared deployment.

The Colab-ready
[`13_typed_primal_dual_runtime.ipynb`](examples/13_typed_primal_dual_runtime.ipynb)
walks through diagnosis, solve events, the cockpit, checkpoint/resume, and a
verified package without credentials or environment-specific paths.
[`14_factor_split_qqa_study.ipynb`](examples/14_factor_split_qqa_study.ipynb)
adds factor backend inspection, guarantee-aware results, a QQA Study/Trial
campaign, and paired Benchmark Hub statistics.

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

print(result.raw_solution)  # untouched optimiser output
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

The evaluation database keys every observation by problem, point, seed,
fidelity, replicate, and evaluator version, so noisy repeats are not silently
overwritten. It records pending/running/completed/failed/timed-out/cancelled
states. A hard per-evaluation timeout uses an isolated process.

For resumable campaigns, `Study` keeps QQA as the default diverse-batch
acquisition engine:

```python
study = qqa.create_study(problem, storage="observations.sqlite", seed=0)
result = study.optimize(budget=100, batch_size=4)
print(study.best_trial.point, study.best_trial.value)
```

The execution contract is inspectable with
`qqa.model.compile_execution_plan(model_ir)`. Capability claims come from
registered eager, fused GPU, and exact factor backends. `SolveResult.status`,
`SolveResult.guarantee_level`, and tri-state feasibility are separate fields.

## Feature status

| Feature | Status |
| --- | --- |
| Sparse factor QQA and local search | Stable |
| Device-resident telemetry and restart control | Stable |
| Softmax/Gumbel, sparsemax/entmax and mirror-descent relaxations | Experimental, opt-in |
| Persistent torch.export/AOTInductor sparse-model cache | Experimental, opt-in |
| Dense QUBO compatibility view | Deprecated for large models |
| Stable `solve` / `plan` / `inspect` contract | Stable |
| Factor-split execution plan and backend registry | Beta |
| Mixed-variable QQA and ModelIR presolve | Beta |
| MPS, QPLIB, OPB, DIMACS, QUBO, Ising inputs | Beta |
| SCIP hybrid and exact certification | Beta, optional |
| RENS/RINS/GINS/local-branching/trust-region portfolio | Beta, optional |
| HiGHS and CP-SAT adapters | Beta, optional |
| Black-box trust-region solver | Beta |
| QQA Study/Trial orchestration and Benchmark Hub schema | Beta |
| Multi-objective and uncertainty extensions | Beta |
| Multi-device replica islands | Experimental, opt-in |
| CRA/CPRA graph GNN backend | Experimental, opt-in |
| Natural-language / TeX model compilation | Experimental |
| cuOpt bridge | Experimental, version-dependent |

## Documentation

- [Quickstart](https://yuma-ichikawa.github.io/QQA4CO/quickstart/)
- [Architecture](https://yuma-ichikawa.github.io/QQA4CO/explanation/architecture/)
- [Advanced opt-in runtime](https://yuma-ichikawa.github.io/QQA4CO/how-to/advanced-runtime/)
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
python -m pip install --upgrade pip
python -m pip install --upgrade -e ".[dev]"
ruff check src tests scripts app
ruff format --check src tests scripts app
pytest -q
mkdocs build --clean --strict
```

Large datasets and generated campaign trajectories are not source files. Keep only
tiny smoke instances, checksums, licenses, and compact summaries in the repository;
use the documented fetch commands and release/CI artifacts for full data.

## Research papers and BibTeX

QQA4CO builds on three peer-reviewed continuous-relaxation methods. The default
solver is pure QQA; the CRA/CPRA-inspired graph-learning route is an explicit
`pignn` extra.

### QQA / PQQA — ICLR 2025

[Paper (OpenReview)](https://openreview.net/forum?id=9EfBeXaXf0) ·
[arXiv](https://arxiv.org/abs/2409.02135)

```bibtex
@inproceedings{ichikawa2025optimization,
  title={Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling},
  author={Ichikawa, Yuma and Arai, Yamato},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=9EfBeXaXf0}
}
```

### CRA — NeurIPS 2024

[Paper (NeurIPS Proceedings)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54191f424e9013fc1d7b923f6e45dff4-Abstract-Conference.html) ·
[OpenReview](https://openreview.net/forum?id=ykACV1IhJD) ·
[Reference implementation](https://github.com/Yuma-Ichikawa/CRA4CO)

```bibtex
@inproceedings{NEURIPS2024_54191f42,
  author={Ichikawa, Yuma},
  booktitle={Advances in Neural Information Processing Systems},
  editor={A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages={47189--47216},
  publisher={Curran Associates, Inc.},
  title={Controlling Continuous Relaxation for Combinatorial Optimization},
  url={https://proceedings.neurips.cc/paper_files/paper/2024/file/54191f424e9013fc1d7b923f6e45dff4-Paper-Conference.pdf},
  volume={37},
  year={2024},
  doi={10.52202/079017-1495}
}
```

### CPRA — TMLR 2025

[Paper (OpenReview / TMLR)](https://openreview.net/forum?id=ix33zd5zCw) ·
[Reference implementation](https://github.com/Yuma-Ichikawa/CPRA4CO)

```bibtex
@article{ichikawa2025continuous,
  title={Continuous Parallel Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems},
  author={Ichikawa, Yuma and Iwashita, Hiroaki},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=ix33zd5zCw}
}
```

## License

QQA4CO is distributed under the BSD 3-Clause License.
