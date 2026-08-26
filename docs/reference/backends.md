# Backend reference

QQA4CO ships a core gradient solver, optional GNN variants, an exact SCIP
hybrid, and sampling baselines. The high-level optimisation layer additionally
provides dedicated Pareto and black-box solvers.

The stable `qqa.solve` adapter returns `SolveResult` for QQA, SA, population
annealing, iSCO, and supported exact routes. Backend-native result classes
remain available from their low-level functions for compatibility.

## At a glance

| | **PQQA** | **QQA→SCIP** | **SG-CQQA** | **CRA-PI-GNN** | **CPRA** |
|---|---|---|---|---|---|
| **Module** | `qqa.anneal` | `qqa.hybrid.solve_qqa_scip` | `qqa.benchmarking.run_miplib` / `run_qplib` | `qqa.pignn.train_cra_pi_gnn` | `qqa.pignn.train_cpra_pi_gnn` |
| **CLI flag** | `--backend qqa` | `--backend scip` | `benchmark run --solver sg-cqqa` | `--backend pignn` | `--backend cpra` |
| **Install** | `pip install qqa` | `pip install "qqa[scip]"` | `pip install "qqa[benchmark]"` | `pip install "qqa[pignn]"` | `pip install "qqa[pignn]"` |
| **Low-level return** | `AnnealResult` | `SCIPHybridResult` | `BenchmarkResult` | `AnnealResult` | `AnnealResult` |
| **Variables** | binary, integer, real, mixed, spin, categorical | binary QUBO | sparse MIP/QP/QCQP | graph QUBO | graph QUBO |
| **Role** | massively parallel heuristic | one-shot warm start and certify | iterative SCIP primal heuristic | graph inductive bias | diverse GNN heads |
| **GPU** | CUDA, MPS | QQA on GPU; SCIP on CPU | QQA on configured Torch device; SCIP on CPU | CUDA | CUDA |

## When to use which

* **Default — PQQA.** The cheapest, most thoroughly tested, and works
  on every problem in the catalogue.
* **QQA→SCIP** when the model is a QUBO and a proof, dual bound, or target
  optimality gap matters. QQA supplies multiple incumbents; SCIP is never
  allowed to worsen the returned solution.
* **SG-CQQA** for MIPLIB/QPLIB files. SCIP repeatedly selects a small
  node-local integer core, tries a cheap original-objective surrogate move,
  and invokes QQA only when the fast path does not improve and enough time
  remains. Candidates first use an in-place SCIP dive and may then use a
  bounded sub-SCIP repair.
  See the [MIPLIB/QPLIB guide](../miplib-qplib.md).
* **CRA-PI-GNN** when you specifically want the GNN inductive bias
  (smoothness over the graph) on large sparse graph problems and you
  can afford a long training run.
* **CPRA** when you need *diverse* solutions (penalty portfolio, mode
  coverage). Returns R solutions in one training run.

Optional exact adapters are selected only through an explicit certificate
profile or `exact_backend` setting:

| Adapter | Intended model | Install | Guarantee |
| --- | --- | --- | --- |
| SCIP | LP/MIP/QP/QCQP and QUBO hybrid | `qqa[scip]` | backend-dependent primal/dual result |
| HiGHS | sparse linear LP/MIP | `qqa[highs]` | backend status and bound when available |
| CP-SAT | bounded integral linear model | `qqa[cpsat]` | integer-feasible result and proof status |
| cuOpt | reserved optional capability probe | vendor package | fails explicitly when the installed API is unsupported |

## Knob translation between backends

The same intuition appears under different names:

| Concept | `qqa.anneal` | `train_cra_pi_gnn` | `train_cpra_pi_gnn` |
|---|---|---|---|
| Schedule start | `min_bg` (default `-2.0`) | `init_reg_param` (default `-20`) | same |
| Schedule slope | derived from `(min_bg, max_bg, num_epochs)` | `annealing_rate` (default `1e-3`) | same |
| Penalty exponent | `curve_rate` (default 2) | `curve_rate` (default 2) | same |
| Population / replicas | `sol_size` (B) | n/a (single) | `num_replicas` (R) |
| Diversity weight | `div_param` | n/a | `vari_param` |
| Per-replica penalty | n/a | n/a | `replica_problems=[...]` |
| Basin recovery / early stop | `restart_patience` | `tol`, `patience` | `tol`, `patience` |

## Result contracts

The gradient-based trio populates at least these `AnnealResult` fields:

```python
result.best_sol  # torch.Tensor, the winning configuration
result.best_obj  # float, the loss value
result.runtime  # float, wall-clock seconds
result.score  # dict, human-readable summary
result.history  # dict[str, list], per-epoch metrics
```

CPRA additionally fills `result.score["extra"]["replicas"]` with
per-replica records.

`SCIPHybridResult` keeps the same `best_sol`, `best_obj`, `runtime`, `score`,
and `history` access while adding `scip_status`, `dual_bound`, `gap`,
`proven_optimal`, `n_warm_starts`, and the complete `qqa_result`.

`pareto_anneal` returns aligned nondominated `solutions` / `objectives`.
`blackbox_optimize` returns every evaluated point plus a feasibility-first
incumbent. These deliberately use feature-specific result types rather than
forcing non-scalar optimisation into `AnnealResult`.

## Performance picture

The historical smoke comparison in
[`docs/verification.md`](../verification.md) is useful for checking that each
backend executes, but it is not a speed claim. Hardware, warm-up, stopping
rules, and stochastic seeds materially affect the ranking. Use paired seeds,
equal wall-clock limits, separated compilation/warm-up, feasibility-first
quality, and confidence intervals for a reportable comparison. The public
MIPLIB/QPLIB runner implements those matched-budget campaign mechanics.

## Adding another backend

`qqa.pignn` is the canonical example of how a third-party can ship a
backend that plugs into the same tooling. See [Extending QQA4CO →
Custom backend](../develop/extending.md#a-new-solver-backend) for the
pattern.
