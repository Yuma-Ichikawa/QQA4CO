# Backend reference

QQA4CO ships a core gradient solver, optional GNN variants, an exact SCIP
hybrid, and sampling baselines. The high-level optimisation layer additionally
provides dedicated Pareto and black-box solvers.

## At a glance

| | **PQQA** | **QQA→SCIP** | **SG-CQQA** | **CRA-PI-GNN** | **CPRA** |
|---|---|---|---|---|---|
| **Module** | `qqa.anneal` | `qqa.solve_qqa_scip` | `qqa.benchmarking.run_miplib` / `run_qplib` | `qqa.pignn.train_cra_pi_gnn` | `qqa.pignn.train_cpra_pi_gnn` |
| **CLI flag** | `--backend qqa` | `--backend scip` | `benchmark run --solver sg-cqqa` | `--backend pignn` | `--backend cpra` |
| **Install** | `pip install qqa` | `pip install "qqa[scip]"` | `pip install "qqa[scip,qplib]"` | `pip install "qqa[pignn]"` | `pip install "qqa[pignn]"` |
| **Returns** | `AnnealResult` | `SCIPHybridResult` | `BenchmarkResult` | `AnnealResult` | `AnnealResult` |
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
  remains. Every candidate is completed in a sub-SCIP.
  See the [MIPLIB/QPLIB guide](../miplib-qplib.md).
* **CRA-PI-GNN** when you specifically want the GNN inductive bias
  (smoothness over the graph) on large sparse graph problems and you
  can afford a long training run.
* **CPRA** when you need *diverse* solutions (penalty portfolio, mode
  coverage). Returns R solutions in one training run.

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

For a fair head-to-head on a representative graph problem (MIS on
ER-small, N=200), see the table reproduced from
`scripts/bench_qqa_vs_pignn.py` in
[`docs/verification.md`](../verification.md). Headline:

* QQA reaches the same MIS size as CRA-PI-GNN in *roughly an
  order-of-magnitude less wall-time* on this problem class, because
  the parallel population effectively replaces the GCN's smoothing
  with raw exploration.
* CRA-PI-GNN is occasionally a hair better on very dense graphs where
  the GCN's locality prior helps; the gap is small.
* CPRA's diversity is real — its 4 heads land on different solutions
  with `vari_param > 0`, which neither QQA nor CRA can do without
  multiple runs.

## Adding another backend

`qqa.pignn` is the canonical example of how a third-party can ship a
backend that plugs into the same tooling. See [Extending QQA4CO →
Custom backend](../develop/extending.md#a-new-solver-backend) for the
pattern.
