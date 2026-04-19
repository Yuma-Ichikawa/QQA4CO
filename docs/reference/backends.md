# Backend reference

QQA4CO ships three solver backends. They share the problem catalogue
and the `AnnealResult` interface, so swapping between them is one
function call (or one CLI flag).

## At a glance

| | **PQQA** | **CRA-PI-GNN** | **CPRA** |
|---|---|---|---|
| **Module** | `qqa.anneal` | `qqa.pignn.train_cra_pi_gnn` | `qqa.pignn.train_cpra_pi_gnn` |
| **CLI flag** | `--backend qqa` (default) | `--backend pignn` | `--backend cpra` |
| **Install** | core `pip install qqa` | `pip install "qqa[pignn]"` | `pip install "qqa[pignn]"` |
| **Returns** | `AnnealResult` | `AnnealResult` | `AnnealResult` (with `score["extra"]["replicas"]`) |
| **Variable kinds** | binary, spin, categorical, batched-instance | binary (graph QUBO only) | binary (graph QUBO only) |
| **Supported problems** | All 17 in catalogue | `mis`, `maxcut`, `maxclique`, `vertex_cover`, `graph_bisection` | Same as CRA-PI-GNN |
| **Diversity** | Optional (`div_param`) | Single solution per run | R diverse solutions per run (`vari_param` or `replica_problems`) |
| **Default LR** | `1.0` | `1e-4` | `1e-4` |
| **Optimiser** | AdamW on raw `(B, N)` tensor | AdamW on a 2-layer GCN | AdamW on a multi-head GCN |
| **GPU** | CUDA, MPS | CUDA (PyG-backed) | CUDA (PyG-backed) |
| **Reference paper** | [arXiv 2409.00184](https://arxiv.org/abs/2409.00184) | [NeurIPS 2024](https://openreview.net/forum?id=ykACV1IhjD) | [TMLR 2025](https://openreview.net/forum?id=ix33zd5zCw) |

## When to use which

* **Default — PQQA.** The cheapest, most thoroughly tested, and works
  on every problem in the catalogue.
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
| Early stopping | not yet | `tol`, `patience` | `tol`, `patience` |

## API contract — the same `AnnealResult`

All three backends populate at least these fields:

```python
result.best_sol  # torch.Tensor, the winning configuration
result.best_obj  # float, the loss value
result.runtime   # float, wall-clock seconds
result.score     # dict, human-readable summary
result.history   # dict[str, list], per-epoch metrics
```

CPRA additionally fills `result.score["extra"]["replicas"]` with
per-replica records.

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

## Adding a fourth backend

`qqa.pignn` is the canonical example of how a third-party can ship a
backend that plugs into the same tooling. See [Extending QQA4CO →
Custom backend](../develop/extending.md#a-new-solver-backend) for the
pattern.
