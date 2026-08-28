# How to tune QQA hyper-parameters

The defaults in `qqa.anneal()` work out of the box for most problems
in the catalogue, but a handful of knobs control the
quality/speed/diversity trade-off. This page is a decision flow chart
in prose form.

## The five knobs that matter

| Knob | What it does | Sensible range |
|---|---|---|
| `sol_size` | Parallel population size | 32 (toy) → 4096 (large GPU) |
| `num_epochs` | Number of gradient steps | 500 (toy) → 50000 (CRA-paper regime) |
| `learning_rate` | AdamW LR | 0.01 → 1.0 (PQQA), 1e-5 → 1e-3 (CRA-PI-GNN/CPRA) |
| `min_bg` / `max_bg` | Linear schedule endpoints for the QQA penalty | `min_bg ∈ [-5, -1]`, `max_bg ∈ [0.05, 1]` |
| `div_param` | Weight of the cross-replica diversity term | `0` (disabled) → `0.1` |

The pignn backends have analogous knobs with different names: see
the [Backends reference](../reference/backends.md) for the mapping.

## `sol_size`

* **Larger `sol_size` improves the best-of-batch result** at almost
  exactly linear cost in GPU time. Increase it until you stop seeing
  improvements or you OOM.
* For a fair benchmark, use the same `sol_size` across runs you
  compare.
* Setting `sol_size = 1` turns off parallelism and the diversity term;
  use it for ablations only.

## `num_epochs`

* Pick `num_epochs` so that the schedule reaches the **annealed
  regime** before stopping. With the default `LinearBGSchedule(-2,
  0.1)` the relaxation locks onto binary corners around `bg ≈ 0`,
  which happens at `epoch ≈ 0.95 × num_epochs`.
* Watch the `loss_min` curve in `result.history`. If it is still
  decreasing at the end, double `num_epochs`.
* Watch the `bg` value in the final log line. If `bg < 0`, you stopped
  before the relaxation locked — increase `num_epochs` or `max_bg`.

## `learning_rate`

* **PQQA (default)** uses `learning_rate = 1.0`. This is *not* a typo
  — the QQA penalty normalises gradients so AdamW behaves more like a
  trust-region solver than a stochastic optimiser. Most problems work
  fine at `1.0`. For very small graphs (N < 50) try `0.5` if the
  optimiser oscillates.
* **CRA-PI-GNN / CPRA** use `1e-4` (matching the published paper).
  This is what the reference uses; deviating tends to hurt.

## `min_bg` and `max_bg`

* The interval `[min_bg, max_bg]` is the schedule's range. **Negative
  `min_bg`** contributes convex curvature from the QQA penalty. It yields a
  globally convex or unique soft minimum only when a verified objective
  curvature bound is dominated (for `c=2`, by `8 |min_bg|`).
  **Positive `max_bg`** drives the relaxation to
  the binary corners.
* If your problem is heavily constrained (lots of penalty edges),
  start with `min_bg = -5` and `max_bg = 0.5` so the soft phase
  dominates.
* If your problem is loosely constrained, the defaults `(-2, 0.1)`
  are fine.
* See the [Algorithm explainer](../explanation/algorithm.md) for the
  geometric intuition.

## `div_param`

* Off by default (`0`). Enable when you want **diverse** solutions in
  one run — different replicas land in different basins.
* Start at `0.01`; multiply by 10 if replicas still collapse to the
  same solution.
* `qqa.AutoDivTuner(target=0.3)` adapts `div_param` online to keep the
  population diversity at a target ratio. Pass it as a callback if you
  do not want to hand-tune.
* For the *CPRA* backend the equivalent knob is `vari_param` (variation
  diversification) or per-head `replica_problems` (penalty
  diversification).

## Decision flow

```
              ┌───────────────────────────────────┐
              │  Best-of-batch result is OK?      │
              └────────────────┬──────────────────┘
                               ▼
       no, too poor                       yes, but slow
            │                                 │
            ▼                                 ▼
  ┌────────────────────┐         ┌─────────────────────┐
  │  Increase epochs   │         │  Decrease sol_size  │
  │  by 2x; if no      │         │  or epochs to fit   │
  │  improvement,      │         │  your time budget   │
  │  increase sol_size │         └─────────────────────┘
  │  by 2x.            │
  └─────────┬──────────┘
            ▼
   ┌──────────────────────┐
   │  Still poor?         │
   │  Make schedule       │
   │  longer in the soft  │
   │  phase: min_bg=-5,   │
   │  max_bg=0.3.         │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  Want diverse        │
   │  solutions?          │
   │  Set div_param=0.05  │
   │  or use AutoDivTuner.│
   └──────────────────────┘
```

## Reading `result.history`

Every solve returns `result.history` (when `record_history=True`,
default). The most useful keys for tuning:

| Key | What to look for |
|---|---|
| `loss_min` | Should plateau before the last epoch — if not, increase `num_epochs` |
| `penalty_mean` | Should decrease toward 0 — if not, your relaxation isn't locking |
| `diversity` | Should be high in the early (soft) phase, drop near the end |
| `bg` | Sanity-check the schedule actually went where you expected |

The Streamlit GUI (`qqa gui`) plots all of these out of the box.

## Per-problem starting points

Roughly tuned defaults for the problems in the catalogue:

| Problem | sol_size | epochs | min_bg / max_bg |
|---|---|---|---|
| MIS / MaxClique / VertexCover (N≤200) | 128 | 2000 | -2 / 0.1 |
| MIS / MaxClique (N=1000+) | 256 | 5000 | -3 / 0.3 |
| MaxCut | 128 | 1500 | -2 / 0.1 |
| GraphBisection | 128 | 2000 | -3 / 0.3 |
| Coloring (K=3–5) | 256 | 3000 | -2 / 0.1 |
| TSP / QAP | 64 | 3000 | -2 / 0.2 |
| Ising 1D / EA / SK | 200 | 2000 | -2 / 0.1 |
| BinaryPerceptron / Hopfield | 200 | 3000 | -2 / 0.1 |
| Knapsack / NumberPartitioning / MaxSAT3 | 128 | 1000 | -2 / 0.1 |

These are a starting point, not a recipe. If you find better defaults
for your problem, please open a PR updating this table.
