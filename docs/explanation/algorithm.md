# Algorithm — PQQA, CRA-PI-GNN, CPRA in one page

QQA4CO ships **three solver families** that all share the same problem
catalogue and the same `AnnealResult` interface. They differ in the
representation of the relaxed variable and in how the relaxation
penalty is annealed.

| Solver | Backend | Variable | Schedule | Diversity |
|---|---|---|---|---|
| **PQQA** (default) | `qqa.anneal` | parallel batch of `B` raw tensors | linear `bg` over epochs | optional `div_param` cross-replica term |
| **CRA-PI-GNN** | `qqa.pignn.train_cra_pi_gnn` | output of a 2-layer GCN over the problem graph | linear `γ` from `init_reg_param` (≈ −20) to ≥ 0 | none (single replica) |
| **CPRA** | `qqa.pignn.train_cpra_pi_gnn` | `R` GCN heads sharing one backbone | same as CRA-PI-GNN, optionally per-head | optional `vari_param` term, or per-head `replica_problems` |

This page summarises what each of them is mathematically, with
pointers to the original papers and to the matching source files.

## Shared substrate — the QQA penalty

All three solvers minimise

\[
L(x;\\,\\beta) \\;=\\; f(x) \\;+\\; \\beta\\sum_{i} \\bigl(1 - (1 - 2x_i)^c\\bigr)
\\;+\\; (\\text{diversity term})
\\]

over the continuous relaxation `x ∈ [0, 1]^N`. Here `f(x)` is the
problem's `loss_fn`, `c` is `curve_rate` (must be even — default 2),
and `β` is the schedule (`bg` in QQA, `γ` / `reg_param` in CRA / CPRA;
the symbol is different but the role is identical).

**Why this penalty?** The function \\(\\Phi(p) = \\sum_i 1 - (1 -
2p_i)^c\\) is concave on `[0, 1]` for even `c` and minimised at the
binary corners `{0, 1}^N`. With negative `β` the *combined* loss is
convex, so AdamW finds a unique soft minimum; as `β` grows the binary
corners become attractors and the soft solution snaps onto a discrete
solution. This is the "continuous relaxation annealing" trick that
unifies all three solvers — see `src/qqa/relaxation.py:69-72` for the
exact implementation.

## PQQA — `qqa.anneal`

Reference paper: Y. Ichikawa, *"Optimization by Parallel Quasi-Quantum
Annealing with Gradient-Based Sampling"*
([arXiv:2409.00184](https://arxiv.org/abs/2409.00184), 2024).

Key ideas:

1. **Lift the discrete optimisation to a continuous one** with the
   penalty above.
2. **Run `B` replicas in parallel** with shared schedule but
   independent latent tensors.
3. **Add an explicit diversity reward** — `−div_param × Σ_i std_b(x_b)`
   — to keep replicas from collapsing into the same basin (set
   `div_param=0` for vanilla single-shot annealing).
4. **(Optional) Langevin perturbation** every step (`temp > 0`) for
   gradient-Langevin dynamics.

Source: `src/qqa/annealing.py` (the entire algorithm in ~280 lines)
plus `src/qqa/relaxation.py` and `src/qqa/schedule.py`.

This is the **recommended default** for graph problems, spin glasses,
permutation problems, and anything that can be formulated as a `loss_fn
(x)`. It is fully differentiable, supports any variable kind, and runs
on CPU/CUDA/MPS.

## CRA-PI-GNN — `qqa.pignn.train_cra_pi_gnn`

Reference paper: Y. Ichikawa, *"Controlling Continuous Relaxation for
Combinatorial Optimization"*
([NeurIPS 2024](https://openreview.net/forum?id=ykACV1IhjD)).

Reference implementation (DGL):
<https://github.com/Yuma-Ichikawa/CRA4CO>.

Key idea: **parameterise** the relaxed solution `p ∈ [0,1]^N` as the
output of a 2-layer GCN over the problem graph, with a learnable
per-node embedding. The GCN's inductive bias (smoothness over the
graph) is what gives CRA-PI-GNN its edge on graph problems.

Algorithmically the loss is *exactly* the same penalty as PQQA with
`B = 1`; what differs is:

* the optimisation variable is the GCN parameters, not `x` itself,
* the schedule defaults are different (`init_reg_param = −20`,
  `annealing_rate = 1e-3` per epoch, AdamW `lr = 1e-4`),
* there is an early-stopping rule (loss-stagnation patience).

Source: `src/qqa/pignn/trainer.py:78`–`train_cra_pi_gnn`. The DGL
backbone of the reference is replaced with `torch_geometric` — see
[`docs/explanation/architecture.md`](architecture.md) for why.

## CPRA — `qqa.pignn.train_cpra_pi_gnn`

Reference paper: Y. Ichikawa & H. Iwashita, *"Continuous Parallel
Relaxation for Finding Diverse Solutions in Combinatorial Optimization
Problems"* ([TMLR
2025](https://openreview.net/forum?id=ix33zd5zCw)).

Reference implementation (DGL):
<https://github.com/Yuma-Ichikawa/CPRA4CO>.

Key idea: **multi-head extension** of CRA-PI-GNN. The same GCN
backbone produces `R` parallel head outputs `p ∈ [0,1]^{N×R}`; each
head can have its own loss (penalty diversification) or share one and
be pushed apart by an inter-head spread term (variation
diversification).

Two diversity modes:

* **Penalty diversification** — pass `replica_problems = [problem_with
  _penalty_p for p in penalties]`. Each head sees a different penalty
  weight, naturally producing a portfolio of solutions trading off
  feasibility and objective.
* **Variation diversification** — pass `vari_param > 0`. The trainer
  adds `−vari_param × spread(probs)` to the loss, where `spread` is
  the cross-head standard deviation. All heads share the same problem
  but explore different basins.

Source: `src/qqa/pignn/trainer.py:407`–`train_cpra_pi_gnn` plus
`src/qqa/pignn/model.py` (`GCNNet(num_replicas=R)`).

## When to use which

See [Backend reference](../reference/backends.md) for a side-by-side
matrix.

* **Default to PQQA.** It works on every problem in the catalogue, is
  the cheapest by far, and is the most thoroughly tested.
* **Switch to CRA-PI-GNN** when you need the GNN's smoothness prior on
  large, sparse graph problems and you can afford a longer training
  run.
* **Switch to CPRA** when you specifically need *diverse* solutions
  (penalty portfolio, mode coverage). Returns `R` solutions in one
  training run.
