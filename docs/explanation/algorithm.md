# Algorithm — SA, PQQA, CRA-PI-GNN, CPRA in one page

QQA4CO ships **four solver families** that all share the same problem
catalogue and a common `AnnealResult` / `SAResult` interface. They differ
in the representation of the candidate variable and in how the relaxation
(or temperature) is annealed.

| Solver | Backend | Variable | Schedule | Diversity |
|---|---|---|---|---|
| **SA** (baseline) | `qqa.simulated_annealing` | discrete `{0,1}` or `{−1,+1}` per chain | inverse-temperature `β` (geometric / linear) | independent parallel chains |
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
binary corners `{0, 1}^N`. A negative `β` makes the penalty contribution
convex. It does **not** by itself make the combined objective convex or its
minimum unique. For `c = 2`, the penalty adds the diagonal curvature
`8 |β| I`; global convexity follows only when this dominates a valid lower
bound on the objective Hessian. QQA uses this soft, curvature-controlled
phase for exploration; as `β` grows the binary
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
2. **Run `B` replicas in parallel**, optionally with heterogeneous roles,
   per-replica schedules, and replica exchange.
3. **Add an explicit diversity reward** — `−div_param × Σ_i std_b(x_b)`
   — to keep replicas from collapsing into the same basin (set
   `div_param=0` for vanilla single-shot annealing).
4. **(Optional) Langevin perturbation** every step (`temp > 0`) for
   gradient-Langevin dynamics.

Source: `src/qqa/annealing.py` (the entire algorithm in ~280 lines)
plus `src/qqa/relaxation.py` and `src/qqa/schedule.py`.

This is the **recommended default** for graph problems, spin glasses,
permutation problems, and anything that can be formulated as a `loss_fn
(x)`. Typed factors must declare a differentiable, subgradient, or prox
capability for the pure QQA route; non-differentiable factors are routed to
CP/SAT/exact engines instead. Supported differentiable models run
on CPU/CUDA/MPS.

## CRA-PI-GNN — `qqa.pignn.train_cra_pi_gnn`

Reference paper: Y. Ichikawa, *"Controlling Continuous Relaxation for
Combinatorial Optimization"*
([NeurIPS 2024](https://openreview.net/forum?id=ykACV1IhJD)).

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

## SA — `qqa.simulated_annealing`

The baseline. Implemented in `src/qqa/sa.py`; no learning, no
relaxation — just classical Simulated Annealing parallelised across
chains on GPU.

For QUBO problems (every problem with a `Q_mat` attribute) we use a
**Glauber-like parallel update**: at each sweep, the change in energy
\(\Delta E_i\) for flipping bit \(i\) is computed in closed form,

\[
\\Delta E_i \\;=\\; (1 - 2 x_i)\\,\\bigl(2(Q\\,x)_i - 2 x_i Q_{ii} + Q_{ii}\\bigr),
\\]

so a single matmul gives \(\\Delta E\\) for every bit. Each bit is then
flipped independently with probability
\(\\min\\bigl(1, e^{-\\beta\\,\\Delta E_i}\\bigr)\). This is *not* a strictly
correct single-spin Metropolis chain — proposals are not conditional on
neighbours updated earlier in the same sweep — and no exact Boltzmann
stationarity claim is made for that parallel transition. It matches the
"parallel-tempered classical SA" baseline used throughout the QQA / CPRA
literature.

For non-QUBO problems (`Knapsack`, `MaxSAT3`, `BinaryPerceptron`, …) we
fall back to **strict single-spin sequential Metropolis**: one
`problem.loss_fn` call per bit per sweep. Correct, but \(N\\times\\) more
expensive than the QUBO fast path.

The β schedule is geometric by default (recommended for SA),
\(\\beta_t = \\beta_0 (\\beta_T/\\beta_0)^{t/T}\\). Use the `beta_schedule="linear"`
flag to disable.

`sol_size` is the number of independent chains, run in parallel as the
batch dimension on GPU. There is no diversity term — the only "diversity"
is the random initialisation per chain, exactly as in textbook SA.

When SA is the right choice:

* **As a baseline.** If your fancy method (QQA / CRA / CPRA / your own)
  doesn't beat SA on a fixed compute budget, the relaxation isn't
  helping for that problem.
* **Tiny instances** (`N ≲ 50`) where SA converges in a few hundred
  sweeps and the GCN training overhead of CRA-PI-GNN is wasted.
* **Sanity check after problem changes.** SA's flat code path makes it
  much easier to debug a new `loss_fn` than the relaxed-then-annealed
  path of QQA.

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
* **Reach for SA first** when you want a *baseline* — if QQA / CRA /
  CPRA can't beat SA on the same budget, the extra machinery isn't
  paying its way for your problem.
* **Switch to CRA-PI-GNN** when you need the GNN's smoothness prior on
  large, sparse graph problems and you can afford a longer training
  run.
* **Switch to CPRA** when you specifically need *diverse* solutions
  (penalty portfolio, mode coverage). Returns `R` solutions in one
  training run.
