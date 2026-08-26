# Architecture

This page is a 10-minute read that maps the conceptual pieces of QQA4CO
to concrete source files. After reading it you should know exactly
where to look in the code for any feature.

## The big picture

QQA4CO has one canonical model/result boundary and several opt-in execution
routes. QQA remains the default primal-search engine:

```
Python / files / legacy models
             │
             ▼
      Canonical ModelIR ── objective sense / sparse factors / constraints
             │
             ▼
      inspect + presolve ─ reversible fixing / scaling / decomposition
             │
             ▼
        solver planner ─── profile / budget / device / VRAM / certificate
             │
      ┌──────┼──────────┐
      ▼      ▼          ▼
 sparse QQA  repair/LNS exact adapters
      └──────┼──────────┘
             ▼
        SolveResult ───── raw/repaired / objective/merit / bounds/gaps
```

The catalogue `COProblem` and `AnnealResult` APIs remain supported. Adapters
form the compatibility boundary; new model-facing functionality belongs in
`model/`, `compile/`, `portfolio/`, `repair/`, or an optional backend rather
than as another top-level solve function.

The standard annealer remains the core engine. Specialised orchestrators are
kept beside it rather than added as branches inside its hot loop:

* `multiobjective/` assigns a scalarisation to each parallel replica and owns
  its nondominated archive.
* `blackbox/` replaces gradients with a surrogate and evaluation budget while
  reusing typed mixed-variable domains.
* `hybrid/scip.py` consumes a QQA population and hands it to an optional exact
  solver.
* `algebraic/`, `io/`, and `presolve/` preserve sparse MIP/QP structure for
  MIPLIB/QPLIB instead of compiling it into Python callables.
* `hybrid/scip_heuristic.py` runs QQA repeatedly inside SCIP on an uncertain
  integer core; `decomposition/` completes the remaining variables in an
  independent sub-SCIP.
* `tex/` is a modelling front-end: it emits `MixedProblem` or
  `MultiObjectiveProblem`, never executable model code.

The CLI, UI, visualisation modules, and optional backends consume only public
planning and result contracts.

The benchmark path therefore has a different data flow:

```text
MPS/QPLIB -> sparse AlgebraicModel -> SCIP presolve/node LP
          -> RENS/RINS core -> objective/active-row surrogate
          -> cheap integral move -> conditional float64 QQA population
          -> independent continuous completion -> SCIP trySol/proof
```

Original and transformed SCIP variables remain mapped across presolve. A
separate tracker evaluates every accepted incumbent in the original sparse
model, which prevents nonlinear objective epigraph slack from corrupting
reported primal progress. QQA never owns the proof, and one wall-clock deadline
covers all arrows.

## Data flow for a stable solve

1. `qqa.solve` loads or adapts the input into a canonical model when possible.
2. `inspect` extracts size, domain, factor, constraint, sparsity, and memory
   features without solving.
3. Conservative presolve records every reduction in a transformation ledger.
4. The rule-based planner selects replicas, QQA route, repair/local search,
   and an exact backend only when requested.
5. Sparse QQA generates diverse incumbents. Constraint models add scaled,
   vector augmented-Lagrangian state and feasibility-first archiving.
6. Repair is pure: it creates a separate candidate and never mutates the raw
   incumbent. Optional exact adapters receive the QQA warm start.
7. `SolveResult` restores original variable order and reports each semantic
   quantity separately.

## Legacy `anneal` data flow

1. **User builds a problem.** `qqa.MaximumIndependentSet(g, penalty=2,
   device='cuda')` constructs a `COProblem` whose `Q_mat` lives on the
   right device and whose `relaxation` is a `BinaryRelaxation()`.
2. **User calls** `qqa.anneal(problem, sol_size=128, num_epochs=2000)`.
3. **`anneal` initialises** the latent tensor `x = relax.init(sol_size,
   problem, device)` — shape `(B, N)` for binary, `(B, N, K)` for
   categorical, `(B, I, N)` for batched-instance.
4. **For every epoch** the loop:
   * computes `bg = schedule(epoch, num_epochs)`,
   * forwards `x_fwd = relax.forward(x)` and gets `losses =
     problem.loss_fn(x_fwd)`,
   * adds `penalties * bg` (the QQA continuous-relaxation penalty) and
     a diversity term scaled by `div_param`,
   * back-propagates and steps AdamW,
   * applies an in-place `relax.perturb_(x, lr, temp)` step (Langevin
     noise + clamping),
   * projects to discrete with `relax.project(x)` to evaluate the true
     objective and update the running best,
   * fires `on_epoch_end(state)` on every callback.
5. **At the end** `anneal` calls `problem.score_summary(best_sol)` to
   produce the human-readable result and packages everything into an
   `AnnealResult` dataclass.

## Why this decomposition

* **One annealer for every variable kind.** Binary, spin, categorical,
  permutation problems all use the same `anneal()` because the
  variable-specific bits live behind the `Relaxation` protocol. Adding
  a fifth variable kind is a single new class — no edits to the loop.
* **Problems are pure functions of `x`.** A problem only has to know
  how to compute its loss; it never sees the optimiser, the schedule,
  or the parallel batch dimension semantics. This is what makes
  `qqa.UserProblem` work — wrap *any* `loss_fn(x)` and you have a
  first-class problem.
* **Callbacks are read-only by default.** They see the full state but
  the only sanctioned write target is `state.hyperparams` (a mutable
  dict). This is enough to implement `AutoDivTuner` and others without
  inviting callbacks to silently corrupt the training loop.
* **Backends are functions, not frameworks.** A "backend" is anything
  that takes a `COProblem` and returns an `AnnealResult`. The
  `qqa.pignn` trainers do not subclass anything; they just satisfy
  that contract, which is why they reuse the same downstream tooling.

## Where extension points live in the source

| Extension | File | Lines | Note |
|---|---|---|---|
| New problem | `src/qqa/problems/*.py` | varies | Subclass `COProblem` |
| New relaxation | `src/qqa/relaxation.py` | ~220 | Implement `Protocol` |
| New schedule | anywhere | n/a | Any `(epoch, T) -> float` callable |
| New callback | `src/qqa/callbacks.py` (or external) | ~170 | Subclass `Callback` |
| New backend | `src/qqa/<name>/` (e.g. `pignn/`) | ~700 reference | Return `AnnealResult` |
| New modelling front-end | `src/qqa/<name>/` | varies | Compile into an existing problem contract |
| New specialised optimiser | feature package `solver.py` | varies | Keep its result and plots beside the feature |

See [Extending QQA4CO](../develop/extending.md) for worked examples of
each.

## The optional `pignn` backend

`qqa.pignn` is the canonical "second backend" example. It illustrates
three idioms worth copying:

1. **Heavy deps stay opt-in.** `torch_geometric` is never imported
   from the top-level `qqa.__init__`; `qqa.pignn._import.require_pyg`
   raises an actionable error if the extra is missing.
2. **Trainers reuse `BinaryRelaxation.penalty`** so the CRA loss and
   the QQA loss are numerically identical for `curve_rate=2`, making
   head-to-head comparisons trustworthy.
3. **They return `qqa.AnnealResult`**, which is why `qqa solve
   --backend pignn ...` and the Streamlit dashboard work with no extra
   code.

## CLI / GUI / scripts as "external consumers"

The CLI, the Streamlit app, and the `scripts/` benchmarks all call
`qqa.anneal` (or `qqa.pignn.train_*`) and then read `AnnealResult`.
None of them peek inside the solver loop. That separation is what lets
you extend the solver without touching the user-facing tooling — the
tooling is bound to the *contract*, not the implementation.
