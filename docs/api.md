# API reference

## Stable solver API

New integrations should use the three top-level entry points and the shared
result/config contracts:

```python
import qqa

inspection = qqa.inspect("model.mps")
plan = qqa.plan("model.mps", profile="balanced", device="auto")
result = qqa.solve("model.mps", profile="balanced", budget=60)

print(result.status, result.best_obj, result.feasible)
print(result.violations.maximum_violation)
```

`qqa.solve` accepts an in-memory catalogue problem, `AlgebraicModel`,
`ModelIR`, or a supported file (`MPS`, `LP`, `QPLIB`, JSON, OPB, CNF/WCNF,
QUBO, or Ising). Configuration is strict: unknown options raise an error.
The default route is pure QQA; exact completion is enabled explicitly with
`profile="certify"` or `exact_backend=...`.

`budget` also accepts duration strings such as `"250ms"`, `"30s"`, and
`"2m"`. `goal="best|feasible|prove|diverse|pareto"` provides a compact
one-call policy. Run `qqa.doctor(model)` for strict bound, capability, scaling,
curvature, decomposition, route, and resource diagnostics before solving an
external model.

`SolveResult` keeps raw and repaired solutions separate. Its
`objective_value` is in the original objective direction,
`internal_energy` is the canonical minimisation value, and `merit_value` is
the quantity used by the search backend.

::: qqa.api

::: qqa.config

::: qqa.result

## Generated module reference

Below is the auto-generated documentation for the public modules. The
[Backends reference](reference/backends.md) page is a hand-curated
comparison if you only need to pick one entry point.

## Top-level

::: qqa
    options:
      show_root_toc_entry: false
      members:
        - solve
        - plan
        - inspect
        - doctor
        - SolverConfig
        - SolveResult
        - anneal
        - AnnealResult
        - fix_seed
        - generate_graph

### Legacy annealing result

`qqa.anneal` and the legacy-compatible solver backends return this result
contract:

::: qqa.annealing.AnnealResult

## Problems

::: qqa.problems.base

::: qqa.problems.qubo

::: qqa.problems.categorical

::: qqa.problems.spin

::: qqa.problems.extras

::: qqa.problems.user

## Mixed-variable optimisation

::: qqa.mixed

::: qqa.mixed.problem

::: qqa.mixed.variables

::: qqa.reporting

## Multi-objective optimisation

::: qqa.multiobjective

::: qqa.multiobjective.problem

::: qqa.multiobjective.solver

## Black-box optimisation

::: qqa.blackbox

::: qqa.blackbox.problem

::: qqa.blackbox.solver

## QQA × SCIP

The functions below require `pip install "qqa[scip]"`.

::: qqa.hybrid

::: qqa.hybrid.scip

## Sparse algebraic benchmarks

The QPLIB importer requires `pip install "qqa[qplib]"`; MPS execution and
SCIP-guided completion require `pip install "qqa[scip]"`.

::: qqa.algebraic

::: qqa.io

::: qqa.presolve

::: qqa.decomposition

::: qqa.dual

::: qqa.exact

::: qqa.runtime

::: qqa.service

::: qqa.templates

::: qqa.uncertainty

::: qqa.benchmarking

## TeX modelling

::: qqa.tex

::: qqa.tex.schema

::: qqa.tex.client

## Relaxations

::: qqa.relaxation

## Schedules

::: qqa.schedule

## Callbacks

::: qqa.callbacks

## Visualization

::: qqa.visualization

::: qqa.visuals

## Optional PyG backends

The functions below require the `pignn` extra
(`pip install "qqa[pignn]"`).

::: qqa.pignn

::: qqa.pignn.trainer
    options:
      members:
        - train_cra_pi_gnn
        - train_cpra_pi_gnn

::: qqa.pignn.model

::: qqa.pignn.graph
