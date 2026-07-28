# API reference

Below is the auto-generated documentation for the public modules. The
[Backends reference](reference/backends.md) page is a hand-curated
comparison if you only need to pick one entry point.

## Top-level

::: qqa
    options:
      show_root_toc_entry: false
      members:
        - anneal
        - AnnealResult
        - fix_seed
        - generate_graph

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
