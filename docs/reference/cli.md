# CLI reference

Installing QQA4CO registers a single console script:

```
qqa <subcommand> [options]
```

The CLI is a thin argparse layer over the public Python API and uses
**no third-party packages**, so it works in minimal containers. For a
quick "what does this flag do?" prefer `qqa <subcommand> --help`; this
page is the explanation of how the flags interact.

## `qqa version`

```bash
qqa version
# 0.3.0
```

Prints the value of `qqa.__version__`, which is single-sourced from
the wheel metadata via `importlib.metadata`. Use this to verify the
installed version inside a Docker container or a CI job.

## `qqa solve`

The main subcommand. Solves a single problem and either prints the
score or pickles the full `AnnealResult` to disk.

### Common flags

| Flag | Default | What it does |
|---|---|---|
| `--problem` | (required) | One of `mis`, `maxcut`, `maxclique`, `coloring`, `ising1d`, `ea`, `sk`, `perceptron`, `hopfield`, `knapsack`, `number_partition`, `vertex_cover`, `graph_bisection`, `maxsat3`, `tsp`, `qap`, `nqueens` |
| `--problem-file` | (none) | Path to a Python file defining `problem` or `make_problem()` — replaces `--problem` |
| `--graph-file` | (none) | Pickled / GraphML / edgelist NetworkX graph (graph problems only) |
| `--size` | `50` | Size for synthetic problem generators |
| `--sol-size` | `100` | Parallel population size |
| `--epochs` | `1000` | Number of gradient steps |
| `--device` | `cpu` | `cpu`, `cuda`, `cuda:0`, `mps` |
| `--seed` | `0` | Seed passed to `qqa.fix_seed` |
| `--quiet` | off | Suppress per-epoch progress logs |
| `--output` | (none) | If given, pickle the full `AnnealResult` to this path |

### Hyper-parameter flags

| Flag | Default | What it does |
|---|---|---|
| `--learning-rate` | backend-aware (1.0 for `qqa`, 1e-4 for `pignn`/`cpra`) | AdamW learning rate |
| `--temp` | `0.0` | Langevin noise temperature (0 = no noise) |
| `--min-bg` / `--max-bg` | `-2.0` / `0.1` | Linear schedule endpoints |
| `--curve-rate` | `2` | QQA penalty exponent (must be even) |
| `--div-param` | `0.0` | Cross-replica diversity weight (0 = off) |

### Backend flags

| Flag | Default | What it does |
|---|---|---|
| `--backend` | `qqa` | `qqa` (default annealer), `pignn` (CRA-PI-GNN), `cpra` (CPRA) — last two require `qqa[pignn]` |
| `--pignn-init-reg-param` | `-20.0` | CRA initial γ (only `--backend pignn`/`cpra`) |
| `--pignn-annealing-rate` | `1e-3` | CRA γ increment per epoch |
| `--pignn-tol` / `--pignn-patience` | `1e-4` / `1000` | Early-stopping (loss-stagnation) |
| `--pignn-hidden` | √N | GCN hidden width |
| `--pignn-no-annealing` | off | Run vanilla PI-GNN (γ ≡ 0) instead of CRA-style annealing |
| `--cpra-num-replicas` | `4` | Number of CPRA heads R |
| `--cpra-vari-param` | `0.0` | CPRA variation-diversification weight |
| `--cpra-penalty-levels` | (none) | Comma-separated penalty weights for penalty-diversification (MIS / VertexCover only) |

### Per-backend defaults

The `--learning-rate` flag is **backend-aware**. Omit it and the CLI
applies the right default:

| Backend | Default `--learning-rate` |
|---|---|
| `qqa` (default) | `1.0` |
| `pignn` | `1e-4` |
| `cpra` | `1e-4` |

This is the behaviour of `qqa.anneal`, `qqa.pignn.train_cra_pi_gnn`,
and `qqa.pignn.train_cpra_pi_gnn` respectively. Pass an explicit
`--learning-rate` to override.

### Problems supported per backend

| Backend | Supported `--problem` values |
|---|---|
| `qqa` | All 17 problems in the catalogue |
| `pignn` | `mis`, `maxcut`, `maxclique`, `vertex_cover`, `graph_bisection` |
| `cpra` | Same as `pignn` (penalty diversification works for `mis`, `vertex_cover` only) |

### Examples

```bash
# Quickest check that the install works.
qqa solve --problem sk --size 60 --sol-size 64 --epochs 500

# A real MIS solve on a saved graph.
qqa solve --problem mis --graph-file data/my_graph.gpickle \
          --sol-size 256 --epochs 2000 --device cuda \
          --output results/mis.pkl

# CRA-PI-GNN comparison run on the same graph.
qqa solve --problem mis --graph-file data/my_graph.gpickle \
          --backend pignn --epochs 5000 --device cuda \
          --output results/mis_cra.pkl

# CPRA portfolio: four MIS solutions at different penalty levels.
qqa solve --problem mis --graph-file data/my_graph.gpickle \
          --backend cpra --cpra-num-replicas 4 \
          --cpra-penalty-levels 1.0,1.5,2.0,2.5 \
          --epochs 5000 --device cuda --output results/mis_cpra.pkl

# A user-defined problem.
qqa solve --problem-file my_problem.py --sol-size 128 --epochs 1500
```

## `qqa bench`

Run a small benchmark on bundled data. Useful as a reproducibility
sanity check.

| Flag | Default | What it does |
|---|---|---|
| `--preset` | `er-small` | One of `er-small`, `sk-small`, `ea-small` |
| `--sol-size` | `64` | |
| `--epochs` | `500` | |
| `--device` | `cpu` | |
| `--seed` | `0` | |

```bash
qqa bench --preset er-small
```

## `qqa gui`

Launch the Streamlit dashboard in a subprocess. Reads the dashboard
sources either from the installed wheel (under `qqa/_app/`) or from
the repo's `app/` directory in editable installs.

| Flag | Default | What it does |
|---|---|---|
| `--port` | `8501` | |
| `--host` | `localhost` | |
| `--headless` | off | Disable the local browser autostart (useful on remote servers) |

```bash
qqa gui --port 8505 --headless
# Open http://<your-server>:8505 in a browser
```

## Exit codes

* `0` — success.
* `1` — non-fatal user error (bad combination of flags, infeasible
  result, etc.). The error message is printed to stderr.
* `2` — argparse failure (unknown flag, missing required argument).

## Where to look in the source

The whole CLI lives in `src/qqa/cli.py` (~600 lines). Read it
top-to-bottom — the structure is one `build_parser()` followed by one
`_cmd_<name>(args)` function per subcommand.
