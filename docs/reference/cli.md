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
| `--device` | `auto` | `auto` chooses CUDA → MPS → CPU; explicit `cpu`, `cuda`, `cuda:0`, `mps` also work |
| `--seed` | `0` | Seed passed to `qqa.fix_seed` |
| `--quiet` | off | Suppress per-epoch progress logs |
| `--output` | (none) | If given, pickle the full `AnnealResult` to this path |
| `--report` | (none) | Write a self-contained interactive HTML diagnostic report |

### Hyper-parameter flags

| Flag | Default | What it does |
|---|---|---|
| `--learning-rate` | backend-aware (1.0 for `qqa`, 1e-4 for `pignn`/`cpra`) | AdamW learning rate |
| `--temp` | `0.0` | Langevin noise temperature (0 = no noise) |
| `--min-bg` / `--max-bg` | `-2.0` / `0.1` | Linear schedule endpoints |
| `--curve-rate` | `2` | QQA penalty exponent (must be even) |
| `--div-param` | `0.0` | Cross-replica diversity weight (0 = off) |
| `--restart-patience` | `250` | Replace weak QQA replicas after this many epochs without an incumbent improvement; `0` disables |
| `--restart-fraction` | `0.15` | Fraction of weak replicas replaced at each recovery event |
| `--restart-jitter` | `0.10` | Local latent jitter around the incumbent for half of restarted replicas |
| `--gradient-clip` | `100` | Global latent-gradient norm cap; `0` disables |

### Backend flags

| Flag | Default | What it does |
|---|---|---|
| `--backend` | `qqa` | `qqa`, `scip` (QQA→exact QUBO refinement), `pignn`, `cpra`, or `sa` |
| `--scip-time-limit` / `--scip-gap` | `60` / `0` | Total QQA+SCIP wall-clock budget and target relative gap |
| `--scip-warm-starts` / `--scip-threads` | `32` / `1` | Diverse QQA incumbents and exact-solver threads |
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
| `qqa` | All problems exposed by the current built-in catalogue |
| `scip` | Single-instance `QUBOProblem` models |
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

# QQA GPU exploration followed by SCIP improvement/certification.
qqa solve --problem maxcut --size 80 --backend scip --device cuda \
          --epochs 1500 --scip-time-limit 120 --scip-warm-starts 64

# A user-defined problem.
qqa solve --problem-file my_problem.py --sol-size 128 --epochs 1500 \
          --report results/model-report.html
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

## `qqa benchmark`

Fetch, inspect, and solve public MIPLIB/QPLIB instances. This is distinct from
`qqa bench`, which runs the bundled combinatorial presets.

```bash
qqa benchmark fetch miplib --instance pk1 --output benchmarks/miplib
qqa benchmark fetch qplib --instance 31 --output benchmarks/qplib
qqa benchmark inspect benchmarks/qplib/QPLIB_0031.qplib
qqa benchmark run benchmarks/miplib/pk1.mps.gz \
  --solver sg-cqqa --time-limit 60 --output result.json
qqa benchmark compare benchmarks/miplib/pk1.mps.gz \
  --baseline-solver scip-aggressive --seeds 0 1 2 \
  --time-limit 60 --output comparison.json
qqa benchmark merge shard-0.json shard-1.json --output comparison.json
```

`fetch` accepts `miplib` or `qplib`; omit `--instance` to download the full
official archive. `inspect` emits sparse dimensions, variable-type counts,
PROBTYPE when available, and portable source provenance.

`run` accepts `--solver scip`, `--solver scip-aggressive`, or
`--solver sg-cqqa`. Shared flags are
`--time-limit`, `--gap`, `--threads`, `--reference-file`, `--format`,
`--output`, and `--quiet`. SG-CQQA additionally accepts `--core-size`,
`--sol-size`, `--epochs`, `--max-calls`, `--max-candidates`,
`--completion-time`, `--completion-nodes`, `--min-call-time`,
`--min-qqa-time`, `--fast-candidates`, `--max-lp-rows`, objective/row/proximity
weights, `--continue-qqa-without-improvement`, `--seed`, and `--device`.
The time limit covers parsing/setup, QQA, continuous completion, and SCIP.

`compare` runs a paired Cartesian product of input instances, `--solvers`, and
`--seeds`. Every pair receives the same total budget and thread count. Its JSON
contains per-run trajectories, portable run configuration, per-solver medians,
and win/tie/loss counts against `--baseline-solver`. Use
`scip-aggressive` as the ablation baseline when measuring the incremental
effect of the SG-CQQA plugin, because both then use the same aggressive native
SCIP heuristic setting.

`--threads` constrains SCIP workers, LP-solver threads, and (for SG-CQQA)
Torch threads. Reproducible CPU campaigns should additionally cap the BLAS and
OpenMP thread pools in their launcher.
For `run`, metric clocks begin before parsing. For paired `compare`, one common
algebraic import is excluded from every solver and the clocks begin before each
solver model is built. Primal integral uses the configured time limit as a
fixed common horizon.

`--output` is also an incremental checkpoint. Add `--continue-on-error` for a
large heterogeneous archive and repeat the identical command with `--resume`
after interruption. A mismatched instance list, solver list, seed set, time,
thread count, reference name, or SG-CQQA configuration is rejected rather than
mixed into an existing campaign. `--retry-failures` retries only the anonymous
failure records during a resumed run.
`merge` combines disjoint comparison shards after checking that every setting
except the instance list is identical. It rejects overlapping instances or
duplicate solver/instance/seed rows and recomputes all medians and W/T/L counts.
See the [MIPLIB/QPLIB guide](../miplib-qplib.md) for metric definitions and
reproducibility guidance.

## `qqa ask`

Describe a bounded optimisation problem in ordinary language, compile it
through an OpenAI-compatible endpoint, validate the model locally, show the
route chosen by trusted code, and solve:

```bash
export QQA_LLM_API_KEY='…'
export QQA_LLM_BASE_URL='https://api.example.com'
export QQA_LLM_MODEL='your-model-id'

qqa ask \
  "Choose integer batches in [0,20] and real overtime in [0,8]. \
   Minimize 3*batches + square(overtime), with 4*batches + overtime >= 45." \
  --solver auto --device auto --output-plan plan.json --report result.html
```

The dedicated system prompt is separate from the untrusted request. Generated
JSON is interpreted through the same restricted grammar as `qqa tex`; it is
never evaluated as Python. Local checks cover schema fields, bounded variable
domains, model-size quotas, expression syntax, scalar shape, and finite sample
values. The route and rationale are printed before execution.

### Input and review flags

Exactly one input source is required:

| Flag/input | What it does |
|---|---|
| `PROMPT` | Compile the quoted request |
| `-` | Read the request from standard input |
| `--file REQUEST.txt` | Read a UTF-8 request file |
| `--spec MODEL.json` | Plan or solve a reviewed `ModelSpec` without an API call |
| `--plan-only` | Stop after validation and route selection |
| `--show-model` | Print the validated model JSON |
| `--output-plan PLAN.json` | Save the audited model and routing explanation |

### Routing and execution flags

| Flag | Default | What it does |
|---|---|---|
| `--solver` | `auto` | `auto`, `qqa`, `qqa-scip`/`hybrid`, `scip`, `pareto`, or `blackbox` |
| `--device` / `--seed` | `auto` / `0` | Compute device and reproducibility seed |
| `--sol-size` / `--epochs` | `256` / `1500` | QQA or Pareto population and iterations |
| `--budget` / `--batch-size` / `--workers` | `96` / `8` / `4` | Black-box evaluation budget and concurrency |
| `--scip-time-limit` / `--scip-gap` | `60` / `0` | Total QQA+SCIP wall-clock budget and target gap |
| `--scip-threads` / `--scip-warm-starts` | `1` / `32` | SCIP threads and QQA primal starts |
| `--json` / `--output-result` | off / none | Print or save a machine-readable result |
| `--report` | none | Save an interactive HTML report |

`auto` routes multiple declared objectives to one-run parallel Pareto QQA.
A compatible single-objective symbolic model uses QQA→SCIP when the optional
backend is installed and QQA otherwise. Black-box intent can select the
budget-aware route when the request explicitly provides a safe objective
formula: the validated expression is evaluated point by point without
gradients. If the objective exists only in a simulator, external API, or
experiment, prose is insufficient—bind the real callable or service adapter
through `qqa.BlackBoxProblem`. QQA must not invent a missing evaluator. Use
`--plan-only`, inspect `notes`, and correct any material assumption before
execution.

The API key deliberately has no command-line flag, avoiding shell-history and
process-list exposure. Configure it as `QQA_LLM_API_KEY`. Endpoint and model
profiles use `QQA_LLM_BASE_URL` and `QQA_LLM_MODEL`, or the corresponding
`--api-base`, `--model`, and `--api-style` options. QQA embeds no
provider-specific endpoint or model default.

The full reviewed workflow is also available in the
[natural-language optimisation Colab](https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/12_natural_language_optimization_colab.ipynb).

## `qqa tex`

Translate a TeX optimisation problem through an OpenAI-compatible Responses
or Messages endpoint, validate the declarative model locally, and solve it:

```bash
export QQA_LLM_API_KEY='…'
export QQA_LLM_BASE_URL='https://api.example.com'
export QQA_LLM_MODEL='your-model-id'
qqa tex '\min_{x\in[-5,5]} (x-1.5)^2' \
  --device cuda --output-model model.json --report result.html
```

The key has no CLI flag by design, so it does not leak into shell history.
`--dry-run` validates without solving. `qqa tex --spec model.json` solves an
audited model offline. `--api-base`, `--model`, and `--api-style` configure a
compatible gateway; `--insecure` explicitly disables TLS verification.

TeX may also be piped through stdin:

```bash
printf '%s' '\min_{n\in\mathbb{Z},0\le n\le 10}(n-4)^2' | qqa tex -
```

Production-style file input and exact refinement:

```bash
qqa tex --file production-plan.tex --solver auto --device auto \
  --output-model audited-model.json --output-result result.json \
  --report result.html
```

`--solver auto` uses QQA→SCIP for a single objective when the `scip` extra is
installed. `--show-model` prints the strict intermediate JSON. Use
`--spec audited-model.json` for repeatable offline solving without an API key.

## `qqa example`

List and run packaged realistic applications:

```bash
qqa example list
qqa example run microgrid-dispatch --output-dir results/dispatch
qqa example run microgrid-pareto --device auto --output-dir results/pareto
qqa example run portfolio-pareto --device auto --output-dir results/portfolio
qqa example run process-blackbox --device auto --output-dir results/process
```

Each output directory contains machine-readable JSON plus CSV and/or a
self-contained interactive HTML report.

## `qqa doctor`

`qqa doctor` reports Python, Torch, CUDA/GPU, SCIP, PyG, Streamlit, Plotly, and
pandas capability. `qqa doctor --json` is suitable for support bundles and CI.

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

The whole CLI lives in `src/qqa/cli.py`. Read it
top-to-bottom — the structure is one `build_parser()` followed by one
`_cmd_<name>(args)` function per subcommand.
