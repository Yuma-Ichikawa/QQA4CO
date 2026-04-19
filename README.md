# Parallel Quasi-Quantum Annealer (PQQA)

A research-grade PyTorch toolkit for **continuous-relaxation combinatorial
optimisation**. PQQA unifies the recent line of work that *lifts* a
discrete CO problem into a continuous space and then solves it with
gradient descent — **PQQA** (ICLR 2025), the **CRA-PI-GNN** baseline
(NeurIPS 2024) and the **CPRA** diverse-solution framework (TMLR 2025) —
behind a single, GPU-friendly API. One installation gives you the PQQA
solver, an optional GNN backend that plugs in CRA-PI-GNN-style methods,
GPU-parallel **SA** and **Population Annealing** baselines for honest
comparisons, a 20+ class problem catalogue, a Streamlit dashboard and a CLI.

The package is published on PyPI as **`qqa`** for backwards compatibility
with earlier QQA4CO releases (``import qqa``).

<p align="center">
  <a href="https://pypi.org/project/qqa/"><img src="https://img.shields.io/pypi/v/qqa.svg?logo=pypi&logoColor=white&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/qqa/"><img src="https://img.shields.io/pypi/pyversions/qqa.svg?logo=python&logoColor=white" alt="Python versions"></a>
  <a href="https://github.com/Yuma-Ichikawa/QQA4CO/blob/main/LICENCE.txt"><img src="https://img.shields.io/pypi/l/qqa.svg" alt="License"></a>
  <a href="https://github.com/Yuma-Ichikawa/QQA4CO/actions/workflows/ci.yml"><img src="https://github.com/Yuma-Ichikawa/QQA4CO/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://yuma-ichikawa.github.io/QQA4CO/"><img src="https://img.shields.io/badge/docs-mkdocs--material-blue?logo=materialformkdocs&logoColor=white" alt="Documentation"></a>
  <a href="https://github.com/Yuma-Ichikawa/QQA4CO/discussions"><img src="https://img.shields.io/github/discussions/Yuma-Ichikawa/QQA4CO?logo=github&label=Discussions" alt="GitHub Discussions"></a>
  <a href="https://codecov.io/gh/Yuma-Ichikawa/QQA4CO"><img src="https://codecov.io/gh/Yuma-Ichikawa/QQA4CO/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://doi.org/10.5281/zenodo.19648231"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19648231-1f6feb?logo=doi&logoColor=white" alt="DOI"></a>
</p>

<p align="center">
  <a href="https://parallelquasiquantum4co.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit Community Cloud" height="40">
  </a>
  &nbsp;
  <a href="https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/00_colab_quickstart.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Quickstart in Colab" height="40">
  </a>
</p>

<h2 align="center">🌐 Try the Web dashboard — zero install, runs in your browser</h2>

<p align="center">
  <b><a href="https://parallelquasiquantum4co.streamlit.app/">→ parallelquasiquantum4co.streamlit.app</a></b>
  &nbsp;·&nbsp; hosted on Streamlit Community Cloud &nbsp;·&nbsp; CPU back-end &nbsp;·&nbsp; light / dark mode
</p>

<p align="center">
  <a href="https://parallelquasiquantum4co.streamlit.app/">
    <img src="https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/demo.gif" width="720" alt="QQA dashboard demo — click to open the live app">
  </a>
</p>

The dashboard is a four-page Streamlit app that drives the same `qqa.anneal`
solver this README documents — pick a problem, watch the relaxed variables
discretise live, race PQQA against simulated annealing, and inspect the
parallel population in 3D solution-space PCA / loss-spectrogram views.

| Page | What you can do |
| --- | --- |
| **Home** | Pick from 19 problems (MIS, Max-Cut, MaxClique, Vertex Cover, GraphBisection, MinDominatingSet, Coloring, BalancedGraphPartition, TSP, QAP, NQueens, Knapsack, NumberPartitioning, MaxSAT3, 1D Ising, Edwards–Anderson, SK, p-spin, RFIM, BinaryPerceptron, HopfieldMemory). Auxiliary sliders are problem-aware. |
| **Solve** | Run PQQA / CRA-PI-GNN / CPRA with live progress, polish (1-flip local search) and warm-start toggles, and a per-problem solution viewer (TSP tour, NQueens board, highlighted IS, colouring, …). |
| **Visualize** | 10 tabs of post-hoc plots — **solution-space PCA** of the final parallel population (3D, replicas coloured by loss, global best highlighted), **loss spectrogram** over time, diversity, replica fate, schedule, ridgeline. |
| **Compare** | Hyper-parameter grid sweep with parallel-coordinates view, **and** a head-to-head **PQQA vs. SA vs. Population-Annealing shootout** that reports the wall-clock speed-up at matched compute budget. |

### Or run it locally — same UI, your hardware

```bash
pip install "qqa[gui]"        # pulls Streamlit + Plotly
qqa gui                       # opens http://localhost:8501
```

Or with uv:

```bash
uv sync --extra gui
uv run qqa gui
```

CUDA is picked up automatically when available.

---

## What's in the box

1. **The PQQA solver** (`qqa.anneal`) — a parallel, population-based annealer
   that lifts any QUBO / Ising / categorical / permutation problem into a
   continuous relaxation and minimises it with gradient descent + diversity
   regularisation. CPU / CUDA / MPS, deterministic with `qqa.fix_seed`.
2. **A 20+ class problem catalogue** spanning QUBO graphs (MIS, Max-Cut,
   MaxClique, Vertex Cover, Graph Bisection, Minimum Dominating Set),
   classic CO (Knapsack, NumberPartitioning, MaxSAT3), permutation problems
   (TSP, QAP, NQueens), categorical/assignment (graph Coloring, Balanced
   K-way Partition), spin glasses (1D Ising, Edwards-Anderson 2D/3D, SK,
   p-spin, Random-Field Ising) and statistical-physics models
   (BinaryPerceptron, HopfieldMemory). Every class implements the same
   `loss_fn` / `score_summary` contract, so the solver, CLI and dashboard
   are completely problem-agnostic.
3. **An optional GNN backend** (`qqa.pignn`, install with `qqa[pignn]`) that
   plugs **GNN-based unsupervised-learning CO solvers** into the same API.
   PQQA, the **CRA-PI-GNN** baseline (NeurIPS 2024) and the **CPRA**
   diverse-solution framework (TMLR 2025) all share `AnnealResult`,
   `score_summary`, the same problem builders and the same CLI flags — A/B
   comparing methods is one `--backend` switch away.
4. **GPU-parallel MCMC baselines** for honest head-to-head comparisons —
   classical **Simulated Annealing** (`qqa.simulated_annealing`) with a
   QUBO Glauber fast path, and **Population Annealing**
   (`qqa.population_annealing`, Hukushima-Iba / Machta) with systematic or
   multinomial resampling between temperature steps. Both expose the same
   `best_sol` / `best_obj` / `history` surface as `qqa.anneal`, so the
   Streamlit *Compare* page can race PQQA against either at a matched
   compute budget.
5. **A polished Streamlit dashboard** (light / dark, live progress, parallel
   population view, per-problem solution viz, hyper-parameter sweeps) and a
   `qqa` **CLI** (`solve / bench / gui / version`) for reproducible
   experiments. A hosted instance lives at
   <https://parallelquasiquantum4co.streamlit.app/>.
6. **MkDocs + Material reference docs** with auto-generated API pages, a
   runnable `examples/` notebook gallery (Open-in-Colab badges) and a
   `scripts/verify_all_problems.py` correctness sweep that benchmarks every
   problem against ground truth or a strong baseline (29 / 29 instances pass
   in the latest sweep — see [`docs/verification.md`](https://github.com/Yuma-Ichikawa/QQA4CO/blob/main/docs/verification.md)).

## Reference papers

QQA4CO implements and unifies the following peer-reviewed work:

1. **Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling**
   — Yuma Ichikawa.
   *International Conference on Learning Representations (ICLR), 2025.*
   [[OpenReview](https://openreview.net/forum?id=9EfBeXaXf0)]
   The headline **PQQA** algorithm — parallel quasi-quantum annealing with
   gradient-based sampling. Implemented as `qqa.anneal`.

2. **Continuous Parallel Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems**
   — Yuma Ichikawa, Hiroaki Iwashita.
   *Transactions on Machine Learning Research (TMLR), 2025.*
   [[OpenReview](https://openreview.net/forum?id=ix33zd5zCw)]
   The **CPRA** framework — generates penalty- and structure-diversified
   solutions from a single training run.

3. **Controlling Continuous Relaxation for Combinatorial Optimization**
   — Yuma Ichikawa.
   *Neural Information Processing Systems (NeurIPS), 2024.*
   [[OpenReview](https://openreview.net/forum?id=ykACV1IhjD)]
   · [[NeurIPS poster](https://neurips.cc/virtual/2024/poster/92998)]
   The **CRA-PI-GNN** GNN-based unsupervised-learning solver. Ported to
   PyTorch Geometric and exposed under `qqa.pignn`.

---

## Install

With [uv](https://github.com/astral-sh/uv) (recommended for development):

```bash
git clone https://github.com/Yuma-Ichikawa/QQA4CO.git && cd QQA4CO
uv sync                                            # core only
uv sync --extra plotly --extra gui --extra dev     # everything
uv run pytest -q                                   # sanity check
```

With pip:

```bash
pip install qqa                # core
pip install "qqa[plotly]"      # + interactive plots
pip install "qqa[gui]"         # + Streamlit dashboard
pip install "qqa[all]"         # everything
```

## Quickstart

```python
import networkx as nx
import qqa

qqa.fix_seed(0)
g = nx.random_regular_graph(d=3, n=100, seed=0)
problem = qqa.MaximumIndependentSet(g, penalty=2)
result = qqa.anneal(problem, sol_size=100, num_epochs=1500)
print(f"MIS size: {-int(result.best_obj)}  in {result.runtime:.2f}s")
```

The same call style applies to spin systems:

```python
problem = qqa.SherringtonKirkpatrick(N=100, seed=0)
result = qqa.anneal(problem, sol_size=200, num_epochs=2000, verbose=False)
print(f"E_0 / N  ≈  {result.best_obj / 100:.4f}   (target ≈ -0.7632)")
```

## Optional CRA-PI-GNN backend (PyTorch Geometric)

QQA4CO ships an **optional** PyTorch Geometric port of the **CRA-PI-GNN**
solver from Ichikawa, NeurIPS 2024 — *"Controlling Continuous Relaxation
for Combinatorial Optimization"* ([paper](https://openreview.net/forum?id=ykACV1IhjD),
[reference DGL implementation](https://github.com/Yuma-Ichikawa/CRA4CO)).
This lets you compare against CRA-PI-GNN from the same codebase as QQA, on
hardware that the original DGL stack does not yet target (e.g. NVIDIA
Blackwell B200 / `sm_100`).

Install with the `pignn` extra (pulls in `torch-geometric`):

```bash
pip install "qqa[pignn]"
```

Python API — drop-in alternative to `qqa.anneal`, returns the same `AnnealResult`:

```python
import networkx as nx
import qqa
from qqa.pignn import train_cra_pi_gnn

qqa.fix_seed(0)
g = nx.random_regular_graph(d=3, n=200, seed=0)
problem = qqa.MaximumIndependentSet(g, penalty=2)

result = train_cra_pi_gnn(
    problem,
    learning_rate=1e-3,
    init_reg_param=-2.0,
    annealing_rate=5e-4,
    num_epochs=5000,
)
print(result.score)   # {'label': 'IS size', 'value': ..., 'feasible': True, ...}
```

CLI — same problem builders, just add `--backend pignn`:

```bash
qqa solve --problem mis --size 200 --backend pignn \
          --learning-rate 1e-3 --pignn-init-reg-param -2 \
          --pignn-annealing-rate 5e-4 --epochs 5000
```

Supported problems for the PyG backend: `mis`, `maxcut`, `maxclique`,
`vertex_cover`, `graph_bisection` (anything QUBO-on-a-graph). For spin
glasses, TSP, perceptron, etc. use the default `qqa.anneal`.

### CPRA — diverse solutions in a single run (TMLR 2025)

The same `qqa[pignn]` extra also installs **CPRA** (Continual Parallel
Relaxation Annealing) from Ichikawa & Iwashita, *TMLR 2025* —
*"Continuous Parallel Relaxation for Finding Diverse Solutions in
Combinatorial Optimization Problems"*
([paper](https://openreview.net/forum?id=ix33zd5zCw),
[reference repo](https://github.com/Yuma-Ichikawa/CPRA4CO)). CPRA shares
QQA4CO's GCN backbone with CRA-PI-GNN but exposes `R` parallel output
heads, so a single training run produces `R` diverse continuous
solutions. Two diversification regimes are supported out of the box:

- **Penalty diversification** — pass one problem instance per replica
  (e.g. MIS with different penalty weights) to sweep a hyperparameter
  in one shot instead of training `R` separate models.
- **Variation diversification** — leave `replica_problems=None` and set
  `vari_param > 0` to add the inter-replica diversity term
  `-R · Σᵢ stdᵣ(p_{i,r})`, pulling replicas apart on the same problem.

Python API:

```python
import networkx as nx
import qqa
from qqa.pignn import train_cpra_pi_gnn

qqa.fix_seed(0)
g = nx.random_regular_graph(d=3, n=200, seed=0)
base = qqa.MaximumIndependentSet(g, penalty=2)

# Penalty diversification: 4 replicas, one per penalty level.
penalties = [1.5, 2.0, 2.5, 3.0]
replica_problems = [qqa.MaximumIndependentSet(g, penalty=p) for p in penalties]

result = train_cpra_pi_gnn(
    base,
    num_replicas=len(replica_problems),
    replica_problems=replica_problems,
    learning_rate=1e-3,
    init_reg_param=-2.0,
    annealing_rate=5e-4,
    num_epochs=5000,
)

# Inspect every replica, not only the best one.
for record in result.score["extra"]["replicas"]:
    r, score = record["replica"], record["score"]
    print(f"replica {r}: |IS|={score['value']}  feasible={score['feasible']}")
```

Variation diversification on a fixed problem is the same call without
`replica_problems` and a positive `vari_param`:

```python
result = train_cpra_pi_gnn(
    qqa.MaxCut(g),
    num_replicas=4,
    vari_param=0.4,            # encourages between-replica spread
    learning_rate=1e-3,
    init_reg_param=-2.0,
    annealing_rate=5e-4,
    num_epochs=5000,
)
```

CLI — same problem builders, add `--backend cpra`:

```bash
qqa solve --problem mis --backend cpra --size 200 \
          --cpra-num-replicas 4 \
          --cpra-penalty-levels 1.5,2.0,2.5,3.0 \
          --learning-rate 1e-3 --pignn-init-reg-param -2 \
          --pignn-annealing-rate 5e-4 --epochs 5000
```

`--cpra-penalty-levels` is currently supported for `--problem mis` and
`--problem vertex_cover` (the QUBO classes that accept a `penalty=...`
constructor kwarg). For variation diversification on any other graph
problem, drop `--cpra-penalty-levels` and pass `--cpra-vari-param 0.4`
instead. All `--pignn-*` flags above (`--pignn-init-reg-param`,
`--pignn-annealing-rate`, `--pignn-tol`, `--pignn-patience`,
`--pignn-hidden`, `--pignn-no-annealing`) apply unchanged to the CPRA
backend.

The returned `AnnealResult` carries the **best replica's** discrete
solution in `best_sol` / `best_obj`; every replica's solution and score
are stored in `result.score["extra"]["replicas"]`, and per-epoch
per-replica objectives in `result.history["per_replica_obj"]` (shape
`(epochs, R)`) for downstream plotting.

### When to use which

| Use case                                                              | Recommended                              |
| --------------------------------------------------------------------- | ---------------------------------------- |
| Most CO and spin-glass problems (default)                             | `qqa.anneal`                             |
| Reproducing the NeurIPS 2024 CRA-PI-GNN paper                         | `qqa.pignn.train_cra_pi_gnn`             |
| Reproducing the TMLR 2025 CPRA paper / sweep penalties in one run     | `qqa.pignn.train_cpra_pi_gnn`            |
| Spin glasses, perceptron, Hopfield, TSP, coloring                     | `qqa.anneal` (PyG backend not supported) |
| Need parallel replicas + diversity term (gradient-based sampler)      | `qqa.anneal`                             |
| Need a single deterministic GNN solver for ablation comparison        | `qqa.pignn.train_cra_pi_gnn`             |
| Need diverse GNN solutions (penalty- or variation-diversified) in one run | `qqa.pignn.train_cpra_pi_gnn`        |

### Empirical comparison (CPU, single thread)

Both solvers given the same instance and seed; QQA uses default 100 parallel
replicas, CRA-PI-GNN uses 5000 epochs of single-replica GCN training. Numbers
from `scripts/bench_qqa_vs_pignn.py` on the bundled CPU runner:

| Instance              | `qqa.anneal` (IS, runtime) | `qqa.pignn.train_cra_pi_gnn` (IS, runtime) |
| --------------------- | -------------------------- | ------------------------------------------ |
| MIS, N=100, d=3-reg   | **43**, **5.9 s**          | 2†, 12.7 s                                 |
| MIS, N=300, d=3-reg   | **125**, **8.1 s**         | 126, 201 s                                 |
| MIS, N=500, d=20-reg  | **29**, **8.2 s**          | 1†, 414 s                                  |

† CRA-PI-GNN collapsed to a near-trivial solution under the README's
"medium-graph" hyperparameters. The paper's headline numbers use a
larger `init_reg_param` (e.g. -20) and longer schedule for `N >= 1000`;
small / dense graphs need per-instance retuning. QQA is robust to
hyperparameter choice across all three rows.

**Takeaways**

- For raw quality and wall-clock speed, **`qqa.anneal` is the recommended
  default**. Its parallel-replica diversity makes it far less hyperparameter-
  sensitive than CRA-PI-GNN.
- CRA-PI-GNN is included so users can A/B against the paper from one
  installation and one device. On its native large-graph regime
  (`N >= 1000`, `d` low, paper defaults) it produces competitive solutions —
  but see the original [CRA4CO repository](https://github.com/Yuma-Ichikawa/CRA4CO)
  for the canonical DGL implementation.

## Problem catalog

| Category                   | Classes                                                                          |
| -------------------------- | -------------------------------------------------------------------------------- |
| Binary QUBO                | `MaximumIndependentSet`, `MaxClique`, `MaxCut` (+ `*Instance` batched variants)  |
| Binary (classic CO)        | `Knapsack`, `NumberPartitioning`, `VertexCover`, `GraphBisection`, `MaxSAT3`     |
| Categorical                | `Coloring`, `BalancedGraphPartition`                                             |
| Categorical (permutation)  | `TSP`, `QAP`, `NQueens`                                                          |
| 1D Ising                   | `Ising1D`                                                                        |
| Spin glass                 | `EdwardsAnderson`, `SherringtonKirkpatrick`                                      |
| Statistical physics        | `BinaryPerceptron`, `HopfieldMemory`                                             |

Every class implements `score_summary(x_disc) -> dict` so the CLI and GUI can
report a human-readable metric (`"IS size: 22"`, `"packed value: 358"`,
`"tour length: 3.28"`) and a feasibility flag alongside the raw loss. Full
mathematical definitions live in [`docs/problems.md`](https://github.com/Yuma-Ichikawa/QQA4CO/blob/main/docs/problems.md).

## Command-line interface

```bash
qqa version
qqa solve --problem sk  --size 100 --sol-size 128 --epochs 1000
qqa solve --problem mis --graph-file mygraph.gpickle --epochs 1500
qqa bench --preset er-small --epochs 500
qqa gui                                  # opens http://localhost:8501
```

Run `qqa <command> --help` for the full option list.

## Streamlit dashboard

```bash
pip install "qqa[gui]" && qqa gui
```

| Page          | Purpose                                                                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Home**      | Pick a problem family (Graph, Classic CO, Categorical/permutation, Physics), size, seed, and problem-specific parameters.                                        |
| **Solve**    | Configure QQA hyper-parameters, launch a run with a live progress bar, mean ± σ loss band over parallel replicas, population heatmap, diversity curve, score card. |
| **Visualize** | Tabbed views: dynamics, best trajectory, schedule, solution heatmap, parallel population, PCA trajectory, ridgeline, per-replica fate.                           |
| **Compare**   | Sweep a small hyper-parameter grid; inspect with parallel coordinates and overlaid trajectories.                                                                 |

A light/dark toggle lives in the sidebar; both themes share an academic,
Plotly-aware palette. A hosted instance runs at
**<https://parallelquasiquantum4co.streamlit.app/>**.

<details>
<summary><b>Deploy your own (free)</b></summary>

The repository ships everything Streamlit Community Cloud needs:
[`requirements.txt`](requirements.txt) (CPU-only PyTorch),
[`runtime.txt`](runtime.txt) (Python pin),
[`.streamlit/config.toml`](.streamlit/config.toml) (theme + telemetry off).

1. Sign in at <https://share.streamlit.io> with GitHub.
2. **New app** → repository `Yuma-Ichikawa/QQA4CO`, branch `main`,
   main file `app/streamlit_app.py`, then **Deploy**.
3. In the app's `⋮` → **Settings** → **Sharing**, set
   *"Who can view this app?"* to **"Anyone with the link can view"**.
   Without this every visitor is redirected to Streamlit SSO.

Re-deploys happen automatically on every push to `main`. The full runbook,
common failure modes, and the health-check endpoint live in
[`deploy/STREAMLIT_DEPLOY.md`](https://github.com/Yuma-Ichikawa/QQA4CO/blob/main/deploy/STREAMLIT_DEPLOY.md). Verify with:

```bash
uv run python scripts/check_streamlit_deploy.py
```

The custom-problem editor is **off by default** on public deployments
(it evaluates arbitrary Python via `exec`). Re-enable it on a trusted
machine with `QQA_ALLOW_CUSTOM=1 uv run qqa gui`.

</details>

<details>
<summary><b>Other free / cheap targets</b></summary>

The repository drops onto any of the usual platforms unchanged:

- **Hugging Face Spaces** (Streamlit SDK) — persistent URL, free CPU tier, HTTPS by default.
- **Fly.io / Render** — Docker-based; entry point `app/streamlit_app.py`, deps `requirements.txt`.
- **Google Cloud Run** — container image, pay-per-request.

Each platform issues a permanent HTTPS URL out of the box.

</details>

## Visualization

```python
from qqa import visualization as viz

viz.plot_history(result)                       # loss / penalty / diversity
viz.plot_best_trajectory(result, backend="plotly")
viz.plot_schedule(qqa.LinearBGSchedule(-2, 0.1), num_epochs=2000)
viz.plot_run_comparison([r1, r2, r3], labels=["lr=1", "lr=0.5", "lr=2"])
viz.plot_solution_heatmap(result, problem)
```

Every helper accepts `backend="matplotlib"` (default) or `backend="plotly"`.

<table>
  <thead>
    <tr>
      <th align="center">Dynamics</th>
      <th align="center">Best trajectory</th>
      <th align="center">Best solution</th>
      <th align="center">Parallel population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/history_sk.png" width="240" alt="SK dynamics"></td>
      <td><img src="https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/best_sk.png"    width="240" alt="SK best trajectory"></td>
      <td><img src="https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/solution_sk.png" width="240" alt="SK best solution"></td>
      <td><img src="https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/population_sk.png" width="240" alt="SK parallel population"></td>
    </tr>
  </tbody>
</table>

<sub>Sherrington–Kirkpatrick spin glass (N=80) — the same helpers work for
every catalog problem.</sub>

The full per-problem gallery (MIS, Max-Cut, coloring, Ising 1D,
Edwards–Anderson, SK, binary perceptron, Hopfield) is in
[`docs/visualization.md`](https://github.com/Yuma-Ichikawa/QQA4CO/blob/main/docs/visualization.md). Regenerate the figures
with `uv run python scripts/make_gallery.py`.

## Verified correctness

We benchmark QQA against ground truth or a strong baseline for every problem
in the catalog via `scripts/verify_all_problems.py`. The most recent sweep
(29 instances across 9 problem families) lives in
[`docs/verification.md`](https://github.com/Yuma-Ichikawa/QQA4CO/blob/main/docs/verification.md):

| Problem                       | Instances              | Reference                       | QQA                                 |
| ----------------------------- | ---------------------- | ------------------------------- | ----------------------------------- |
| Maximum Independent Set       | 3 × (3-reg, N=50)      | networkx degree-greedy          | matches or beats greedy on all seeds |
| MaxCut                        | 3 × ER (N=30/40/60)    | best-of-400 random partition    | +6 / +16 / +27 edges over random    |
| MaxClique                     | 3 × ER (N=30/40/50)    | `nx.approximation.max_clique`   | +1 vertex on every seed             |
| Graph coloring (K=3)          | 3 × (3-reg, N=40)      | Welsh–Powell greedy             | 0 conflicts on all seeds            |
| Ising 1D ferromagnet          | N ∈ {16, 32, 64}       | exact E₀ = −N                  | gap = 0 on every size               |
| Edwards–Anderson 2D, L=3      | 3 seeds                | brute force (2⁹)                | matches exact ground state          |
| Edwards–Anderson 3D, L=4      | 2 seeds                | —                               | E/N ≈ −1.61 (no exact solver)       |
| Sherrington–Kirkpatrick       | N ∈ {50, 100, 200}     | Parisi e₀ = −0.7632             | ≤ 3.2 % gap at N=200                |
| Binary perceptron             | α ∈ {0.3, 0.5, 0.7}    | teacher reaches 0 errors        | 0 errors on all α                   |
| Hopfield memory               | (N, P) ∈ {(32,2),(64,3),(128,4)} | ≥ 0.95 overlap        | overlap = 1.0                       |

Overall: **29 / 29 checks pass (100 %)**. Re-run with
`uv run python scripts/verify_all_problems.py` to regenerate the report
in place.

## Notebooks

Nine runnable notebooks live in [`examples/`](https://github.com/Yuma-Ichikawa/QQA4CO/tree/main/examples). Each carries an
**Open in Colab** badge in its first cell and auto-installs `qqa`.

| #   | Notebook                                  |
| --- | ----------------------------------------- |
| 0   | `00_colab_quickstart.ipynb` — one-click tour of every problem |
| 1   | `01_maximum_independent_set.ipynb`        |
| 2   | `02_graph_coloring.ipynb`                 |
| 3   | `03_max_cut.ipynb`                        |
| 4   | `04_edwards_anderson_3d.ipynb`            |
| 5   | `05_sherrington_kirkpatrick.ipynb`        |
| 6   | `06_binary_perceptron.ipynb`              |
| 7   | `07_hopfield_memory.ipynb`                |
| 8   | `08_parallel_benchmark.ipynb`             |

Regenerate them deterministically with
`uv run python scripts/_generate_notebooks.py`.

## Documentation

```bash
uv run mkdocs serve            # http://127.0.0.1:8000
uv run mkdocs build --strict   # produces site/
```

The site covers the quickstart, full problem catalog with mathematical
definitions, GUI walk-through, visualization guide, auto-generated API
reference, and a migration guide from 0.2.x.

## Scripts

| Script                            | Purpose                                       |
| --------------------------------- | --------------------------------------------- |
| `scripts/demo_mis.py`             | Minimal MIS end-to-end demo                   |
| `scripts/demo_coloring.py`        | 3-coloring end-to-end demo                    |
| `scripts/demo_parallel.py`        | Parallel instances of MIS                     |
| `scripts/demo_pignn_mis.py`       | MIS via the optional CRA-PI-GNN backend (PyG) |
| `scripts/bench_er_small.py`       | Benchmark on the bundled ER-small MIS dataset |
| `scripts/bench_qqa_vs_pignn.py`   | QQA vs. CRA-PI-GNN comparison (README table)  |
| `scripts/make_gallery.py`         | Regenerate the figures used in the README     |
| `scripts/verify_all_problems.py`  | Run the catalog-wide correctness sweep        |
| `scripts/_generate_notebooks.py`  | Regenerate the shipped example notebooks      |

Run any script via `uv run python scripts/<name>.py`.

## Notebooks

| Notebook                              | Purpose                                                                                                                       |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `notebooks/cra_pignn_example.ipynb`   | Walkthrough of the optional **CRA-PI-GNN** (PyTorch Geometric) backend on all five supported graph problems (MIS, MaxCut, MaxClique, VertexCover, GraphBisection) with side-by-side `qqa.anneal` runs. Requires the `pignn` extra. |

## Repository layout

```
QQA4CO/
├── src/qqa/          # importable package (annealing, problems, viz, ...)
│   └── problems/     # qubo.py, categorical.py, spin.py, extras.py, user.py
├── app/              # Streamlit dashboard (streamlit_app.py + pages/)
├── docs/             # MkDocs site sources
├── examples/         # 9 example notebooks
├── scripts/          # demo, benchmark, verification, gallery scripts
├── tests/            # pytest suite
├── data/             # bundled datasets and gallery figures
├── pyproject.toml
└── README.md
```

## Contributing

Issues and pull requests are welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, style, and test commands.

## License

BSD-3-Clause — see [`LICENCE.txt`](LICENCE.txt).

## Cite

If you use the **QQA4CO software** in your research (regardless of which
backend), please cite the software via its concept DOI alongside the
companion paper:

```bibtex
@software{qqa4co_software,
  title   = {QQA4CO: Quasi-Quantum Annealing for Combinatorial Optimization},
  author  = {Ichikawa, Yuma and Arai, Yamato},
  year    = {2026},
  version = {0.5.0},
  doi     = {10.5281/zenodo.19648231},
  url     = {https://github.com/Yuma-Ichikawa/QQA4CO}
}
```

If you use the **PQQA** solver (`qqa.anneal`), please also cite:

```bibtex
@inproceedings{ichikawa2025pqqa,
  title     = {Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling},
  author    = {Ichikawa, Yuma and Arai, Yamato},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=9EfBeXaXf0}
}
```

If you use the **CPRA** diverse-solution framework, please cite:

```bibtex
@article{ichikawa2025cpra,
  title   = {Continuous Parallel Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems},
  author  = {Ichikawa, Yuma and Iwashita, Hiroaki},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2025},
  url     = {https://openreview.net/forum?id=ix33zd5zCw}
}
```

If you use the optional **CRA-PI-GNN** backend (`qqa.pignn`), please **also**
cite the original paper and the reference DGL implementation it was ported
from:

```bibtex
@inproceedings{ichikawa2024controlling,
  title     = {Controlling Continuous Relaxation for Combinatorial Optimization},
  author    = {Ichikawa, Yuma},
  booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  url       = {https://openreview.net/forum?id=ykACV1IhjD}
}
```

Reference implementation: <https://github.com/Yuma-Ichikawa/CRA4CO>.
