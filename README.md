# QQA — Quasi-Quantum Annealing

PyTorch implementation of the ICLR 2025 paper
**[Continuous Tensor Relaxation for Finding Diverse Solutions in Combinatorial Optimization](https://openreview.net/forum?id=9EfBeXaXf0)**
by Yuma Ichikawa and Yamato Arai.

<p align="center">
  <a href="https://pypi.org/project/qqa/"><img src="https://img.shields.io/pypi/v/qqa.svg?logo=pypi&logoColor=white&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/qqa/"><img src="https://img.shields.io/pypi/pyversions/qqa.svg?logo=python&logoColor=white" alt="Python versions"></a>
  <a href="https://github.com/Yuma-Ichikawa/QQA4CO/blob/main/LICENCE.txt"><img src="https://img.shields.io/pypi/l/qqa.svg" alt="License"></a>
  <a href="https://github.com/Yuma-Ichikawa/QQA4CO/actions/workflows/ci.yml"><img src="https://github.com/Yuma-Ichikawa/QQA4CO/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/00_colab_quickstart.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Quickstart in Colab">
  </a>
  <a href="https://parallelquasiquantum4co.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
</p>

<p align="center">
  <img src="data/fig/demo.gif" width="420" alt="QQA dashboard demo">
</p>

QQA relaxes a discrete problem to a continuous, differentiable objective and
anneals towards a discrete minimum using gradient-based sampling. A single
loop handles classical combinatorial problems (MIS, Max-Cut, coloring, TSP,
…) and statistical-physics spin systems (Ising, Edwards–Anderson, SK, binary
perceptron, Hopfield).

**Highlights**

- One unified API — `qqa.anneal(problem, ...)` for every problem.
- Rich problem catalog out of the box: 17 classes across 7 categories.
- Matplotlib and Plotly backends; Plotly is optional.
- CLI: `qqa solve / bench / gui / version`.
- Streamlit dashboard with live progress, sweeps, and an academic light/dark theme.
- MkDocs + Material documentation with auto-generated API reference.

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
mathematical definitions live in [`docs/problems.md`](docs/problems.md).

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
[`deploy/STREAMLIT_DEPLOY.md`](deploy/STREAMLIT_DEPLOY.md). Verify with:

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
      <td><img src="data/fig/gallery/history_sk.png" width="240" alt="SK dynamics"></td>
      <td><img src="data/fig/gallery/best_sk.png"    width="240" alt="SK best trajectory"></td>
      <td><img src="data/fig/gallery/solution_sk.png" width="240" alt="SK best solution"></td>
      <td><img src="data/fig/gallery/population_sk.png" width="240" alt="SK parallel population"></td>
    </tr>
  </tbody>
</table>

<sub>Sherrington–Kirkpatrick spin glass (N=80) — the same helpers work for
every catalog problem.</sub>

The full per-problem gallery (MIS, Max-Cut, coloring, Ising 1D,
Edwards–Anderson, SK, binary perceptron, Hopfield) is in
[`docs/visualization.md`](docs/visualization.md). Regenerate the figures
with `uv run python scripts/make_gallery.py`.

## Verified correctness

We benchmark QQA against ground truth or a strong baseline for every problem
in the catalog via `scripts/verify_all_problems.py`. The most recent sweep
(29 instances across 9 problem families) lives in
[`docs/verification.md`](docs/verification.md):

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

Nine runnable notebooks live in [`examples/`](examples/). Each carries an
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
| `scripts/bench_er_small.py`       | Benchmark on the bundled ER-small MIS dataset |
| `scripts/make_gallery.py`         | Regenerate the figures used in the README     |
| `scripts/verify_all_problems.py`  | Run the catalog-wide correctness sweep        |
| `scripts/_generate_notebooks.py`  | Regenerate the shipped example notebooks      |

Run any script via `uv run python scripts/<name>.py`.

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

```bibtex
@inproceedings{ichikawa2025qqa,
  title     = {Continuous Tensor Relaxation for Finding Diverse Solutions in Combinatorial Optimization},
  author    = {Ichikawa, Yuma and Arai, Yamato},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=9EfBeXaXf0}
}
```
