# QQA — Quasi-Quantum Annealing

PyTorch implementation of the ICLR 2025 paper
**[Continuous Tensor Relaxation for Finding Diverse Solutions in Combinatorial Optimization](https://openreview.net/forum?id=9EfBeXaXf0)**
by Yuma Ichikawa and Yamato Arai.

<p align="center">
  <a href="https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/00_colab_quickstart.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Quickstart in Colab">
  </a>
  <a href="https://parallelquasiquantum4co.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
</p>

<p align="center">
  <img src="data/fig/demo.gif" width="400">
</p>

**QQA** relaxes a discrete problem to a continuous, differentiable objective
and anneals towards a discrete minimum using gradient-based sampling. The
same loop handles combinatorial problems (MIS, Max-Cut, coloring, …) and
physics-flavoured spin problems (Ising, Edwards-Anderson, SK, binary
perceptron, Hopfield memory).

Highlights:

- **Unified Python API**: one `qqa.anneal()` for every problem.
- **Rich problem catalog** out of the box (10+ classes).
- **Interactive visualization**: matplotlib by default, Plotly optional.
- **CLI**: `qqa solve`, `qqa bench`, `qqa gui`, `qqa version`.
- **Streamlit GUI**: browser dashboard with live progress and sweep tools.
- **Docs site**: MkDocs + Material + auto API reference.

---

## Install

### With [uv](https://github.com/astral-sh/uv) (recommended)

```bash
git clone https://github.com/Yuma-Ichikawa/QQA4CO.git
cd QQA4CO
uv sync                                               # core only
uv sync --extra plotly --extra gui --extra dev        # with extras
uv run pytest -q                                      # sanity check
```

### With pip

```bash
pip install qqa                   # core
pip install "qqa[plotly]"         # + interactive plots
pip install "qqa[gui]"            # + Streamlit dashboard
pip install "qqa[all]"            # everything
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

The same call style applies to spin problems:

```python
problem = qqa.SherringtonKirkpatrick(N=100, seed=0)
result = qqa.anneal(problem, sol_size=200, num_epochs=2000, verbose=False)
print(f"E_0 / N  ≈  {result.best_obj / 100:.4f}  (target ≈ -0.7632)")
```

## Problem catalog

| Category          | Classes |
| ----------------- | ------- |
| Binary QUBO       | `MaximumIndependentSet`, `MaxClique`, `MaxCut` (+ `*Instance` batched variants) |
| Binary (classic CO) | `Knapsack`, `NumberPartitioning`, `VertexCover`, `GraphBisection`, `MaxSAT3` |
| Categorical       | `Coloring`, `BalancedGraphPartition` |
| Categorical (permutation) | `TSP`, `QAP`, `NQueens` |
| 1D Ising          | `Ising1D` |
| Spin glass        | `EdwardsAnderson`, `SherringtonKirkpatrick` |
| Statistical phys. | `BinaryPerceptron`, `HopfieldMemory` |

Every problem exposes `problem.score_summary(x_disc) -> dict` so the CLI /
GUI can display a human-readable metric (e.g. "IS size: 22", "packed
value: 358", "tour length: 3.28") and a feasibility flag alongside the
raw loss.

Read the full mathematical definitions in
[`docs/problems.md`](docs/problems.md).

## Command-line interface

```bash
qqa version
qqa solve --problem sk --size 100 --sol-size 128 --epochs 1000
qqa solve --problem mis --graph-file mygraph.gpickle --epochs 1500
qqa bench --preset er-small --epochs 500
qqa gui                                  # open http://localhost:8501
```

Run `qqa <command> --help` for the full option list.

## Streamlit GUI

```bash
pip install "qqa[gui]"
qqa gui
```

The dashboard has four pages:

- **Home** — pick a problem family (Graph, Classic CO, Categorical /
  permutation, Physics), size, seed, and problem-specific parameters.
- **Solve** — set QQA hyper-parameters and launch a run with a live
  progress bar, a `mean ± σ` loss band across the parallel replicas, a
  population heatmap sorted by best-so-far, a population-diversity curve,
  and a headline score card (e.g. "IS size: 22  /  40").
- **Visualize** — tabbed view of dynamics, best trajectory, the applied
  annealing schedule, a solution heatmap, parallel population, PCA
  trajectory, ridgeline of loss distributions, and per-replica fate
  lines.
- **Compare** — run a small hyper-parameter grid and inspect the result
  with parallel-coordinates and overlaid trajectories.

A light / dark toggle lives in the sidebar; both themes share an
academic, Plotly-aware palette.

### Live demo

A hosted instance runs at
**<https://parallelquasiquantum4co.streamlit.app/>**. The operator runbook,
including how to switch the Streamlit Community Cloud app from *Private* to
*Anyone with the link*, is in
[`deploy/STREAMLIT_DEPLOY.md`](deploy/STREAMLIT_DEPLOY.md). A quick health
check is available via:

```bash
uv run python scripts/check_streamlit_deploy.py
```

> **Note.** If the URL currently redirects to `/-/auth/app?…`, the app is
> still set to Private on Streamlit Community Cloud. The fix is a single
> setting in the Streamlit Cloud dashboard — see
> [`deploy/STREAMLIT_DEPLOY.md §1`](deploy/STREAMLIT_DEPLOY.md#1-why-is-the-url-redirecting-to--auth-app).

## Deploy to the public web (free)

The dashboard can be published for free via **Streamlit Community Cloud**
(or Hugging Face Spaces) and bridged from any domain you already own
(Xserver / お名前ドットコム / Cloudflare / …).

### 1. Deploy to Streamlit Community Cloud

The repository already ships the two files needed by Community Cloud:

- [`requirements.txt`](requirements.txt) — CPU-only PyTorch pin plus the
  minimal runtime. Installs this repo as the `qqa` package via the
  trailing `.`.
- [`.streamlit/config.toml`](.streamlit/config.toml) — on-brand dark
  theme and telemetry off.

Then:

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → Repository `Yuma-Ichikawa/QQA4CO`, Branch `main`,
   Main file path `app/streamlit_app.py`.
3. Click **Deploy**. Your app will be served at
   `https://<something>.streamlit.app` after a 3–5 min build.

The custom-problem editor is **off by default** on public deployments
(it evaluates arbitrary Python via `exec`). Re-enable it on a trusted
machine with:

```bash
QQA_ALLOW_CUSTOM=1 uv run qqa gui
```

### 2. Point your own domain at the app

If you own a domain on a shared rental host (e.g. Xserver), drop this
snippet into the `public_html/` of the subdomain that you want to use
(adjust the target URL):

```apache
RewriteEngine On
RewriteRule ^(.*)$ https://qqa4co.streamlit.app/$1 [R=301,L]
```

A ready-to-copy template with both `301` and `iframe` variants lives at
[`deploy/xserver-htaccess.example`](deploy/xserver-htaccess.example).

### Other targets

The repository is portable enough to drop onto any of the usual
platforms: Hugging Face Spaces (Streamlit SDK), Fly.io / Render
(Docker), Google Cloud Run. The same `requirements.txt` and
`app/streamlit_app.py` serve as the entry points.

## Visualization

```python
from qqa import visualization as viz

viz.plot_history(result)                       # loss / penalty / diversity
viz.plot_best_trajectory(result, backend="plotly")
viz.plot_schedule(qqa.LinearBGSchedule(-2, 0.1), num_epochs=2000)
viz.plot_run_comparison([r1, r2, r3], labels=["lr=1", "lr=0.5", "lr=2"])
viz.plot_parallel_coordinates(sweep_df, objective="best_obj")
viz.plot_solution_heatmap(result, problem)
```

Every function accepts `backend="matplotlib"` (default) or `backend="plotly"`.
Plotly is optional; if it is not installed the plot silently falls back to
matplotlib.

### Visualization gallery

All figures below are produced by `scripts/make_gallery.py` (regenerate with
`uv run python scripts/make_gallery.py`) and stored under
[`data/fig/gallery/`](data/fig/gallery). Each row shows one problem family
from the catalog; the columns are, left → right, **dynamics**
(`plot_history`), **best trajectory** (`plot_best_trajectory`), **best
solution heatmap** (`plot_solution_heatmap`), and **parallel-population**
evolution (`plot_population_evolution`).

#### Default annealing schedule

<p align="center">
  <img src="data/fig/gallery/schedule_default.png" width="520" alt="Default linear bg schedule from -3.0 to +0.1">
</p>

#### Maximum Independent Set (N=40, 3-regular)

<p align="center">
  <img src="data/fig/gallery/history_mis.png" width="900" alt="MIS loss/penalty/diversity dynamics">
</p>
<p align="center">
  <img src="data/fig/gallery/best_mis.png" width="440">
  <img src="data/fig/gallery/solution_mis.png" width="440">
  <img src="data/fig/gallery/population_mis.png" width="440">
</p>

#### Max-Cut (Erdős–Rényi, N=40, p=0.15)

<p align="center">
  <img src="data/fig/gallery/history_maxcut.png" width="900">
</p>
<p align="center">
  <img src="data/fig/gallery/best_maxcut.png" width="440">
  <img src="data/fig/gallery/solution_maxcut.png" width="440">
  <img src="data/fig/gallery/population_maxcut.png" width="440">
</p>

#### Graph coloring (N=30, 4-regular, K=3)

<p align="center">
  <img src="data/fig/gallery/history_coloring.png" width="900">
</p>
<p align="center">
  <img src="data/fig/gallery/best_coloring.png" width="440">
  <img src="data/fig/gallery/population_coloring.png" width="440">
</p>

#### Ising 1D ferromagnet (N=32, J=1, periodic)

<p align="center">
  <img src="data/fig/gallery/history_ising1d.png" width="900">
</p>
<p align="center">
  <img src="data/fig/gallery/best_ising1d.png" width="440">
  <img src="data/fig/gallery/solution_ising1d.png" width="440">
  <img src="data/fig/gallery/population_ising1d.png" width="440">
</p>

#### Edwards–Anderson 3D spin glass (L=4, seed=0)

<p align="center">
  <img src="data/fig/gallery/history_ea3d.png" width="900">
</p>
<p align="center">
  <img src="data/fig/gallery/best_ea3d.png" width="440">
  <img src="data/fig/gallery/solution_ea3d.png" width="440">
  <img src="data/fig/gallery/population_ea3d.png" width="440">
</p>

#### Sherrington–Kirkpatrick mean-field spin glass (N=80)

<p align="center">
  <img src="data/fig/gallery/history_sk.png" width="900">
</p>
<p align="center">
  <img src="data/fig/gallery/best_sk.png" width="440">
  <img src="data/fig/gallery/solution_sk.png" width="440">
  <img src="data/fig/gallery/population_sk.png" width="440">
</p>

#### Binary perceptron (N=40, α=0.4)

<p align="center">
  <img src="data/fig/gallery/history_perceptron.png" width="900">
</p>
<p align="center">
  <img src="data/fig/gallery/best_perceptron.png" width="440">
  <img src="data/fig/gallery/solution_perceptron.png" width="440">
  <img src="data/fig/gallery/population_perceptron.png" width="440">
</p>

#### Hopfield memory (N=64, P=3)

<p align="center">
  <img src="data/fig/gallery/history_hopfield.png" width="900">
</p>
<p align="center">
  <img src="data/fig/gallery/best_hopfield.png" width="440">
  <img src="data/fig/gallery/solution_hopfield.png" width="440">
  <img src="data/fig/gallery/population_hopfield.png" width="440">
</p>

## Verified correctness

We run QQA against a ground truth or a strong baseline for every problem in
the catalog via `scripts/verify_all_problems.py`. The most recent sweep
(29 instances across 9 problem families) is stored in
[`tasks/verification_report.md`](tasks/verification_report.md); headline
numbers:

| Problem | Instances | Reference | QQA |
| --- | --- | --- | --- |
| Maximum Independent Set | 3×(3-reg, N=50) | networkx degree-greedy | **matches or beats greedy on all seeds** |
| MaxCut | 3×ER (N=30/40/60) | best-of-400 random partition | **+6 / +16 / +27 edges over random** |
| MaxClique | 3×ER (N=30/40/50) | nx.approximation.max_clique | **+1 vertex on every seed** |
| Graph coloring (K=3) | 3×(3-reg, N=40) | Welsh–Powell greedy | **0 conflicts on all seeds** |
| Ising 1D ferromagnet | N ∈ {16, 32, 64} | exact E₀ = −N | **gap = 0 on every size** |
| Edwards–Anderson 2D L=3 | 3 seeds | brute force (2⁹) | **matches exact ground state** |
| Edwards–Anderson 3D L=4 | 2 seeds | — | E/N ≈ −1.61 (no exact solver) |
| Sherrington–Kirkpatrick | N ∈ {50, 100, 200} | Parisi e₀ = −0.7632 | **≤ 3.2 % gap at N=200** |
| Binary perceptron | α ∈ {0.3, 0.5, 0.7} | teacher reaches 0 errors | **0 errors on all α** |
| Hopfield memory | (N, P) ∈ {(32,2),(64,3),(128,4)} | ≥ 0.95 overlap | **overlap = 1.0** |

Overall: **29 / 29 checks pass (100 %)**. Re-run with
`uv run python scripts/verify_all_problems.py`; the command regenerates the
Markdown report in place.

## Notebooks

Nine runnable notebooks live in [`examples/`](examples/). Each notebook has
an **Open in Colab** badge in its first cell and auto-installs `qqa` on Colab.

0. `00_colab_quickstart.ipynb` — one-click tour of every problem
1. `01_maximum_independent_set.ipynb`
2. `02_graph_coloring.ipynb`
3. `03_max_cut.ipynb`
4. `04_edwards_anderson_3d.ipynb`
5. `05_sherrington_kirkpatrick.ipynb`
6. `06_binary_perceptron.ipynb`
7. `07_hopfield_memory.ipynb`
8. `08_parallel_benchmark.ipynb`

Regenerate them deterministically with
`uv run python scripts/_generate_notebooks.py`.

## Documentation

```bash
uv run mkdocs serve            # http://127.0.0.1:8000
uv run mkdocs build --strict   # produces site/
```

The docs cover quickstart, the full problem catalog with mathematical
definitions, GUI walk-through, visualization guide, auto-generated API
reference, and a migration guide from 0.2.x.

## Scripts

| Script | Purpose |
| ------ | ------- |
| `scripts/demo_mis.py`        | Minimal MIS end-to-end demo |
| `scripts/demo_coloring.py`   | 3-coloring end-to-end demo |
| `scripts/demo_parallel.py`   | Parallel instances of MIS |
| `scripts/bench_er_small.py`  | Benchmark on bundled ER-small MIS dataset |
| `scripts/_generate_notebooks.py` | Regenerate the shipped example notebooks |

Run any script via `uv run python scripts/<name>.py`.

## Repository layout

```
QQA4CO/
├── src/qqa/                 # importable package
│   ├── __init__.py
│   ├── annealing.py
│   ├── callbacks.py
│   ├── cli.py
│   ├── datasets.py
│   ├── legacy.py
│   ├── problems/            # qubo.py / categorical.py / spin.py
│   ├── relaxation.py
│   ├── schedule.py
│   ├── utils.py
│   └── visualization.py
├── app/                     # Streamlit dashboard
│   ├── streamlit_app.py
│   └── pages/
├── docs/                    # MkDocs site sources
├── examples/                # 8 example notebooks
├── scripts/                 # demo / benchmark scripts
├── tests/                   # pytest suite
├── data/                    # bundled datasets
├── pyproject.toml
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CITATION.cff
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
