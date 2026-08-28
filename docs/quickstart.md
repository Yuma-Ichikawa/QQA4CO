# Quickstart

## With `uv` (recommended)

```bash
git clone https://github.com/Yuma-Ichikawa/QQA4CO.git
cd QQA4CO
uv sync --extra plotly --extra gui --extra dev
```

The first sync takes a couple of minutes (PyTorch wheel). Subsequent runs
reuse the resolved lockfile.

## With pip

```bash
pip install qqa                          # core (CPU torch + numpy/networkx)
pip install "qqa[plotly]"                # + interactive Plotly figures
pip install "qqa[gui]"                   # + Streamlit dashboard
pip install "qqa[all]"                   # everything (docs, dev, notebook, gui, plotly)
```

Quote the extras list — bare brackets are a glob pattern in zsh and
Bash with `noglob` disabled.

## Solve a first problem

```python
import networkx as nx
import qqa

qqa.fix_seed(0)
g = nx.random_regular_graph(d=3, n=100, seed=0)
problem = qqa.MaximumIndependentSet(g, penalty=2)
result = qqa.solve(problem, profile="balanced", device="auto")
print(f"MIS size: {-int(result.best_obj)}  in {result.runtime:.2f}s")
```

`qqa.solve(...)` is the stable entry point. `qqa.anneal(...)` remains
available when direct control of the QQA loop is useful.

Inspect the route before spending a budget:

```python
print(qqa.inspect(problem).to_dict())
print(qqa.plan(problem, profile="quality", device="auto").to_dict())
```

Solve a public MPS/QPLIB file with the same API:

```python
result = qqa.solve("instance.mps", profile="balanced", budget=60)
# Explicit opt-in certification; install the selected backend extra first.
certified = qqa.solve("instance.mps", profile="certify", budget=60)
```

## Run the CLI

```bash
qqa version
qqa inspect model.mps
qqa plan model.mps --profile balanced
qqa solve model.mps --profile balanced --budget 60
qqa solve --problem sk --size 100 --sol-size 128 --epochs 1000
qqa benchmark fetch miplib --output benchmarks/miplib
qqa benchmark fetch qplib --output benchmarks/qplib
```

## Launch the GUI

```bash
qqa gui
```

## Browse example notebooks

- `examples/01_maximum_independent_set.ipynb`
- `examples/04_edwards_anderson_3d.ipynb`
- `examples/06_binary_perceptron.ipynb`
- `examples/13_typed_primal_dual_runtime.ipynb` — Model Doctor, goal/budget
  solve, cockpit, checkpoint/resume, and verified result package
- …
- `notebooks/cra_pignn_example.ipynb` — CRA-PI-GNN walkthrough across
  every supported graph problem
- `notebooks/cpra_pignn_example.ipynb` — CPRA penalty / variation
  diversification

Run any of them with ``uv run jupyter lab``.

## Where to go next

* [Backends reference](reference/backends.md) — pick `qqa` / `pignn`
  / `cpra` for your problem.
* [How-to → Tuning](how-to/tuning.md) — `sol_size`, `num_epochs`,
  schedule defaults that work.
* [How-to → GPU](how-to/gpu.md) — CUDA / MPS / Blackwell notes and the
  device-mismatch pitfall.
* [How-to → Integrate](how-to/integrate.md) — embed QQA4CO into a
  pipeline.
* [Develop → Extending QQA4CO](develop/extending.md) — add a new
  problem, relaxation, callback, or whole backend.
