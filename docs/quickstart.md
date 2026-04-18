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
result = qqa.anneal(problem, sol_size=100, num_epochs=1500)
print(f"MIS size: {-int(result.best_obj)}  in {result.runtime:.2f}s")
```

The same `qqa.anneal(...)` call works for every problem in the
[catalog](problems.md).

## Run the CLI

```bash
qqa version
qqa solve --problem sk --size 100 --sol-size 128 --epochs 1000
qqa bench --preset er-small --epochs 500
```

## Launch the GUI

```bash
qqa gui              # opens http://localhost:8501
```

## Browse example notebooks

- `examples/01_maximum_independent_set.ipynb`
- `examples/04_edwards_anderson_3d.ipynb`
- `examples/06_binary_perceptron.ipynb`
- …

Run any of them with ``uv run jupyter lab``.
