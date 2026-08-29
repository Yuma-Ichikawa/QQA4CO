# How to integrate QQA4CO into your pipeline

QQA4CO is designed to be **import-friendly**: a single library call
returns a pickle-able dataclass, with no global state and no required
side effects. This page walks through the most common pipeline
integrations.

## The contract — `AnnealResult`

`qqa.anneal()` always returns a
[`qqa.AnnealResult`](../api.md#qqa.annealing.AnnealResult)
dataclass with these fields:

| Field | Type | What it is |
|---|---|---|
| `best_sol` | `torch.Tensor` | Discrete winning configuration. `(N,)` for single-instance, `(I, N)` for batched-instance. |
| `best_obj` | `float` or `np.ndarray` | Loss value(s). Scalar for single-instance, `(I,)` for batched. |
| `runtime` | `float` | Wall-clock seconds for the solve. |
| `history` | `dict[str, list]` | Per-epoch metrics. Empty if `record_history=False`. |
| `callbacks` | `list[Callback]` | The active callbacks (so you can inspect their state post-hoc). |
| `score` | `dict` | Human-readable summary from `problem.score_summary` — `label`, `value`, `unit`, `feasible`, `extra`. |

`qqa.pignn.train_cra_pi_gnn` and `train_cpra_pi_gnn` return the same
type, so the rest of this page applies to all three backends.

## Pattern 1 — call inside an existing Python script

The simplest integration. Three lines:

```python
import qqa, networkx as nx

g = nx.read_gml("my_graph.gml")
problem = qqa.MaximumIndependentSet(g, penalty=2)
result = qqa.anneal(problem, sol_size=256, num_epochs=2000)

if not result.score["feasible"]:
    raise ValueError(f"QQA returned an infeasible solution: {result.score['extra']}")
my_independent_set = [int(node) for node, bit in enumerate(result.best_sol.tolist()) if bit]
```

Key points:

* Always check `result.score["feasible"]` before trusting the
  solution. The QQA penalty is soft — feasibility is *typical* but not
  guaranteed.
* `result.best_sol` is a tensor; convert to a Python list (or NumPy
  array) before handing off to non-Torch downstream code.

## Pattern 2 — persist the full result for later analysis

`AnnealResult` is a plain dataclass and pickles cleanly:

```python
import pickle
import qqa

result = qqa.anneal(problem, ...)
with open("result.pkl", "wb") as f:
    pickle.dump(result, f)

# Days later, in a Jupyter notebook:
with open("result.pkl", "rb") as f:
    result = pickle.load(f)

import qqa.visualization as viz

viz.plot_history(result.history)  # or any other helper
```

The CLI does exactly this when you pass `--output result.pkl`. Use it
for HPC runs that should hand off to a notebook on a different
machine.

## Pattern 3 — call from a CLI / shell pipeline

When a Python import is overkill (e.g. shell scripts, Snakemake,
Nextflow) use the CLI:

```bash
qqa solve \
  --problem mis \
  --graph-file my_graph.gpickle \
  --sol-size 256 \
  --epochs 2000 \
  --device cuda \
  --output result.pkl

python -c "import pickle; r = pickle.load(open('result.pkl','rb')); print(r.score)"
```

See [CLI reference](../reference/cli.md) for the full flag list. The
CLI returns a non-zero exit code on infeasibility and prints the same
`score` dict you would see from the Python API.

## Pattern 4 — wrap your domain solver

If your problem is not in the catalogue, wrap it in `qqa.UserProblem`:

```python
import torch, qqa


def my_loss(x: torch.Tensor) -> torch.Tensor:
    # x has shape (B, N) for binary problems
    return -x.sum(dim=-1) + 5 * (x[:, 0] - x[:, -1]).pow(2)


problem = qqa.UserProblem(num_vars=64, variable_kind="binary", loss_fn=my_loss)
result = qqa.anneal(problem, sol_size=128, num_epochs=1500)
```

For the full extension story (including a real `score_summary`,
constraints, and CLI hooks) see [Extending QQA4CO](../develop/extending.md).

## Reproducibility — the QQA contract

Two ingredients are required for run-to-run determinism:

1. `qqa.fix_seed(seed)` at the start of every script. (See [GPU
   notes](gpu.md#determinism) for the determinism caveat on PyG.)
2. The same `device`, `sol_size`, `num_epochs`, schedule, and
   `learning_rate` between runs.

The `qqa.anneal()` loop itself is deterministic given those inputs,
modulo tiny floating-point variation across CUDA driver versions.

## Logging

QQA emits human-readable progress with `print()` (controlled by
`verbose=True / False`). For a library use-case where you want
structured logs:

```python
import logging
import qqa

logging.basicConfig(level=logging.INFO)

# Quiet the human-readable per-epoch output:
result = qqa.anneal(problem, verbose=False, num_epochs=1000)
logging.info("QQA done in %.2fs (best=%s)", result.runtime, result.best_obj)
```

## Don'ts

* **Do not call `qqa.fix_seed` inside a hot loop.** It flips
  `cudnn.deterministic = True` for the whole process and adds a
  measurable overhead. Once per script is enough.
* **Do not mutate `result.best_sol` in place.** It is the canonical
  reference to the winning solution; clone it (`bs = result.best_sol
  .clone()`) before any modification.
* **Do not pickle the whole `AnnealResult` if you logged populations
  with `PopulationTracker(record_x=True)`.** That can balloon to
  hundreds of MB. Pickle only `result.score` and `result.history` if
  you do not need the population.
