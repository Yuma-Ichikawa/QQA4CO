# How to run on GPU (CUDA / MPS / Blackwell)

QQA4CO is a thin layer over PyTorch — every call accepts a `device`
keyword and forwards it down. The pitfalls below are the ones we have
hit in practice; treat this page as the GPU operations manual.

## Pick a device

```python
import torch

# Most common: pick CUDA when present, fall back to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Apple Silicon — supported because we never call CUDA-only kernels.
# device = "mps" if torch.backends.mps.is_available() else "cpu"
```

`qqa.anneal()` and `qqa.pignn.train_*` both raise an early, actionable
`RuntimeError` when you pass `device="cuda"` without a CUDA-enabled
PyTorch build. You will not get a deep, cryptic stack trace.

## Always construct the problem on the *same* device you train on

This is the **#1 source of GPU bugs** for Python-API users:

```python
# WRONG — will crash on the first epoch:
problem = qqa.MaximumIndependentSet(g, penalty=2, device="cpu")
qqa.anneal(problem, device="cuda", num_epochs=1000)
```

The constructor materialises tensors (`Q_mat`, etc.) on whatever
`device=` you pass. If the solver later runs on a different device the
inner `einsum` raises a CUDA / CPU mismatch.

* `qqa.anneal()` — **does not auto-migrate** the problem. You must
  construct on the right device.
* `qqa.pignn.train_cra_pi_gnn` / `train_cpra_pi_gnn` — **does
  auto-migrate** silently via `_ensure_problem_on_device`. The CLI
  always builds on the right device, so this only matters for
  Python-API users of the PyG backend.

### Recommended pattern

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
qqa.fix_seed(0)
g = nx.random_regular_graph(d=3, n=200, seed=0)
problem = qqa.MaximumIndependentSet(g, penalty=2, device=device)
result = qqa.anneal(problem, sol_size=128, num_epochs=2000, device=device)
```

Note: passing `device=` to both the problem **and** the solver. Always.

## Determinism

`qqa.fix_seed(seed)` seeds Python, NumPy, and PyTorch (CPU + CUDA) and
flips `torch.backends.cudnn.deterministic = True`. That is enough for
QQA's pure tensor ops to be reproducible. The PyG GCN forward includes
some non-deterministic kernels by default; if you need bit-identical
runs across processes you may also need:

```python
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
```

This will trade a few percent of throughput for full determinism.

## Picking `sol_size`

`sol_size` is the parallel population — these run *as a batch* on the
GPU. Larger is better up to your VRAM ceiling.

| GPU memory | `MaximumIndependentSet`(N=1024) safe `sol_size` |
|---|---|
| 8 GB | ~256 |
| 24 GB | ~1024 |
| 48 GB | ~2048 |
| 80 GB (H100) | ~4096 |
| 192 GB (B200) | ~8192 |

If you OOM, halve `sol_size` and try again — the loop is otherwise
identical.

## Blackwell / B200 / `sm_100`

QQA's default backend uses only stock PyTorch ops, so any CUDA build
that supports your card works. As of April 2026 you need PyTorch
**≥ 2.7 with CUDA 12.8** for B200; the project's `pyproject.toml`
keeps the lower bound permissive (`torch >= 2.2`) but our development
environment uses the cu130 wheel from <https://download.pytorch.org/whl/cu130>.

The optional **`qqa.pignn`** backend uses
`torch_geometric`, which itself uses pure PyTorch kernels (no
hand-written CUDA). It is therefore Blackwell-ready, unlike the
upstream DGL-based CRA reference which lacks a Blackwell wheel.

```bash
# Blackwell-friendly install (pip)
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install "qqa[pignn]"
```

## MPS (Apple Silicon)

`device="mps"` works for the default `qqa.anneal` backend. The
`qqa.pignn` backend has not been validated on MPS and may fall back to
CPU for unsupported ops; for the PyG path we recommend CUDA or CPU.

## Slurm / cluster recipe

A Slurm template script lives at
`scripts/sanity_pignn_gpu.sbatch`. Adapt it for your cluster:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
module load cuda/12.8
uv run python scripts/demo_pignn_mis.py
```

Use `qqa solve --device cuda --output result.pkl` if you want to run
the CLI from a job script and post-process the result later.

## Diagnosing slow GPU runs

* **Look at `time per epoch`.** Print the `result.runtime / num_epochs`.
  For MIS on N=200 the QQA backend should be < 1 ms/epoch on an A100.
* **Profile.** `torch.profiler.profile()` works unmodified on
  `qqa.anneal`. The hot path is `loss_fn` + `optimizer.step()`.
* **Check `pin_memory` and async copies.** QQA does not move data
  between host and device per-epoch; if your problem does so in its
  `loss_fn`, factor those tensors out into the constructor.
