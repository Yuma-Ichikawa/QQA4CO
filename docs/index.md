# QQA4CO — Parallel Quasi-Quantum Annealing

**QQA4CO** is a research-grade PyTorch toolkit for combinatorial and
spin-glass optimisation. It frames a discrete problem as a
continuous, differentiable objective and anneals towards a discrete
minimum using gradient-based parallel sampling on the GPU.

- **Unified API** — one `qqa.anneal()` for binary, categorical,
  permutation, batched-instance, and spin problems.
- **17-class problem catalogue** — MIS, MaxCut, MaxClique, Vertex
  Cover, Graph Bisection, Coloring, BalancedGraphPartition, Knapsack,
  NumberPartitioning, MaxSAT3, TSP, QAP, NQueens, Ising 1D,
  Edwards-Anderson, Sherrington-Kirkpatrick, BinaryPerceptron,
  HopfieldMemory.
- **Optional GNN backend** (`qqa[pignn]`) — the **CRA-PI-GNN**
  baseline (NeurIPS 2024) and the **CPRA** diverse-solution framework
  (TMLR 2025), both ported to PyTorch Geometric so they run on
  Blackwell-class GPUs.
- **Streamlit dashboard** — `qqa gui` opens a polished UI with live
  progress, per-problem visualisations, and parallel-population view.
- **`qqa` CLI** — reproducible solves and benchmarks from the shell.

## Install

```bash
pip install qqa                # core
pip install "qqa[plotly]"      # + interactive plots
pip install "qqa[gui]"         # + Streamlit GUI
pip install "qqa[pignn]"       # + CRA-PI-GNN / CPRA backends
pip install "qqa[all]"         # everything
```

See [Quickstart](quickstart.md) to run your first solve.

## Where to go next

| If you want to … | Go to |
|---|---|
| Run your first solve | [Quickstart](quickstart.md) |
| Understand which solver to pick | [Backend reference](reference/backends.md) |
| Tune hyper-parameters | [How-to → Tuning](how-to/tuning.md) |
| Run on GPU / Blackwell | [How-to → GPU](how-to/gpu.md) |
| Integrate QQA4CO in a pipeline | [How-to → Integrate](how-to/integrate.md) |
| Browse every problem class | [Problem catalog](problems.md) |
| Browse every CLI flag | [CLI reference](reference/cli.md) |
| Understand the algorithm | [Algorithm explainer](explanation/algorithm.md) |
| Add a new problem / relaxation / backend | [Develop → Extending QQA4CO](develop/extending.md) |
| Read the source map | [Develop → Internals](develop/internals.md) |
| Cut a release | [Develop → Releasing](develop/releasing.md) |

## Cite

If you use QQA4CO in your research, please cite the paper(s) that
match the backend you used. The full BibTeX is in the
[README](https://github.com/Yuma-Ichikawa/QQA4CO#cite); a one-line
summary:

* **`qqa.anneal` (PQQA)** — Y. Ichikawa, *"Optimization by Parallel
  Quasi-Quantum Annealing with Gradient-Based Sampling,"*
  [arXiv:2409.00184](https://arxiv.org/abs/2409.00184), 2024.
* **`qqa.pignn.train_cra_pi_gnn`** — Y. Ichikawa, *"Controlling
  Continuous Relaxation for Combinatorial Optimization,"* NeurIPS
  2024 ([OpenReview](https://openreview.net/forum?id=ykACV1IhjD)).
* **`qqa.pignn.train_cpra_pi_gnn`** — Y. Ichikawa & H. Iwashita,
  *"Continuous Parallel Relaxation for Finding Diverse Solutions in
  Combinatorial Optimization Problems,"* TMLR 2025
  ([OpenReview](https://openreview.net/forum?id=ix33zd5zCw)).
