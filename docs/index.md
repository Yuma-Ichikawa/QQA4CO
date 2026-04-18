# QQA — Quasi-Quantum Annealing

**QQA** is a GPU-native solver for combinatorial optimization and spin-glass
problems. It frames a discrete problem as a continuous, differentiable
objective and anneals towards a discrete minimum using gradient-based
sampling.

- **Unified API**: one `qqa.anneal()` for binary, categorical, and spin
  problems.
- **Problem catalog**: MIS, MaxCut, MaxClique, graph coloring, balanced graph
  partitioning, 1D Ising, Edwards-Anderson, Sherrington-Kirkpatrick, binary
  perceptron, Hopfield memory.
- **Interactive visualization**: matplotlib by default, Plotly when
  installed.
- **CLI** and **Streamlit GUI** for exploratory work.

## Install

```bash
pip install qqa                # core
pip install "qqa[plotly]"      # + interactive plots
pip install "qqa[gui]"         # + Streamlit GUI
pip install "qqa[all]"         # everything
```

See [Quickstart](quickstart.md) to run your first solve.

## Cite

If you use QQA in your research, please cite the paper:

> Y. Ichikawa, Y. Arai. *Continuous Tensor Relaxation for Finding Diverse
> Solutions in Combinatorial Optimization.* ICLR 2025.
