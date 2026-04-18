"""Minimal CRA-PI-GNN (PyTorch Geometric) demo on a small MIS instance.

Counterpart of ``scripts/demo_mis.py``, exercising the optional PyG backend
implemented in :mod:`qqa.pignn`. Requires installing the ``pignn`` extra::

    pip install "qqa[pignn]"

Run with::

    uv run scripts/demo_pignn_mis.py
or, after ``pip install -e ".[pignn]"``::

    python scripts/demo_pignn_mis.py
"""

from __future__ import annotations

import networkx as nx
import torch

import qqa
from qqa.pignn import train_cra_pi_gnn


def main() -> int:
    qqa.fix_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo_pignn_mis] device = {device}")

    N, d = 200, 3
    g = nx.random_regular_graph(d=d, n=N, seed=0)
    problem = qqa.MaximumIndependentSet(g, penalty=2, device=device)

    # Hyperparameters tuned for medium-sized graphs (N ~ 100-500). The
    # function defaults are taken straight from the NeurIPS 2024 paper and
    # are tuned for N >= 1000; smaller instances need a gentler schedule.
    result = train_cra_pi_gnn(
        problem,
        learning_rate=1e-3,
        init_reg_param=-2.0,
        annealing_rate=5e-4,
        num_epochs=5000,
        check_interval=1000,
        device=device,
        seed=0,
    )

    score = result.score
    print(
        f"\n[demo_pignn_mis] Graph: {N}-node {d}-regular | "
        f"IS size = {score['value']} {score.get('unit', '')} "
        f"({'feasible' if score['feasible'] else 'INFEASIBLE'})"
    )
    print(f"[demo_pignn_mis] Runtime = {result.runtime:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
