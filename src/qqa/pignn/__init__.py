"""Optional PyTorch Geometric backend: CRA-PI-GNN and CPRA trainers.

This subpackage provides self-contained re-implementations of two
GNN-based unsupervised-learning combinatorial-optimization solvers:

* **CRA-PI-GNN** — Y. Ichikawa, *"Controlling Continuous Relaxation for
  Combinatorial Optimization,"* NeurIPS 2024
  (https://openreview.net/forum?id=ykACV1IhJD). Single-replica
  continuous-relaxation annealing on a 2-layer GCN.

* **CPRA** — Y. Ichikawa & H. Iwashita, *"Continuous Parallel Relaxation
  for Finding Diverse Solutions in Combinatorial Optimization
  Problems,"* TMLR 2025
  (https://openreview.net/forum?id=ix33zd5zCw). A multi-head extension
  of CRA-PI-GNN that returns ``R`` diverse solutions in one training
  run, supporting both penalty- and variation-diversification.

Both reference releases use **DGL**. Because DGL's prebuilt wheels do
not yet target NVIDIA Blackwell (``sm_100``) and lag the latest PyTorch
/ CUDA combos, we ship ports written in **PyTorch Geometric** so QQA
users on modern GPUs can compare against either method without juggling
DGL.

Why this lives here, not in the main ``qqa`` namespace
------------------------------------------------------
* PyG and its transitive deps are heavy. We do not want ``import qqa`` to
  pay for them when the user only needs ``qqa.anneal``.
* The trainers are *backend alternatives* to ``qqa.anneal``, not building
  blocks of it. Keeping them isolated also makes the README's "QQA vs.
  CRA-PI-GNN vs. CPRA" comparison story clear.

Quickstart
----------
::

    pip install "qqa[pignn]"

CRA-PI-GNN (single solution per run)::

    import networkx as nx
    import qqa
    from qqa.pignn import train_cra_pi_gnn

    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=200, seed=0)
    problem = qqa.MaximumIndependentSet(g, penalty=2)
    result = train_cra_pi_gnn(problem, num_epochs=4000)
    print(result.score)

CPRA (R diverse solutions per run, e.g. one per penalty level)::

    from qqa.pignn import train_cpra_pi_gnn

    penalties = [1.5, 2.0, 2.5, 3.0]
    replicas = [qqa.MaximumIndependentSet(g, penalty=p) for p in penalties]
    result = train_cpra_pi_gnn(
        problem,
        num_replicas=len(replicas),
        replica_problems=replicas,
        num_epochs=4000,
    )
    for record in result.score["extra"]["replicas"]:
        print(record["score"])

Both functions return a :class:`qqa.AnnealResult`, so every downstream
helper that consumes ``AnnealResult`` (visualisation, CLI scoring) keeps
working.

See also
--------
* CRA reference (DGL): https://github.com/Yuma-Ichikawa/CRA4CO
* CPRA reference (DGL): https://github.com/Yuma-Ichikawa/CPRA4CO
"""

from __future__ import annotations

from qqa.pignn.model import GCNNet
from qqa.pignn.trainer import train_cpra_pi_gnn, train_cra_pi_gnn

__all__ = ["GCNNet", "train_cpra_pi_gnn", "train_cra_pi_gnn"]
