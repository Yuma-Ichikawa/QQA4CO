"""Optional PyTorch Geometric backend: CRA-PI-GNN trainer.

This subpackage provides a self-contained re-implementation of the
**CRA-PI-GNN** algorithm from

    Y. Ichikawa. "Controlling Continuous Relaxation for Combinatorial
    Optimization." NeurIPS 2024.
    https://openreview.net/forum?id=ykACV1IhjD

The reference release uses **DGL**. Because DGL's prebuilt wheels do not
yet target NVIDIA Blackwell (``sm_100``) and lag the latest PyTorch /
CUDA combos, we ship a port written in **PyTorch Geometric** so QQA users
on modern GPUs can compare against CRA-PI-GNN without juggling DGL.

Why this lives here, not in the main ``qqa`` namespace
------------------------------------------------------
* PyG and its transitive deps are heavy. We do not want ``import qqa`` to
  pay for them when the user only needs ``qqa.anneal``.
* The trainer is a *backend alternative* to ``qqa.anneal``, not a building
  block of it. Keeping it isolated also makes the README's "QQA vs.
  CRA-PI-GNN" comparison story clear.

Quickstart
----------
::

    pip install "qqa[pignn]"

::

    import networkx as nx
    import qqa
    from qqa.pignn import train_cra_pi_gnn

    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=200, seed=0)
    problem = qqa.MaximumIndependentSet(g, penalty=2)
    result = train_cra_pi_gnn(problem, num_epochs=4000)
    print(result.score)

The function returns a :class:`qqa.AnnealResult`, so it is a drop-in
alternative to :func:`qqa.anneal` — every downstream helper that consumes
``AnnealResult`` (visualisation, CLI scoring) keeps working.

See also
--------
The original DGL implementation:
https://github.com/Yuma-Ichikawa/CRA4CO
"""

from __future__ import annotations

from qqa.pignn.model import GCNNet
from qqa.pignn.trainer import train_cra_pi_gnn

__all__ = ["GCNNet", "train_cra_pi_gnn"]
