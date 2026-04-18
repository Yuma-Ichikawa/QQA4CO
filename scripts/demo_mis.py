"""Minimal QQA demo: Maximum Independent Set on a small random regular graph.

Run with:
    uv run scripts/demo_mis.py
or, after `pip install -e .`:
    python scripts/demo_mis.py
"""

from __future__ import annotations

import networkx as nx
import torch

import qqa


def main() -> int:
    qqa.fix_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo_mis] device = {device}")

    N, d = 50, 3
    g = nx.random_regular_graph(d=d, n=N, seed=0)
    problem = qqa.MaximumIndependentSet(g, penalty=2, device=device)

    result = qqa.anneal(
        problem,
        sol_size=64,
        learning_rate=1.0,
        temp=1e-3,
        schedule=qqa.LinearBGSchedule(min_bg=-3, max_bg=0.1),
        curve_rate=4,
        div_param=0.2,
        num_epochs=1500,
        check_interval=500,
        device=device,
    )

    # loss = x^T Q x with diag(-1) and off-diag = penalty * 1_edge.
    # If all selected nodes are mutually non-adjacent, loss == -|MIS|.
    mis_size = int(round(-result.best_obj))
    print(f"\n[demo_mis] Graph: {N}-node {d}-regular | MIS size found = {mis_size}")
    print(f"[demo_mis] Runtime = {result.runtime:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
