"""Solve multiple MIS instances in parallel (single GPU kernel)."""

from __future__ import annotations

import networkx as nx
import torch

import qqa


def main() -> int:
    qqa.fix_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo_parallel] device = {device}")

    N = 100
    degrees = [2, 3, 4, 5]
    graphs = [nx.random_regular_graph(d=d, n=N, seed=d) for d in degrees]
    problem = qqa.MaximumIndependentSetInstance(graphs, max_node=N, penalty=2, device=device)

    result = qqa.anneal(
        problem,
        sol_size=64,
        learning_rate=1.0,
        temp=1e-3,
        schedule=qqa.LinearBGSchedule(min_bg=-3, max_bg=0.1),
        curve_rate=4,
        div_param=0.2,
        num_epochs=1000,
        check_interval=500,
        device=device,
    )

    print("\n[demo_parallel] Per-instance MIS sizes:")
    for deg, obj in zip(degrees, result.best_obj, strict=False):
        print(f"  degree={deg}: MIS size = {int(round(-float(obj)))}")
    print(f"[demo_parallel] Runtime = {result.runtime:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
