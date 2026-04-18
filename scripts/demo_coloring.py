"""Minimal QQA demo: K-coloring of a random regular graph."""

from __future__ import annotations

import networkx as nx
import torch

import qqa


def main() -> int:
    qqa.fix_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo_coloring] device = {device}")

    N, d, K = 50, 5, 4
    g = nx.random_regular_graph(d=d, n=N, seed=0)
    problem = qqa.Coloring(g, num_category=K, device=device)

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

    violations = int(round(result.best_obj))
    print(
        f"\n[demo_coloring] {K}-colorable? {'YES' if violations == 0 else 'NO'} "
        f"(conflicts={violations})"
    )
    print(f"[demo_coloring] Runtime = {result.runtime:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
