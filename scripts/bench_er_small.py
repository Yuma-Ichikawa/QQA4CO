"""Benchmark: solve the 6 MIS-ER-small graphs shipped in ``data/``."""

from __future__ import annotations

import torch

import qqa
from qqa import datasets


def main() -> int:
    qqa.fix_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bench_er_small] device = {device}")

    problem = datasets.mis_er_small(penalty=2, problem_type="all", device=device)
    print(f"[bench_er_small] Loaded {problem.num_instance} graphs, max_node={problem.max_node}")

    result = qqa.anneal(
        problem,
        sol_size=50,
        learning_rate=1.0,
        temp=1e-3,
        schedule=qqa.LinearBGSchedule(min_bg=-3, max_bg=0.1),
        curve_rate=4,
        div_param=0.2,
        num_epochs=1000,
        check_interval=250,
        device=device,
    )

    print("\n[bench_er_small] Per-instance MIS sizes:")
    for i, obj in enumerate(result.best_obj):
        print(f"  instance {i}: MIS size = {int(round(-float(obj)))}")
    print(f"[bench_er_small] Runtime = {result.runtime:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
