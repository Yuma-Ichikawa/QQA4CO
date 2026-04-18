"""QQA vs. CRA-PI-GNN comparison (regenerates the README's comparison table).

Runs both solvers on a few random-regular MIS instances and prints a
markdown-friendly summary. Intentionally CPU-runnable in well under a
minute per row so anyone can reproduce the numbers locally.

Usage::

    uv run python scripts/bench_qqa_vs_pignn.py
or::

    python scripts/bench_qqa_vs_pignn.py --device cuda

The PyG backend requires ``pip install qqa[pignn]``; the script tells
you so explicitly if it cannot import :mod:`qqa.pignn`.
"""

from __future__ import annotations

import argparse
import time

import networkx as nx
import torch

import qqa

INSTANCES = [
    # label,                  N,    d,  penalty
    ("MIS, N=100,  d=3-reg", 100, 3, 2),
    ("MIS, N=300,  d=3-reg", 300, 3, 2),
    ("MIS, N=500,  d=20-reg", 500, 20, 2),
]

# QQA defaults — the same ``qqa solve --problem mis`` would invoke.
QQA_HP = dict(
    sol_size=100,
    num_epochs=2000,
    learning_rate=1.0,
    min_bg=-3,
    max_bg=0.1,
    curve_rate=2,
)

# CRA-PI-GNN with the README's "medium graph" hyperparameters. The function
# defaults are paper values (init_reg_param=-20) tuned for N >= 1000; on
# small/medium graphs they routinely under-converge, so we ship a gentler
# schedule here for fair head-to-head numbers.
PIGNN_HP = dict(
    learning_rate=1e-3,
    init_reg_param=-2.0,
    annealing_rate=5e-4,
    num_epochs=5000,
    curve_rate=2,
)


def _bench_one(label: str, n: int, d: int, penalty: int, device: str) -> dict:
    g = nx.random_regular_graph(d=d, n=n, seed=0)
    qqa.fix_seed(0)
    p_qqa = qqa.MaximumIndependentSet(g, penalty=penalty, device=device)
    t0 = time.time()
    r_qqa = qqa.anneal(p_qqa, device=device, verbose=False, **QQA_HP)
    qqa_runtime = time.time() - t0

    qqa.fix_seed(0)
    p_pi = qqa.MaximumIndependentSet(g, penalty=penalty, device=device)
    from qqa.pignn import train_cra_pi_gnn

    t0 = time.time()
    r_pi = train_cra_pi_gnn(p_pi, device=device, verbose=False, seed=0, **PIGNN_HP)
    pignn_runtime = time.time() - t0

    return {
        "label": label,
        "qqa_size": r_qqa.score["value"],
        "qqa_feas": r_qqa.score["feasible"],
        "qqa_rt": qqa_runtime,
        "pignn_size": r_pi.score["value"],
        "pignn_feas": r_pi.score["feasible"],
        "pignn_rt": pignn_runtime,
    }


def _format_row(row: dict) -> str:
    return (
        f"| {row['label']:<22} | "
        f"{row['qqa_size']:>4} ({'feas' if row['qqa_feas'] else 'INFEAS'}), "
        f"{row['qqa_rt']:>5.1f}s | "
        f"{row['pignn_size']:>4} ({'feas' if row['pignn_feas'] else 'INFEAS'}), "
        f"{row['pignn_rt']:>5.1f}s |"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    print(f"# QQA vs. CRA-PI-GNN bench — device={device}")
    if device.startswith("cuda"):
        print(f"# GPU : {torch.cuda.get_device_name(0)}")
        print(f"# Arch: {torch.cuda.get_arch_list()}")
    print(f"# QQA HP   : {QQA_HP}")
    print(f"# PIGNN HP : {PIGNN_HP}\n")

    print("| Instance               | qqa.anneal           | qqa.pignn (CRA-PI-GNN) |")
    print("|------------------------|----------------------|------------------------|")
    for spec in INSTANCES:
        row = _bench_one(*spec, device=device)
        print(_format_row(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
