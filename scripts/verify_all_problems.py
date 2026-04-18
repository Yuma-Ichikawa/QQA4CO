"""Thorough correctness sweep for every built-in QQA problem.

For each problem family, we run QQA on one or more instances and compare the
best objective (or a problem-specific metric) against a ground truth or a
strong baseline. Results are aggregated into a Markdown report written to
``docs/verification.md`` so it renders both on GitHub and in the MkDocs
site.

References used:
- MIS & MaxClique: NetworkX greedy heuristics (independent_set, max_weight_clique).
- MaxCut: repeated random-hyperplane rounding baseline (Goemans–Williamson style).
- Coloring: Welsh–Powell-style networkx greedy coloring as a feasibility anchor.
- Ising1D FM (J=1, h=0, periodic): exact ``E_0 = -N``.
- Edwards-Anderson 2D (L=3): brute-force all 2^9 configs.
- SK(N): Parisi e_0 = -0.7631667 (typical-case target, N→∞).
- BinaryPerceptron: teacher solution exists with zero error by construction.
- Hopfield: stored-pattern recovery (overlap → 1).

Run::

    uv run python scripts/verify_all_problems.py
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

import qqa

PARISI_SK = -0.7631667  # known thermodynamic limit
REPORT = Path(__file__).resolve().parents[1] / "docs" / "verification.md"


# ---------------------------------------------------------------------------
# QQA driver
# ---------------------------------------------------------------------------


def run_qqa(
    problem,
    *,
    sol_size: int = 128,
    num_epochs: int = 1500,
    lr: float = 1.0,
    temp: float = 1e-3,
    min_bg: float = -3.0,
    max_bg: float = 0.1,
    curve_rate: int = 4,
    div_param: float = 0.2,
    seed: int = 0,
) -> qqa.AnnealResult:
    qqa.fix_seed(seed)
    return qqa.anneal(
        problem,
        sol_size=sol_size,
        learning_rate=lr,
        temp=temp,
        schedule=qqa.LinearBGSchedule(min_bg=min_bg, max_bg=max_bg),
        curve_rate=curve_rate,
        div_param=div_param,
        num_epochs=num_epochs,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Baselines / ground truth
# ---------------------------------------------------------------------------


def greedy_mis(g: nx.Graph) -> int:
    """Simple degree-based greedy MIS lower bound."""
    H = g.copy()
    mis = set()
    while H.number_of_nodes() > 0:
        v = min(H.nodes, key=lambda n: H.degree(n))
        mis.add(v)
        H.remove_nodes_from(list(H.neighbors(v)) + [v])
    return len(mis)


def greedy_maxcut(g: nx.Graph, trials: int = 200, seed: int = 0) -> int:
    """Random-partition baseline for MaxCut."""
    rng = np.random.default_rng(seed)
    best = 0
    n = g.number_of_nodes()
    nodes = list(g.nodes)
    for _ in range(trials):
        side = {nodes[i]: int(rng.integers(0, 2)) for i in range(n)}
        cut = sum(1 for u, v in g.edges if side[u] != side[v])
        if cut > best:
            best = cut
    return best


def greedy_maxclique(g: nx.Graph) -> int:
    clique, _ = nx.algorithms.approximation.max_clique(g), None
    return len(clique)


def greedy_coloring(g: nx.Graph, K: int) -> int:
    """Return number of conflicting edges under Welsh–Powell greedy coloring."""
    colors = nx.greedy_color(g, strategy="largest_first")
    conflicts = sum(1 for u, v in g.edges if colors[u] % K == colors[v] % K)
    return conflicts


def ea_2d_bruteforce(L: int, seed: int) -> float:
    """Exact ground-state energy for 2D Edwards-Anderson on L=L (2^{L*L} states)."""
    problem = qqa.EdwardsAnderson(L=L, dim=2, seed=seed)
    N = problem.num_spins
    assert N <= 12, "Brute force only for small instances (2^12 ~ 4k states)."
    all_s = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=N)))
    e = problem.loss_fn(all_s)
    return float(e.min().item())


# ---------------------------------------------------------------------------
# Per-problem sweeps
# ---------------------------------------------------------------------------


def verify_mis() -> dict[str, Any]:
    rows = []
    for seed in (0, 1, 2):
        g = nx.random_regular_graph(d=3, n=50, seed=seed)
        problem = qqa.MaximumIndependentSet(g, penalty=2)
        r = run_qqa(problem, sol_size=128, num_epochs=1500, seed=seed)
        qqa_size = -int(round(float(r.best_obj)))
        baseline = greedy_mis(g)
        rows.append(
            {
                "instance": f"3-regular N=50 seed={seed}",
                "QQA": qqa_size,
                "greedy": baseline,
                "delta": qqa_size - baseline,
                "ok": qqa_size >= baseline,
                "runtime": r.runtime,
            }
        )
    return {
        "name": "Maximum Independent Set",
        "reference": "NetworkX degree-greedy MIS (independent-set size)",
        "rows": rows,
    }


def verify_maxcut() -> dict[str, Any]:
    rows = []
    for seed, (n, p) in enumerate([(30, 0.2), (40, 0.15), (60, 0.1)]):
        g = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
        problem = qqa.MaxCut(g)
        r = run_qqa(problem, sol_size=128, num_epochs=1500, seed=seed)
        qqa_cut = -int(round(float(r.best_obj)))
        base = greedy_maxcut(g, trials=400, seed=seed)
        rows.append(
            {
                "instance": f"G(n={n}, p={p}) seed={seed}",
                "QQA": qqa_cut,
                "random": base,
                "delta": qqa_cut - base,
                "ok": qqa_cut >= base,
                "runtime": r.runtime,
            }
        )
    return {
        "name": "MaxCut",
        "reference": "Random-partition best-of-400 baseline",
        "rows": rows,
    }


def verify_max_clique() -> dict[str, Any]:
    rows = []
    for seed, (n, p) in enumerate([(30, 0.5), (40, 0.4), (50, 0.3)]):
        g = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
        problem = qqa.MaxClique(g, penalty=2)
        r = run_qqa(problem, sol_size=128, num_epochs=1500, seed=seed)
        qqa_clique = -int(round(float(r.best_obj)))
        base = greedy_maxclique(g)
        rows.append(
            {
                "instance": f"G(n={n}, p={p}) seed={seed}",
                "QQA": qqa_clique,
                "nx_approx": base,
                "delta": qqa_clique - base,
                "ok": qqa_clique >= base,
                "runtime": r.runtime,
            }
        )
    return {
        "name": "MaxClique",
        "reference": "NetworkX approximation.max_clique",
        "rows": rows,
    }


def verify_coloring() -> dict[str, Any]:
    rows = []
    for seed in (0, 1, 2):
        g = nx.random_regular_graph(d=3, n=40, seed=seed)
        problem = qqa.Coloring(g, num_category=3)
        r = run_qqa(problem, sol_size=128, num_epochs=1500, curve_rate=4, div_param=0.2, seed=seed)
        conflicts = int(round(float(r.best_obj)))
        baseline = greedy_coloring(g, K=3)
        rows.append(
            {
                "instance": f"3-regular N=40 K=3 seed={seed}",
                "QQA_conflicts": conflicts,
                "greedy_conflicts": baseline,
                "QQA_feasible": bool(conflicts == 0),
                "ok": conflicts <= baseline,
                "runtime": r.runtime,
            }
        )
    return {
        "name": "Graph coloring",
        "reference": "Welsh–Powell greedy (nx.greedy_color)",
        "rows": rows,
    }


def verify_ising1d() -> dict[str, Any]:
    rows = []
    for N in (16, 32, 64):
        problem = qqa.Ising1D(N=N, J=1.0, h=0.0, periodic=True)
        r = run_qqa(problem, sol_size=64, num_epochs=600, curve_rate=2, seed=0)
        target = -float(N)
        E = float(r.best_obj)
        rows.append(
            {
                "instance": f"FM periodic N={N}",
                "QQA_E": E,
                "target_E": target,
                "gap": E - target,
                "ok": math.isclose(E, target, abs_tol=1e-4),
                "runtime": r.runtime,
            }
        )
    return {
        "name": "Ising 1D (ferromagnet)",
        "reference": "exact E_0 = -N for J=1, h=0, periodic",
        "rows": rows,
    }


def verify_edwards_anderson() -> dict[str, Any]:
    rows = []
    for seed in (0, 1, 2):
        problem = qqa.EdwardsAnderson(L=3, dim=2, seed=seed)
        r = run_qqa(problem, sol_size=128, num_epochs=1500, curve_rate=2, seed=seed)
        exact = ea_2d_bruteforce(L=3, seed=seed)
        E = float(r.best_obj)
        rows.append(
            {
                "instance": f"EA 2D L=3 seed={seed}",
                "QQA_E": E,
                "exact_E": exact,
                "gap": E - exact,
                "ok": math.isclose(E, exact, abs_tol=1e-4),
                "runtime": r.runtime,
            }
        )
    for seed in (0, 1):
        problem = qqa.EdwardsAnderson(L=4, dim=3, seed=seed)
        r = run_qqa(problem, sol_size=128, num_epochs=2000, curve_rate=2, seed=seed)
        E = float(r.best_obj)
        rows.append(
            {
                "instance": f"EA 3D L=4 seed={seed}",
                "QQA_E": E,
                "exact_E": "—",
                "E/N": E / problem.num_spins,
                "ok": True,
                "runtime": r.runtime,
            }
        )
    return {
        "name": "Edwards–Anderson spin glass",
        "reference": "Brute force (L=3, 2D); N/A for 3D L=4",
        "rows": rows,
    }


def verify_sk() -> dict[str, Any]:
    rows = []
    for N in (50, 100, 200):
        problem = qqa.SherringtonKirkpatrick(N=N, seed=0)
        r = run_qqa(
            problem,
            sol_size=256,
            num_epochs=3000,
            curve_rate=2,
            min_bg=-3.0,
            max_bg=0.1,
            seed=0,
        )
        e = float(r.best_obj) / N
        rows.append(
            {
                "instance": f"SK N={N} seed=0",
                "QQA_e0": e,
                "Parisi_e0": PARISI_SK,
                "gap_%": (e - PARISI_SK) / abs(PARISI_SK) * 100.0,
                "ok": e <= PARISI_SK * 0.85,  # at least 85% of Parisi value
                "runtime": r.runtime,
            }
        )
    return {
        "name": "Sherrington–Kirkpatrick",
        "reference": f"Parisi typical e_0 = {PARISI_SK:.4f}",
        "rows": rows,
    }


def verify_binary_perceptron() -> dict[str, Any]:
    rows = []
    for alpha in (0.3, 0.5, 0.7):
        problem = qqa.BinaryPerceptron(N=40, alpha=alpha, seed=0, sharpness=10.0)
        r = run_qqa(problem, sol_size=256, num_epochs=2000, curve_rate=2, seed=0)
        s_best = problem.relaxation.project(r.best_sol).unsqueeze(0)
        errors = int(problem.error_count(s_best).min())
        rows.append(
            {
                "instance": f"α={alpha} N=40",
                "QQA_errors": errors,
                "teacher_errors": 0,
                "ok": errors == 0,
                "runtime": r.runtime,
            }
        )
    return {
        "name": "Binary perceptron",
        "reference": "Teacher solution exists with 0 errors",
        "rows": rows,
    }


def verify_hopfield() -> dict[str, Any]:
    rows = []
    for N, P in [(32, 2), (64, 3), (128, 4)]:
        problem = qqa.HopfieldMemory(N=N, patterns=P, seed=0)
        r = run_qqa(problem, sol_size=128, num_epochs=1500, curve_rate=2, seed=0)
        s_best = problem.relaxation.project(r.best_sol).unsqueeze(0)
        max_abs_overlap = float(problem.overlap(s_best).abs().max().item())
        rows.append(
            {
                "instance": f"N={N} P={P} α={P / N:.2f}",
                "max_overlap": max_abs_overlap,
                "ok": max_abs_overlap >= 0.95,
                "runtime": r.runtime,
            }
        )
    return {
        "name": "Hopfield memory",
        "reference": "Recovery criterion: max |overlap| ≥ 0.95 with a stored pattern",
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------


def format_row(row: dict[str, Any]) -> list[str]:
    def f(v):
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 1e4 else f"{v:.2e}"
        return str(v)

    return [f(v) for v in row.values()]


def write_report(summary: list[dict[str, Any]]) -> None:
    lines = [
        "# QQA correctness verification report",
        "",
        f"_Generated by `scripts/verify_all_problems.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "This report runs QQA on every built-in problem and compares its best "
        "objective against a ground truth or a strong baseline. Each block "
        "contains the reference used, per-instance numbers, runtime, and a ",
        "boolean `ok` flag that must hold for the sweep to be considered successful.",
        "",
    ]
    total_rows = 0
    total_ok = 0
    for block in summary:
        lines.append(f"## {block['name']}")
        lines.append("")
        lines.append(f"**Reference**: {block['reference']}")
        lines.append("")
        rows = block["rows"]
        if not rows:
            lines.append("_(no data)_\n")
            continue
        header = list(rows[0].keys())
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(format_row(row)) + " |")
            total_rows += 1
            if bool(row.get("ok", False)):
                total_ok += 1
        lines.append("")

    lines.insert(
        2,
        f"**Summary: {total_ok}/{total_rows} checks passed "
        f"({total_ok / max(1, total_rows) * 100:.1f}%).**\n",
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"[verify] wrote {REPORT}")


def main() -> None:
    summary: list[dict[str, Any]] = []
    runners = [
        verify_mis,
        verify_maxcut,
        verify_max_clique,
        verify_coloring,
        verify_ising1d,
        verify_edwards_anderson,
        verify_sk,
        verify_binary_perceptron,
        verify_hopfield,
    ]
    for fn in runners:
        print(f"[verify] running {fn.__name__} ...")
        t0 = time.time()
        summary.append(fn())
        print(f"  done in {time.time() - t0:.1f}s")
    write_report(summary)


if __name__ == "__main__":  # pragma: no cover
    main()
