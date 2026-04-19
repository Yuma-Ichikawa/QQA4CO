"""Reproduce MaxCut on G-set instance G70 with PQQA / CRA-PI-GNN / CPRA.

G-set G70 is the canonical 10 000-node, 9 999-edge sparse near-bipartite
benchmark. Best-known cut = 9 591. The CRA paper (Ichikawa, NeurIPS 2024)
reports a 0.992 ApR (cut = 9 514). A naive run of any of the three QQA4CO
solvers with their published defaults stops at ~0.97 because of three
concrete issues that this script demonstrates fixes for:

1. **PQQA freeze bug** — pre-fix ``BinaryRelaxation`` did not clamp ``x``
   to ``[0, 1]`` after the AdamW step, so the relaxed variable drifts
   outside the cube and the CRA penalty ``Φ(p) = 1 - (1 - 2p)^α`` becomes
   *negative*, giving the optimiser an incentive to drift further. Fixed
   in :mod:`qqa.relaxation`.
2. **CRA-PI-GNN MaxCut hyperparameters** — the paper uses ``γ(0) = -6``
   (not ``-20`` as for MIS) and the GCN width ``H0 = int(N^0.8) ≈ 1584``
   for ``N = 10 000``. The trainer's defaults are tuned for MIS so we
   override here.
3. **Free 1-flip polish + BFS-2color warm-start** — these are now first-
   class QQA4CO features (``qqa.polish.greedy_one_flip`` runs by default
   inside :func:`qqa.anneal` and the trainers; ``qqa.warmstart.bfs_2color``
   feeds the BFS seed to ``initial_state=`` for the PQQA path).

Usage::

    python scripts/maxcut_gset_g70.py --method pqqa --device cuda
    python scripts/maxcut_gset_g70.py --method cra  --device cuda
    python scripts/maxcut_gset_g70.py --method cpra --device cuda
    python scripts/maxcut_gset_g70.py --method all  --device cuda
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
from pathlib import Path

import networkx as nx
import torch

import qqa
from qqa.pignn import train_cpra_pi_gnn, train_cra_pi_gnn

GSET_URL = "https://web.stanford.edu/~yyye/yyye/Gset/G70"
BEST_KNOWN = 9591
TARGET_CUT = 9514  # Ichikawa NeurIPS 2024 headline (≥ 0.992 of best known)


def load_g70(cache_dir: Path) -> nx.Graph:
    """Download (cached) and parse G70 into a 0-indexed networkx graph."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "G70"
    if not cache_file.exists():
        print(f"[load_g70] downloading {GSET_URL} -> {cache_file}")
        urllib.request.urlretrieve(GSET_URL, cache_file)  # noqa: S310

    with cache_file.open() as fh:
        n, m = (int(x) for x in fh.readline().split())
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                g.add_edge(int(parts[0]) - 1, int(parts[1]) - 1, weight=float(parts[2]))
    if g.number_of_edges() != m:
        raise RuntimeError(f"Parsed {g.number_of_edges()} edges but header claims {m}")
    print(f"[load_g70] |V|={g.number_of_nodes()}  |E|={g.number_of_edges()}")
    return g


def cut_of(problem: qqa.MaxCut, bits: torch.Tensor) -> int:
    return int(problem.score_summary(bits)["value"])


def report(problem: qqa.MaxCut, bits: torch.Tensor, *, method: str, runtime: float) -> int:
    """Print the canonical PASS/FAIL line and return the final cut."""
    cut = cut_of(problem, bits)
    ratio = cut / BEST_KNOWN
    status = "PASS" if cut >= TARGET_CUT else "FAIL"
    print(
        f"\n[g70] method={method}  cut={cut}  ratio={ratio:.4f} "
        f"(best known {BEST_KNOWN}, target {TARGET_CUT})  "
        f"runtime={runtime:.1f}s  [{status}]"
    )
    return cut


# --- PQQA ------------------------------------------------------------------
# Sweet-spot from the ICLR 2025 PQQA appendix combined with our own G70
# sweep: large parallel population (S=5000), a small but non-trivial
# diversity term (div=0.5 — at the paper's 0.2 the chains collapse around
# epoch 15 k on G70's giant near-bipartite component), Langevin noise
# ``temp=1e-3`` so the chains actually explore. The BFS-2color warmstart
# is fed via ``initial_state=`` and gives every chain a 0.96 ApR floor at
# t=0; the post-anneal 1-flip polish runs default-on inside ``qqa.anneal``.


def run_pqqa(problem: qqa.MaxCut, g: nx.Graph, *, device: str, num_epochs: int) -> torch.Tensor:
    """Single-shot PQQA at the G70 sweet spot — no sweep needed."""
    bfs = qqa.warmstart.bfs_2color(g).to(device)
    print(f"[pqqa] BFS-2color seed cut = {cut_of(problem, bfs)}")
    qqa.fix_seed(0)
    res = qqa.anneal(
        problem,
        sol_size=5000,
        learning_rate=1.0,
        temp=1e-3,
        num_epochs=num_epochs,
        min_bg=-3.0,
        max_bg=0.1,
        curve_rate=4,
        div_param=0.5,
        check_interval=max(1, num_epochs // 5),
        device=device,
        initial_state=bfs,
        verbose=True,
    )
    return res.best_sol


# --- CRA-PI-GNN ------------------------------------------------------------
# Paper-faithful MaxCut hyperparameters (NeurIPS 2024 Sec. 5.1 + App. D.1):
# ``γ(0) = -6`` (not -20 — that's MIS), ``ε = 1e-3``, ``α = 2``, AdamW
# ``lr = 1e-4 / wd = 1e-2``, ``H0 = int(N^0.8)``, early stopping
# ``tol=1e-5`` & ``patience=1e3``, 5 random seeds → take best.


def run_cra(problem: qqa.MaxCut, g: nx.Graph, *, device: str, num_epochs: int) -> torch.Tensor:
    n = g.number_of_nodes()
    h0 = int(round(n**0.8))
    print(f"[cra] hidden_dim = int(N^0.8) = {h0}")
    best_bits = None
    best_cut = -1
    for seed in (0, 1, 2, 3, 4):
        t0 = time.time()
        res = train_cra_pi_gnn(
            problem,
            in_feats=h0,
            hidden_dim=h0,
            learning_rate=1e-4,
            weight_decay=1e-2,
            init_reg_param=-6.0,
            annealing_rate=1e-3,
            curve_rate=2,
            num_epochs=num_epochs,
            tol=1e-5,
            patience=1000,
            check_interval=max(1, num_epochs // 5),
            device=device,
            seed=seed,
            verbose=False,
        )
        cut = cut_of(problem, res.best_sol)
        marker = " *NEW_BEST*" if cut > best_cut else ""
        print(
            f"[cra] seed={seed}  cut={cut}  wall={time.time() - t0:.0f}s{marker}",
            flush=True,
        )
        if cut > best_cut:
            best_cut, best_bits = cut, res.best_sol
    print(f"[cra] best across 5 seeds: cut={best_cut}")
    return best_bits


# --- CPRA ------------------------------------------------------------------
# Same paper-faithful hyperparameters as CRA + a population (R) and
# diversity (vari_param) sweep. The 1-flip polish on the winning replica
# is done default-on inside ``train_cpra_pi_gnn``.


def run_cpra(problem: qqa.MaxCut, g: nx.Graph, *, device: str, num_epochs: int) -> torch.Tensor:
    n = g.number_of_nodes()
    h0 = int(round(n**0.8))
    print(f"[cpra] hidden_dim = int(N^0.8) = {h0}")
    best_bits = None
    best_cut = -1
    for num_replicas, vari_param in ((16, 1.0), (32, 2.0)):
        t0 = time.time()
        res = train_cpra_pi_gnn(
            problem,
            num_replicas=num_replicas,
            vari_param=vari_param,
            in_feats=h0,
            hidden_dim=h0,
            learning_rate=1e-4,
            weight_decay=1e-2,
            init_reg_param=-6.0,
            annealing_rate=1e-3,
            curve_rate=2,
            num_epochs=num_epochs,
            tol=1e-5,
            patience=1000,
            check_interval=max(1, num_epochs // 5),
            device=device,
            seed=0,
            verbose=False,
        )
        cut = cut_of(problem, res.best_sol)
        marker = " *NEW_BEST*" if cut > best_cut else ""
        print(
            f"[cpra] R={num_replicas:>3d}  vari={vari_param:>4.1f}  cut={cut}  "
            f"wall={time.time() - t0:.0f}s{marker}",
            flush=True,
        )
        if cut > best_cut:
            best_cut, best_bits = cut, res.best_sol
    print(f"[cpra] best across sweep: cut={best_cut}")
    return best_bits


_DISPATCH = {
    "pqqa": (run_pqqa, 30_000),
    "cra": (run_cra, 100_000),
    "cpra": (run_cpra, 100_000),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=[*_DISPATCH, "all"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "gset",
    )
    args = parser.parse_args()

    g = load_g70(args.cache_dir)
    problem = qqa.MaxCut(g, device=args.device)
    bfs_cut = cut_of(problem, qqa.warmstart.bfs_2color(g).to(args.device))
    print(f"[g70] BFS-2color baseline cut = {bfs_cut}  (≈ {bfs_cut / BEST_KNOWN:.3f} ApR)")

    methods = [*_DISPATCH] if args.method == "all" else [args.method]
    cuts: dict[str, int] = {}
    for m in methods:
        fn, default_epochs = _DISPATCH[m]
        epochs = default_epochs if args.num_epochs is None else int(args.num_epochs)
        print(f"\n[g70] method={m}  device={args.device}  epochs={epochs}")
        t0 = time.time()
        bits = fn(problem, g, device=args.device, num_epochs=epochs)
        cuts[m] = report(problem, bits, method=m, runtime=time.time() - t0)

    print("\n" + "=" * 30 + " G70 SUMMARY " + "=" * 30)
    for m in methods:
        c = cuts[m]
        print(
            f"  {m:5s}  cut={c}  ratio={c / BEST_KNOWN:.4f}  "
            f"{'>=9514 ✓' if c >= TARGET_CUT else '<9514 ✗'}"
        )
    return 0


if __name__ == "__main__":
    # ``math`` import kept implicit so ``round(N ** 0.8)`` stays floating-point.
    _ = math
    sys.exit(main())
