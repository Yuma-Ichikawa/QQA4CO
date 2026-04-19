"""Small utilities: seed control, random graph generation, MIS evaluation."""

from __future__ import annotations

import random
from itertools import combinations, islice
from time import time
from typing import Any

import networkx as nx
import numpy as np
import torch


def require_cuda_if_requested(device: str | torch.device) -> None:
    """Raise a friendly :class:`RuntimeError` if a CUDA device is requested
    but CUDA is unavailable.

    All solver entry points (``qqa.anneal``, ``qqa.simulated_annealing``,
    ``train_cra_pi_gnn``, ``train_cpra_pi_gnn``) call this once at the
    start so users see a single, copy-pasteable message instead of a
    cryptic stack trace from deep inside ``.to(device)``.
    """
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled torch build, or pass device='cpu'."
        )


def safe_score_summary(problem: Any, sol: Any, fallback_obj: float) -> dict:
    """Call ``problem.score_summary(sol)`` but never let it abort a solve.

    All four solver backends wrap their final ``score_summary`` call in
    the same ``try/except`` that returns a uniform ``{"label": "loss",
    "feasible": False, "extra": {"error": ...}}`` dict on failure. This
    helper centralises that contract so the fields stay in lock-step
    across :mod:`qqa.annealing`, :mod:`qqa.sa`, and
    :mod:`qqa.pignn.trainer`.
    """
    try:
        return problem.score_summary(sol)
    except Exception as exc:  # noqa: BLE001 - surface but never abort
        # ``feasible=False`` (not True) so that downstream UIs / CLI don't
        # mis-advertise an unchecked solution as valid just because the
        # scorer itself crashed.
        return {
            "label": "loss",
            "value": float(fallback_obj),
            "unit": "",
            "feasible": False,
            "extra": {"error": str(exc)},
        }


def fix_seed(seed: int) -> None:
    """Seed Python/Numpy/Torch (CPU + CUDA) for deterministic runs.

    .. note::
       This call flips ``torch.backends.cudnn.deterministic`` to ``True`` for
       the whole process, which disables some CuDNN auto-tuning and may slow
       non-QQA code sharing the same interpreter. Call it once near the start
       of your script (or test session) and not inside hot loops.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def enable_tf32(enabled: bool = True) -> None:
    """Toggle TF32 matmul / cuDNN paths for Ampere+ GPUs.

    QQA's hot path is dominated by dense ``einsum`` / ``bmm`` over the QUBO
    matrix. On Ampere (A100), Hopper (H100) and Blackwell GPUs, allowing
    TF32 typically yields a **1.1x–1.5x** wall-clock improvement on those
    matmuls with negligible accuracy loss for the binary projection
    downstream.

    The library deliberately **does not** flip these flags on import (that
    would be a global side effect). Call this once at the top of a benchmark
    script when you want maximum speed::

        import qqa
        qqa.enable_tf32()
        qqa.fix_seed(0)
        result = qqa.anneal(problem, sol_size=256, num_epochs=2000, device="cuda")

    Pass ``enabled=False`` to restore PyTorch's default precision policy.
    Has no effect on CPU or pre-Ampere GPUs.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)


def generate_graph(
    n: int,
    d: int | None = None,
    p: float | None = None,
    graph_type: str = "reg",
    random_seed: int = 0,
) -> nx.Graph:
    """Generate a random graph.

    Parameters
    ----------
    graph_type : {"reg", "prob", "erdos"}
        * ``reg``   -- ``n``-node, ``d``-regular graph
        * ``prob``  -- fast G(n, p)
        * ``erdos`` -- classic Erdős-Rényi G(n, p)
    """
    if graph_type == "reg":
        return nx.random_regular_graph(d=d, n=n, seed=random_seed)
    if graph_type == "prob":
        return nx.fast_gnp_random_graph(n, p, seed=random_seed)
    if graph_type == "erdos":
        return nx.erdos_renyi_graph(n, p, seed=random_seed)
    raise ValueError(f"Unknown graph_type: {graph_type!r}")


def _gen_combinations(combs, chunk_size: int):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])


# -----------------------------------------------------------------------------
# Legacy graph-evaluation helpers
# -----------------------------------------------------------------------------
# The functions below predate the :mod:`qqa.problems` API and are no longer
# referenced anywhere inside the library. They remain here because external
# callers may import them as ``from qqa.utils import approximate_mis`` etc.
# Prefer ``problem.score_summary(...)`` from any modern :class:`COProblem`
# subclass instead — it returns a richer dict with feasibility info.


def approximate_mis(nx_graph: nx.Graph):
    """Run NetworkX's greedy MIS approximation as a quick baseline.

    Returns
    -------
    bitstring : list[int]
    size : int
    num_violations : int
    runtime : float
    """
    t0 = time()
    ind_set = nx.algorithms.approximation.clique.maximum_independent_set(nx_graph)
    elapsed = time() - t0
    bitstring = [1 if v in ind_set else 0 for v in sorted(nx_graph.nodes)]
    edge_set = set(nx_graph.edges)
    violations = 0
    for chunk in _gen_combinations(combinations(ind_set, 2), 100_000):
        violations += len(set(chunk).intersection(edge_set))
    return bitstring, len(ind_set), violations, elapsed


def mis_stats(bitstring, nx_graph: nx.Graph) -> tuple[int, set, int]:
    """Return (size, independent-set nodes, #violations) for a given bitstring."""
    vs = [int(b) for b in bitstring]
    ind_set = {node for node, entry in enumerate(vs) if entry == 1}
    edge_set = set(nx_graph.edges)
    violations = 0
    for chunk in _gen_combinations(combinations(ind_set, 2), 100_000):
        violations += len(set(chunk).intersection(edge_set))
    return sum(vs), ind_set, violations


def max_cut_stats(bitstring, nx_graph: nx.Graph):
    """Return (cut_size, [S0, S1], cut_edges, uncut_edges) for a bitstring."""
    vs = [int(b) for b in bitstring]
    S0 = [n for n in nx_graph.nodes if not vs[n]]
    S1 = [n for n in nx_graph.nodes if vs[n]]
    cut = [(u, v) for u, v in nx_graph.edges if vs[u] != vs[v]]
    uncut = [(u, v) for u, v in nx_graph.edges if vs[u] == vs[v]]
    return len(cut), [S0, S1], cut, uncut
