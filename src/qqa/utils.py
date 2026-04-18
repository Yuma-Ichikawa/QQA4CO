"""Small utilities: seed control, random graph generation, MIS evaluation."""

from __future__ import annotations

import random
from itertools import combinations, islice
from time import time

import networkx as nx
import numpy as np
import torch


def fix_seed(seed: int) -> None:
    """Seed Python/Numpy/Torch (CPU + CUDA) for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
