"""Graph extraction & PyG conversion utilities for :mod:`qqa.pignn`.

CRA-PI-GNN is a *graph* neural network method, so it only applies to QQA
problems whose underlying combinatorial structure is a graph (MIS,
MaxClique, MaxCut, VertexCover, GraphBisection). Spin-glass problems
defined by a coupling matrix ``J`` (Edwards–Anderson, SK, perceptron,
Hopfield, …) and pure categorical / permutation problems (TSP, QAP,
NQueens, Knapsack, …) have no node-edge structure to convolve over and
are silently rejected here with a clear ``TypeError``.
"""

from __future__ import annotations

import networkx as nx
import torch

GRAPH_PROBLEM_HINT = (
    "Supported problems expose their networkx graph as ``problem.nx_graph`` "
    "(MaximumIndependentSet, MaxClique, MaxCut) or ``problem.graph`` "
    "(VertexCover, GraphBisection). Pass ``nx_graph=...`` explicitly if your "
    "custom problem stores the graph under a different attribute."
)

# Attribute names searched, in order, when extracting the underlying nx.Graph.
# Kept narrow on purpose so we don't pick up unrelated attributes that happen
# to be named ``g`` / ``G`` on user subclasses.
_GRAPH_ATTRS: tuple[str, ...] = ("nx_graph", "graph")


def extract_nx_graph(problem, override: nx.Graph | None = None) -> nx.Graph:
    """Return the networkx graph backing a QQA problem.

    Parameters
    ----------
    problem:
        A :class:`~qqa.problems.COProblem` instance. The function checks
        ``problem.nx_graph`` first (used by ``MaximumIndependentSet``,
        ``MaxClique``, ``MaxCut``) and falls back to ``problem.graph``
        (used by ``VertexCover``, ``GraphBisection``).
    override:
        If supplied, used directly (for the rare case where a custom
        problem stores its graph elsewhere). The caller is then
        responsible for ensuring node labels are ``0..N-1`` and that the
        graph matches ``problem``'s QUBO matrix.

    Returns
    -------
    networkx.Graph
        With node labels ``0..N-1``.

    Raises
    ------
    TypeError
        If neither ``override`` nor any of the supported attribute names
        on ``problem`` resolve to a ``networkx.Graph``. The error lists
        the supported problem families so the user can diagnose at a
        glance.
    """
    if override is not None:
        return override
    for attr in _GRAPH_ATTRS:
        g = getattr(problem, attr, None)
        if isinstance(g, nx.Graph):
            return g
    raise TypeError(
        f"qqa.pignn requires a graph-based problem; "
        f"{type(problem).__name__} does not expose any of {list(_GRAPH_ATTRS)}. "
        + GRAPH_PROBLEM_HINT
    )


def nx_to_edge_index(graph: nx.Graph, device: str | torch.device = "cpu") -> torch.Tensor:
    """Convert a networkx graph to a symmetric PyG ``edge_index``.

    PyG's :class:`~torch_geometric.nn.GCNConv` expects ``edge_index`` of
    shape ``(2, 2|E|)`` listing both ``(u, v)`` and ``(v, u)`` for every
    undirected edge. Self-loops are not added here — :class:`GCNConv`
    inserts them internally when ``add_self_loops=True`` (the default).

    Parameters
    ----------
    graph:
        Undirected networkx graph with node labels in ``0..N-1``.
    device:
        Target torch device for the returned tensor.

    Returns
    -------
    torch.Tensor
        ``(2, 2|E|)`` ``long`` tensor on ``device``.
    """
    if graph.number_of_edges() == 0:
        # Empty graphs would produce a (2, 0) tensor; PyG handles that
        # but downstream slicing/loss math breaks. Surface it loudly.
        raise ValueError(
            "Graph has no edges; CRA-PI-GNN is undefined on an edgeless graph. "
            "Use ``qqa.anneal`` (which handles trivial cases gracefully) instead."
        )
    src: list[int] = []
    dst: list[int] = []
    for u, v in graph.edges:
        src.append(int(u))
        dst.append(int(v))
        src.append(int(v))
        dst.append(int(u))
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return edge_index
