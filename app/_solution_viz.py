"""Problem-specific solution visualisation for the QQA dashboard.

Each supported problem kind gets a dedicated Plotly figure that conveys the
shape of the solution rather than the raw loss number: a highlighted
independent set, a TSP tour, an N-Queens board, coloured partitions, a
retrieved Hopfield pattern, etc.

All renderers follow the same contract:

* they accept ``(problem, result, cfg)``;
* they read ``result.best_sol`` (a single solution tensor, not a batch)
  plus any problem-specific auxiliary data from ``problem``;
* they emit one or two ``st.plotly_chart`` calls decorated with
  ``plotly_layout()`` so the figure stays consistent with the current
  light / dark theme.

A single entry point ``render_solution_view(problem, result, cfg)`` dispatches
on ``cfg["kind"]`` and falls back to a generic heatmap if no specialist
renderer exists.
"""

from __future__ import annotations

import contextlib
from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from _common import hex_to_rgba, palette, plotly_layout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_numpy(x) -> np.ndarray:
    """Return a CPU ``numpy`` view of a torch tensor / array / scalar."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def _graph_layout(g: nx.Graph, seed: int = 0) -> dict:
    """Cache-free 2-D layout suitable for the solution plots."""
    if g.number_of_nodes() <= 40:
        return nx.kamada_kawai_layout(g)
    return nx.spring_layout(g, seed=seed, k=1.6 / np.sqrt(max(2, g.number_of_nodes())))


def _edge_traces(g: nx.Graph, pos: dict, *, colour: str, width: float = 1.0, dash: str = "solid"):
    xs, ys = [], []
    for u, v in g.edges:
        xs.extend([pos[u][0], pos[v][0], None])
        ys.extend([pos[u][1], pos[v][1], None])
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": colour, "width": width, "dash": dash},
        hoverinfo="skip",
        showlegend=False,
    )


def _graph_node_traces(
    g: nx.Graph,
    pos: dict,
    *,
    colours,
    size=14,
    border: str,
    text=None,
) -> go.Scatter:
    xs = [pos[n][0] for n in g.nodes]
    ys = [pos[n][1] for n in g.nodes]
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers" if text is None else "markers+text",
        marker={
            "size": size,
            "color": colours,
            "line": {"color": border, "width": 1.2},
        },
        text=text,
        textposition="middle center",
        textfont={"size": 10, "color": "#0f172a"},
        hoverinfo="text",
        hovertext=[f"node {n}" for n in g.nodes],
        showlegend=False,
    )


def _axes_off() -> dict:
    return {
        "xaxis": {"visible": False, "showgrid": False, "zeroline": False},
        "yaxis": {
            "visible": False,
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    }


def _render(fig: go.Figure, *, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key)


# ---------------------------------------------------------------------------
# Binary graph problems
# ---------------------------------------------------------------------------


def _render_binary_graph(
    problem,
    result,
    cfg,
    *,
    title: str,
    selected_label: str,
    unselected_label: str,
    key: str,
) -> None:
    g: nx.Graph = problem.graph if hasattr(problem, "graph") else problem.nx_graph
    sol = _as_numpy(result.best_sol).astype(float).reshape(-1)
    x = (sol > 0.5).astype(int)
    pos = _graph_layout(g)
    p = palette()
    accent = p["palette"][0]
    muted = p["palette"][1]

    nodes = list(g.nodes)
    node_colour = [accent if x[i] else hex_to_rgba(muted, 0.35) for i in range(len(nodes))]
    border = "#0f172a" if st.session_state.get("theme", "light") == "light" else "#f8fafc"

    fig = go.Figure()
    fig.add_trace(
        _edge_traces(g, pos, colour=hex_to_rgba(p["muted"], 0.55 if len(g.edges) < 80 else 0.35))
    )
    fig.add_trace(_graph_node_traces(g, pos, colours=node_colour, border=border, size=15))
    fig.update_layout(
        **plotly_layout(
            title={"text": title, "x": 0.5},
            height=480,
            showlegend=False,
            **_axes_off(),
        )
    )

    n_sel = int(x.sum())
    n_tot = len(nodes)
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=(
            f"<b>{selected_label}</b>: {n_sel}/{n_tot}  ·  "
            f"<b>{unselected_label}</b>: {n_tot - n_sel}"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0)",
        font={"size": 12},
    )
    _render(fig, key=key)


def render_mis(problem, result, cfg) -> None:
    _render_binary_graph(
        problem,
        result,
        cfg,
        title=f"Maximum Independent Set · |IS| = {int(_as_numpy(result.best_sol).sum())}",
        selected_label="IS",
        unselected_label="outside",
        key="soln_mis",
    )


def render_max_clique(problem, result, cfg) -> None:
    _render_binary_graph(
        problem,
        result,
        cfg,
        title=f"Max Clique · |C| = {int(_as_numpy(result.best_sol).sum())}",
        selected_label="clique",
        unselected_label="outside",
        key="soln_clique",
    )


def render_vertex_cover(problem, result, cfg) -> None:
    _render_binary_graph(
        problem,
        result,
        cfg,
        title=f"Vertex Cover · |VC| = {int(_as_numpy(result.best_sol).sum())}",
        selected_label="cover",
        unselected_label="uncovered",
        key="soln_vc",
    )


def render_max_cut(problem, result, cfg) -> None:
    g: nx.Graph = problem.graph if hasattr(problem, "graph") else problem.nx_graph
    sol = _as_numpy(result.best_sol).astype(float).reshape(-1)
    x = (sol > 0.5).astype(int)
    pos = _graph_layout(g)
    p = palette()
    c_cut = p["palette"][1]  # accent2 for cut edges
    c_nocut = hex_to_rgba(p["muted"], 0.35)
    c_a = p["palette"][0]
    c_b = p["palette"][2]

    nodes = list(g.nodes)
    node_colour = [c_a if x[i] == 0 else c_b for i in range(len(nodes))]
    border = "#0f172a" if st.session_state.get("theme", "light") == "light" else "#f8fafc"

    cut_x, cut_y, nc_x, nc_y = [], [], [], []
    for u, v in g.edges:
        if x[u] != x[v]:
            cut_x.extend([pos[u][0], pos[v][0], None])
            cut_y.extend([pos[u][1], pos[v][1], None])
        else:
            nc_x.extend([pos[u][0], pos[v][0], None])
            nc_y.extend([pos[u][1], pos[v][1], None])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nc_x,
            y=nc_y,
            mode="lines",
            line={"color": c_nocut, "width": 1.0},
            hoverinfo="skip",
            name="within partition",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cut_x,
            y=cut_y,
            mode="lines",
            line={"color": c_cut, "width": 2.2},
            hoverinfo="skip",
            name="cut edge",
            showlegend=True,
        )
    )
    fig.add_trace(_graph_node_traces(g, pos, colours=node_colour, border=border, size=15))

    n_cut = sum(1 for u, v in g.edges if x[u] != x[v])
    fig.update_layout(
        **plotly_layout(
            title={"text": f"Max-Cut · |cut edges| = {n_cut}", "x": 0.5},
            height=480,
            showlegend=True,
            legend={"x": 0.01, "y": 0.02, "bgcolor": "rgba(255,255,255,0.0)"},
            **_axes_off(),
        )
    )
    _render(fig, key="soln_maxcut")


def render_graph_bisection(problem, result, cfg) -> None:
    g: nx.Graph = problem.graph if hasattr(problem, "graph") else problem.nx_graph
    sol = _as_numpy(result.best_sol).astype(float).reshape(-1)
    x = (sol > 0.5).astype(int)
    pos = _graph_layout(g)
    p = palette()
    c_a = p["palette"][0]
    c_b = p["palette"][2]
    border = "#0f172a" if st.session_state.get("theme", "light") == "light" else "#f8fafc"
    nodes = list(g.nodes)
    node_colour = [c_a if x[i] == 0 else c_b for i in range(len(nodes))]

    cut_x, cut_y, nc_x, nc_y = [], [], [], []
    for u, v in g.edges:
        if x[u] != x[v]:
            cut_x.extend([pos[u][0], pos[v][0], None])
            cut_y.extend([pos[u][1], pos[v][1], None])
        else:
            nc_x.extend([pos[u][0], pos[v][0], None])
            nc_y.extend([pos[u][1], pos[v][1], None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nc_x,
            y=nc_y,
            mode="lines",
            line={"color": hex_to_rgba(p["muted"], 0.35), "width": 1.0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cut_x,
            y=cut_y,
            mode="lines",
            line={"color": p["palette"][1], "width": 2.0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(_graph_node_traces(g, pos, colours=node_colour, border=border, size=15))

    size_a = int((x == 0).sum())
    size_b = int((x == 1).sum())
    n_cut = sum(1 for u, v in g.edges if x[u] != x[v])
    fig.update_layout(
        **plotly_layout(
            title={
                "text": f"Graph bisection · cut = {n_cut}, |A| = {size_a}, |B| = {size_b}",
                "x": 0.5,
            },
            height=480,
            showlegend=False,
            **_axes_off(),
        )
    )
    _render(fig, key="soln_bisection")


def render_coloring(problem, result, cfg) -> None:
    g: nx.Graph = problem.graph if hasattr(problem, "graph") else problem.nx_graph
    sol = _as_numpy(result.best_sol)
    colours_idx = sol.argmax(axis=-1) if sol.ndim == 2 else sol.astype(int)
    pos = _graph_layout(g)
    p = palette()
    k = int(max(3, colours_idx.max() + 1))
    wheel = p["palette"] + [p["accent"], p["accent2"], "#9333ea", "#059669", "#dc2626"]
    node_colour = [wheel[int(colours_idx[i]) % len(wheel)] for i in range(len(colours_idx))]
    border = "#0f172a" if st.session_state.get("theme", "light") == "light" else "#f8fafc"

    conflicts = [(u, v) for u, v in g.edges if colours_idx[u] == colours_idx[v]]
    fig = go.Figure()
    fig.add_trace(
        _edge_traces(g, pos, colour=hex_to_rgba(p["muted"], 0.35 if len(g.edges) < 120 else 0.22))
    )
    if conflicts:
        cx, cy = [], []
        for u, v in conflicts:
            cx.extend([pos[u][0], pos[v][0], None])
            cy.extend([pos[u][1], pos[v][1], None])
        fig.add_trace(
            go.Scatter(
                x=cx,
                y=cy,
                mode="lines",
                line={"color": "#dc2626", "width": 2.4},
                hoverinfo="skip",
                name="conflict",
            )
        )
    fig.add_trace(_graph_node_traces(g, pos, colours=node_colour, border=border, size=15))
    fig.update_layout(
        **plotly_layout(
            title={
                "text": (
                    f"Graph colouring · used {len(set(int(c) for c in colours_idx))}/{k} colours "
                    f"· {len(conflicts)} conflicts"
                ),
                "x": 0.5,
            },
            height=480,
            showlegend=False,
            **_axes_off(),
        )
    )
    _render(fig, key="soln_coloring")


# ---------------------------------------------------------------------------
# Permutation / assignment problems
# ---------------------------------------------------------------------------


def render_tsp(problem, result, cfg) -> None:
    sol = _as_numpy(result.best_sol)  # (N, N) categorical
    tour = sol.argmax(axis=-1) if sol.ndim == 2 else sol.astype(int)
    coords = _as_numpy(problem.coords)  # (N, 2)
    p = palette()

    # Close the loop for plotting.
    ordered = np.concatenate([tour, tour[:1]])
    xs = coords[ordered, 0]
    ys = coords[ordered, 1]
    dist = float(result.score.get("value", 0.0))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line={"color": p["palette"][0], "width": 2.2},
            marker={"size": 11, "color": p["palette"][1], "line": {"color": "#0f172a", "width": 1}},
            name="tour",
            showlegend=False,
        )
    )
    # Start marker (larger, distinct colour) at position 0.
    fig.add_trace(
        go.Scatter(
            x=[coords[tour[0], 0]],
            y=[coords[tour[0], 1]],
            mode="markers",
            marker={
                "size": 18,
                "color": p["palette"][2],
                "line": {"color": "#0f172a", "width": 1.4},
                "symbol": "star",
            },
            name="start",
        )
    )
    # Annotate each city with its visit order.
    for step, city in enumerate(tour):
        fig.add_annotation(
            x=coords[city, 0],
            y=coords[city, 1],
            text=str(step),
            font={"size": 10, "color": p["text"]},
            showarrow=False,
            yshift=14,
        )

    fig.update_layout(
        **plotly_layout(
            title={"text": f"TSP tour · length ≈ {dist:.4g}", "x": 0.5},
            height=500,
            showlegend=False,
            xaxis={"scaleanchor": "y", "scaleratio": 1, "showgrid": True},
            yaxis={"showgrid": True},
        )
    )
    _render(fig, key="soln_tsp")


def render_qap(problem, result, cfg) -> None:
    sol = _as_numpy(result.best_sol)
    assignment = sol.argmax(axis=-1) if sol.ndim == 2 else sol.astype(int)
    N = int(getattr(problem, "N", len(assignment)))
    M = np.zeros((N, N))
    for facility, location in enumerate(assignment):
        if 0 <= int(location) < N:
            M[facility, int(location)] = 1.0

    fig = go.Figure(
        data=go.Heatmap(
            z=M,
            colorscale=[[0.0, "rgba(0,0,0,0.04)"], [1.0, palette()["palette"][0]]],
            showscale=False,
            xgap=1,
            ygap=1,
        )
    )
    for facility, location in enumerate(assignment):
        fig.add_annotation(
            x=int(location),
            y=facility,
            text="●",
            showarrow=False,
            font={"size": 14, "color": "#ffffff"},
        )
    dist = float(result.score.get("value", 0.0))
    fig.update_layout(
        **plotly_layout(
            title={"text": f"QAP assignment · cost ≈ {dist:.4g}", "x": 0.5},
            height=460,
            xaxis_title="Location",
            yaxis_title="Facility",
            xaxis={"tickmode": "linear", "dtick": 1},
            yaxis={"tickmode": "linear", "dtick": 1, "autorange": "reversed"},
        )
    )
    _render(fig, key="soln_qap")


def render_nqueens(problem, result, cfg) -> None:
    sol = _as_numpy(result.best_sol)
    cols_per_row = sol.argmax(axis=-1) if sol.ndim == 2 else sol.astype(int)
    N = int(getattr(problem, "N", len(cols_per_row)))

    board = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            board[r, c] = (r + c) % 2  # checkerboard 0/1
    p = palette()
    light = "#f1e9d0" if st.session_state.get("theme", "light") == "light" else "#334155"
    dark = "#a57a4e" if st.session_state.get("theme", "light") == "light" else "#0f172a"
    fig = go.Figure(
        data=go.Heatmap(
            z=board,
            colorscale=[[0.0, light], [1.0, dark]],
            showscale=False,
            xgap=0,
            ygap=0,
        )
    )
    qx, qy = [], []
    for r, c in enumerate(cols_per_row):
        if 0 <= int(c) < N:
            qx.append(int(c))
            qy.append(r)
    # Detect conflicts for visual warning.
    conflict_rows = set()
    for r1 in range(N):
        for r2 in range(r1 + 1, N):
            c1, c2 = int(cols_per_row[r1]), int(cols_per_row[r2])
            if c1 == c2 or abs(c1 - c2) == abs(r1 - r2):
                conflict_rows.update({r1, r2})
    colours = ["#dc2626" if r in conflict_rows else p["palette"][0] for r in qy]
    fig.add_trace(
        go.Scatter(
            x=qx,
            y=qy,
            mode="markers+text",
            marker={"size": 26, "color": colours, "line": {"color": "#0f172a", "width": 1.5}},
            text=["♛"] * len(qx),
            textfont={"size": 20, "color": "#f8fafc"},
            textposition="middle center",
            showlegend=False,
        )
    )
    fig.update_layout(
        **plotly_layout(
            title={
                "text": f"N-Queens (N={N}) · conflicting queens: {len(conflict_rows)}",
                "x": 0.5,
            },
            height=540,
            xaxis={"showgrid": False, "tickmode": "linear", "dtick": 1},
            yaxis={
                "showgrid": False,
                "tickmode": "linear",
                "dtick": 1,
                "scaleanchor": "x",
                "scaleratio": 1,
                "autorange": "reversed",
            },
            showlegend=False,
        )
    )
    _render(fig, key="soln_nqueens")


# ---------------------------------------------------------------------------
# Classic CO
# ---------------------------------------------------------------------------


def render_knapsack(problem, result, cfg) -> None:
    x = (_as_numpy(result.best_sol).astype(float) > 0.5).astype(int)
    values = _as_numpy(problem.values)
    weights = _as_numpy(problem.weights)
    capacity = float(problem.capacity)

    total_value = float((values * x).sum())
    total_weight = float((weights * x).sum())
    p = palette()

    order = np.argsort(-values)
    sel_colour = [p["palette"][0] if x[i] else hex_to_rgba(p["muted"], 0.25) for i in order]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"item {i}" for i in order],
            y=values[order],
            marker={"color": sel_colour, "line": {"color": "#0f172a", "width": 0.6}},
            hovertemplate="item %{x}<br>value %{y:.3f}<br>weight %{customdata:.3f}<extra></extra>",
            customdata=weights[order],
            name="value",
        )
    )
    fig.update_layout(
        **plotly_layout(
            title={
                "text": (
                    f"0/1 Knapsack · value={total_value:.3g} · "
                    f"weight={total_weight:.3g} / C={capacity:.3g}"
                ),
                "x": 0.5,
            },
            height=420,
            showlegend=False,
            xaxis={"showticklabels": False, "title": "items (sorted by value, filled = chosen)"},
            yaxis_title="value",
            bargap=0.12,
        )
    )
    _render(fig, key="soln_knapsack_values")

    # Secondary plot: weight budget with capacity line.
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=["chosen weight", "capacity"],
            y=[total_weight, capacity],
            marker={"color": [p["palette"][0], hex_to_rgba(p["muted"], 0.45)]},
        )
    )
    fig2.update_layout(
        **plotly_layout(
            title={"text": "Weight vs capacity", "x": 0.5},
            height=220,
            showlegend=False,
            yaxis_title="weight",
        )
    )
    _render(fig2, key="soln_knapsack_weights")


def render_number_partitioning(problem, result, cfg) -> None:
    sol = _as_numpy(result.best_sol).reshape(-1)
    # Spin variables ∈ {-1, +1}.
    s = np.where(sol >= 0, 1, -1)
    values = _as_numpy(problem.values)
    plus = values[s > 0]
    minus = values[s < 0]
    sum_plus = float(plus.sum())
    sum_minus = float(minus.sum())
    diff = abs(sum_plus - sum_minus)
    p = palette()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["subset A (+)", "subset B (-)"],
            y=[sum_plus, sum_minus],
            marker={"color": [p["palette"][0], p["palette"][1]]},
            text=[f"Σ={sum_plus:.1f}", f"Σ={sum_minus:.1f}"],
            textposition="auto",
        )
    )
    fig.add_hline(
        y=(sum_plus + sum_minus) / 2,
        line={"color": hex_to_rgba(p["muted"], 0.55), "dash": "dot"},
        annotation_text="ideal",
    )
    fig.update_layout(
        **plotly_layout(
            title={"text": f"Number partitioning · |Σ_A − Σ_B| = {diff:.3g}", "x": 0.5},
            height=340,
            showlegend=False,
            yaxis_title="sum",
        )
    )
    _render(fig, key="soln_number_partition")


def render_maxsat3(problem, result, cfg) -> None:
    # The satisfied-clause count is what QQA returns as "value"; the total
    # number of clauses lives in problem.literals / problem.num_clauses.
    score = result.score or {}
    value = float(score.get("value", 0.0))
    extra = score.get("extra", {}) or {}
    M = int(extra.get("clauses", 0) or getattr(problem, "num_clauses", 0) or 1)
    satisfied = int(round(value)) if value >= 0 else max(0, M + int(round(value)))
    unsat = max(0, M - satisfied)
    p = palette()

    fig = go.Figure(
        data=go.Pie(
            labels=["satisfied", "unsatisfied"],
            values=[satisfied, unsat],
            hole=0.6,
            marker={"colors": [p["palette"][0], hex_to_rgba(p["muted"], 0.35)]},
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        **plotly_layout(
            title={"text": f"MaxSAT · {satisfied}/{M} clauses satisfied", "x": 0.5},
            height=360,
            showlegend=False,
            annotations=[
                {
                    "text": f"{100 * satisfied / M:.1f}%",
                    "showarrow": False,
                    "font": {"size": 22, "family": "Source Serif 4"},
                }
            ],
        )
    )
    _render(fig, key="soln_maxsat3")


# ---------------------------------------------------------------------------
# Physics / spin problems
# ---------------------------------------------------------------------------


def _spin_strip(
    sol: np.ndarray,
    *,
    title: str,
    key: str,
    aspect_height: int = 80,
) -> None:
    z = sol.reshape(1, -1)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            colorscale=[[0.0, "#1e3a8a"], [0.5, "#e5e7eb"], [1.0, "#b45309"]],
            zmin=-1,
            zmax=1,
            showscale=False,
            xgap=0,
            ygap=0,
        )
    )
    fig.update_layout(
        **plotly_layout(
            title={"text": title, "x": 0.5},
            height=aspect_height,
            xaxis={"showgrid": False, "zeroline": False, "tickmode": "linear", "dtick": 5},
            yaxis={"showticklabels": False, "showgrid": False, "zeroline": False},
            margin={"l": 40, "r": 20, "t": 48, "b": 30},
        )
    )
    _render(fig, key=key)


def render_ising1d(problem, result, cfg) -> None:
    s = _as_numpy(result.best_sol).astype(float).reshape(-1)
    _spin_strip(s, title=f"1D Ising · N={len(s)} · magnetization={s.mean():+.3f}", key="soln_ising")


def render_sk(problem, result, cfg) -> None:
    s = _as_numpy(result.best_sol).astype(float).reshape(-1)
    N = len(s)
    # Show the spin configuration as a strip + a small order-parameter bar.
    _spin_strip(s, title=f"SK spin glass · N={N}", key="soln_sk_strip")

    p = palette()
    mag = float(s.mean())
    fig = go.Figure(
        data=go.Bar(
            x=["magnetization", "|m|"],
            y=[mag, abs(mag)],
            marker={"color": [p["palette"][0], p["palette"][1]]},
        )
    )
    fig.update_layout(
        **plotly_layout(
            title={"text": "SK order parameters", "x": 0.5},
            height=260,
            showlegend=False,
            yaxis_title="value",
        )
    )
    _render(fig, key="soln_sk_mag")


def render_ea(problem, result, cfg) -> None:
    s = _as_numpy(result.best_sol).astype(float).reshape(-1)
    L = int(round(len(s) ** (1.0 / getattr(problem, "dim", 3))))
    if L ** getattr(problem, "dim", 3) == len(s) and getattr(problem, "dim", 3) == 2:
        grid = s.reshape(L, L)
        fig = go.Figure(
            data=go.Heatmap(
                z=grid,
                colorscale=[[0.0, "#1e3a8a"], [0.5, "#e5e7eb"], [1.0, "#b45309"]],
                zmin=-1,
                zmax=1,
                showscale=False,
            )
        )
        fig.update_layout(
            **plotly_layout(
                title={"text": f"EA spin glass · 2D L×L = {L}×{L}", "x": 0.5},
                height=460,
                xaxis={"visible": False},
                yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
            )
        )
        _render(fig, key="soln_ea_2d")
    else:
        _spin_strip(
            s,
            title=(f"EA spin glass · dim={getattr(problem, 'dim', '?')} · N={len(s)} (flattened)"),
            key="soln_ea_flat",
            aspect_height=100,
        )


def render_perceptron(problem, result, cfg) -> None:
    s = _as_numpy(result.best_sol).astype(float).reshape(-1)
    s = np.where(s >= 0, 1, -1)
    xi = _as_numpy(problem.xi_signed)  # (M, N)
    scores = xi @ s  # (M,)
    correct = int((scores > 0).sum())
    M = scores.shape[0]
    p = palette()

    fig = go.Figure(
        data=go.Bar(
            y=np.arange(M),
            x=scores,
            orientation="h",
            marker={
                "color": [p["palette"][0] if v > 0 else p["palette"][1] for v in scores],
                "line": {"color": "#0f172a", "width": 0.4},
            },
        )
    )
    fig.add_vline(x=0, line={"color": hex_to_rgba(p["muted"], 0.55), "dash": "dot"})
    fig.update_layout(
        **plotly_layout(
            title={
                "text": f"Binary perceptron · {correct}/{M} patterns classified correctly",
                "x": 0.5,
            },
            height=max(260, 16 * M),
            xaxis_title="s · ξ̂ (signed)",
            yaxis_title="pattern index",
            showlegend=False,
        )
    )
    _render(fig, key="soln_perceptron")


def render_hopfield(problem, result, cfg) -> None:
    s = _as_numpy(result.best_sol).astype(float).reshape(-1)
    s = np.where(s >= 0, 1, -1)
    patterns = _as_numpy(problem.patterns)  # (P, N) with ±1
    if patterns.ndim != 2:
        patterns = patterns.reshape(-1, s.shape[0])
    overlaps = (patterns @ s) / s.shape[0]  # in [-1, 1]
    p = palette()
    fig = go.Figure(
        data=go.Bar(
            x=[f"ξ^({k})" for k in range(patterns.shape[0])],
            y=overlaps,
            marker={
                "color": [p["palette"][0] if abs(v) > 0.6 else p["palette"][1] for v in overlaps],
                "line": {"color": "#0f172a", "width": 0.5},
            },
            text=[f"{v:+.2f}" for v in overlaps],
            textposition="auto",
        )
    )
    fig.update_layout(
        **plotly_layout(
            title={
                "text": f"Hopfield memory · max |overlap| = {np.abs(overlaps).max():.3f}",
                "x": 0.5,
            },
            height=320,
            yaxis_title="overlap",
            yaxis={"range": [-1.05, 1.05]},
            showlegend=False,
        )
    )
    _render(fig, key="soln_hopfield")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_RENDERERS: dict[str, Any] = {
    "mis": render_mis,
    "maxclique": render_max_clique,
    "vertex_cover": render_vertex_cover,
    "maxcut": render_max_cut,
    "graph_bisection": render_graph_bisection,
    "coloring": render_coloring,
    "tsp": render_tsp,
    "qap": render_qap,
    "nqueens": render_nqueens,
    "knapsack": render_knapsack,
    "number_partition": render_number_partitioning,
    "maxsat3": render_maxsat3,
    "ising1d": render_ising1d,
    "sk": render_sk,
    "ea": render_ea,
    "perceptron": render_perceptron,
    "hopfield": render_hopfield,
}


def _fallback_render(problem, result, cfg) -> None:
    """Generic fallback: show the solution as a 1-D strip."""
    try:
        s = _as_numpy(result.best_sol).astype(float).reshape(-1)
    except Exception:
        st.info("No generic solution visualisation available for this problem.")
        return
    fig = go.Figure(data=go.Heatmap(z=s.reshape(1, -1), colorscale="Viridis", showscale=False))
    fig.update_layout(
        **plotly_layout(
            title={"text": "Solution (raw)", "x": 0.5},
            height=120,
            xaxis={"showgrid": False},
            yaxis={"visible": False},
        )
    )
    _render(fig, key="soln_fallback")


def render_solution_view(problem, result, cfg) -> None:
    """Render a professional, problem-aware visualisation of ``result.best_sol``.

    Falls back to a generic 1-D heatmap for any ``cfg["kind"]`` we don't
    know about (e.g. custom problems). Never raises: errors are surfaced
    via :pyfunc:`st.warning` so the surrounding page stays functional.
    """
    kind = cfg.get("kind", "")
    renderer = _RENDERERS.get(kind, _fallback_render)
    try:
        renderer(problem, result, cfg)
    except Exception as exc:  # pragma: no cover - UI-surface, never abort
        st.warning(f"Solution visualisation failed: {exc}")
        with contextlib.suppress(Exception):
            _fallback_render(problem, result, cfg)
