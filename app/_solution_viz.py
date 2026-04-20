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
from _common import as_numpy as _as_numpy
from _common import hex_to_rgba, palette, plotly_layout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    sol = _as_numpy(result.best_sol)  # (N, N) one-hot (post Hungarian-snap)
    tour = sol.argmax(axis=-1) if sol.ndim == 2 else sol.astype(int)
    coords = _as_numpy(problem.coords)  # (N, 2)
    p = palette()

    ordered = np.concatenate([tour, tour[:1]])
    xs = coords[ordered, 0]
    ys = coords[ordered, 1]
    dist = float(result.score.get("value", 0.0))
    extra = (result.score or {}).get("extra", {}) or {}
    snapped = bool(extra.get("snapped", False))
    raw_feasible = bool(extra.get("raw_feasible", True))

    fig = go.Figure()
    # Faint "potential edges" backdrop — every city pair, opacity ∝ 1/distance.
    # Helps the reader judge whether the tour is locally near-optimal without
    # hiding the chosen tour itself.
    if len(coords) <= 16:
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                fig.add_trace(
                    go.Scatter(
                        x=[coords[i, 0], coords[j, 0]],
                        y=[coords[i, 1], coords[j, 1]],
                        mode="lines",
                        line={"color": hex_to_rgba(p["muted"], 0.10), "width": 0.6},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
    # Tour with directional arrows.
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line={"color": p["palette"][0], "width": 2.4},
            marker={"size": 11, "color": p["palette"][1], "line": {"color": "#0f172a", "width": 1}},
            name="tour",
            hovertemplate="position %{pointNumber}<br>(%{x:.3f}, %{y:.3f})<extra></extra>",
            showlegend=False,
        )
    )
    # Direction arrows at the midpoint of each segment.
    for k in range(len(tour)):
        a = coords[tour[k]]
        b = coords[tour[(k + 1) % len(tour)]]
        mid_x = (a[0] + b[0]) / 2
        mid_y = (a[1] + b[1]) / 2
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            ax=a[0],
            ay=a[1],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.0,
            arrowwidth=1.6,
            arrowcolor=hex_to_rgba(p["palette"][0], 0.85),
            standoff=4,
            startstandoff=4,
            opacity=0.9,
        )
    # Start marker.
    fig.add_trace(
        go.Scatter(
            x=[coords[tour[0], 0]],
            y=[coords[tour[0], 1]],
            mode="markers",
            marker={
                "size": 20,
                "color": p["palette"][2],
                "line": {"color": "#0f172a", "width": 1.4},
                "symbol": "star",
            },
            name="start",
            hovertemplate="START · city %{text}<extra></extra>",
            text=[str(int(tour[0]))],
        )
    )
    for step, city in enumerate(tour):
        fig.add_annotation(
            x=coords[city, 0],
            y=coords[city, 1],
            text=str(step),
            font={"size": 10, "color": p["text"]},
            showarrow=False,
            yshift=14,
        )

    badge = (
        " <span style='color:#16a34a'>● raw permutation</span>"
        if raw_feasible
        else " <span style='color:#b45309'>● Hungarian-snapped</span>"
    )
    extra_note = " (snapped to nearest permutation)" if snapped else ""
    fig.update_layout(
        **plotly_layout(
            title={
                "text": f"TSP tour · length ≈ {dist:.4g}{extra_note}{badge}",
                "x": 0.5,
            },
            height=520,
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


# Colour scheme shared across spin renderers: warm = up, cool = down.
_SPIN_UP_COLOR = "#dc2626"  # crimson
_SPIN_DOWN_COLOR = "#1d4ed8"  # deep blue


def _spin_arrow_traces(
    xs: np.ndarray,
    ys: np.ndarray,
    spins: np.ndarray,
    *,
    size: int = 22,
    border: str = "#0f172a",
):
    """Return up/down marker traces (one per spin state).

    Spins are drawn as triangle markers (▲ up, ▼ down) with two complementary
    colours and a thin outline so they read on both light and dark themes.
    """
    s = np.asarray(spins).astype(float).reshape(-1)
    up = s > 0
    down = ~up
    traces = []
    if up.any():
        traces.append(
            go.Scatter(
                x=xs[up],
                y=ys[up],
                mode="markers",
                marker={
                    "symbol": "triangle-up",
                    "size": size,
                    "color": _SPIN_UP_COLOR,
                    "line": {"color": border, "width": 1.4},
                },
                name="↑ +1",
                hovertemplate="spin=%{text}<extra>↑ +1</extra>",
                text=[f"node {i}: ↑" for i in np.flatnonzero(up)],
                showlegend=True,
            )
        )
    if down.any():
        traces.append(
            go.Scatter(
                x=xs[down],
                y=ys[down],
                mode="markers",
                marker={
                    "symbol": "triangle-down",
                    "size": size,
                    "color": _SPIN_DOWN_COLOR,
                    "line": {"color": border, "width": 1.4},
                },
                name="↓ −1",
                hovertemplate="spin=%{text}<extra>↓ −1</extra>",
                text=[f"node {i}: ↓" for i in np.flatnonzero(down)],
                showlegend=True,
            )
        )
    return traces


def _border_for_theme() -> str:
    return "#0f172a" if st.session_state.get("theme", "light") == "light" else "#f8fafc"


def _ring_layout(N: int) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.cos(angles), np.sin(angles)


def render_ising1d(problem, result, cfg) -> None:
    s = _as_numpy(result.best_sol).astype(float).reshape(-1)
    s = np.where(s >= 0, 1, -1)
    N = len(s)
    p = palette()

    # Ring layout: 1D periodic chain → circle. Bonds are nearest neighbours.
    xs, ys = _ring_layout(N)

    # Bond ribbon: faint background ring connecting consecutive spins.
    ring_x = np.append(xs, xs[0])
    ring_y = np.append(ys, ys[0])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ring_x,
            y=ring_y,
            mode="lines",
            line={"color": hex_to_rgba(p["muted"], 0.45), "width": 1.2},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for tr in _spin_arrow_traces(xs, ys, s, size=24, border=_border_for_theme()):
        fig.add_trace(tr)

    # Tag a few spins with their index so users can read the configuration.
    step = max(1, N // 16)
    for i in range(0, N, step):
        fig.add_annotation(
            x=xs[i] * 1.15,
            y=ys[i] * 1.15,
            text=str(i),
            showarrow=False,
            font={"size": 9, "color": p["muted"]},
        )

    n_up = int((s > 0).sum())
    n_down = N - n_up
    mag = float(s.mean())
    fig.update_layout(
        **plotly_layout(
            title={
                "text": (
                    f"1D Ising — N={N}, m={mag:+.3f}  "
                    f"<span style='color:{_SPIN_UP_COLOR}'>↑ {n_up}</span>"
                    f" · <span style='color:{_SPIN_DOWN_COLOR}'>↓ {n_down}</span>"
                ),
                "x": 0.5,
            },
            height=460,
            showlegend=True,
            legend={"x": 0.01, "y": 0.99},
            **_axes_off(),
        )
    )
    _render(fig, key="soln_ising_ring")


def render_sk(problem, result, cfg) -> None:
    s = _as_numpy(result.best_sol).astype(float).reshape(-1)
    s = np.where(s >= 0, 1, -1)
    N = len(s)
    p = palette()

    # SK is fully connected — drawing every bond is unreadable. Show the
    # spin orientation on a circle, and add an order-parameter panel below.
    xs, ys = _ring_layout(N)

    fig = go.Figure()
    # Faint backdrop circle (no bonds drawn — the problem is dense).
    th = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=np.cos(th),
            y=np.sin(th),
            mode="lines",
            line={"color": hex_to_rgba(p["muted"], 0.25), "width": 1.0, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for tr in _spin_arrow_traces(xs, ys, s, size=20, border=_border_for_theme()):
        fig.add_trace(tr)

    mag = float(s.mean())
    fig.update_layout(
        **plotly_layout(
            title={
                "text": (
                    f"SK spin glass — N={N}, m={mag:+.3f}  "
                    f"<span style='color:{_SPIN_UP_COLOR}'>↑ {int((s > 0).sum())}</span>"
                    f" · <span style='color:{_SPIN_DOWN_COLOR}'>↓ {int((s < 0).sum())}</span>"
                ),
                "x": 0.5,
            },
            height=440,
            showlegend=True,
            legend={"x": 0.01, "y": 0.99},
            **_axes_off(),
        )
    )
    _render(fig, key="soln_sk_ring")

    # Energy contribution histogram if J is available.
    J = getattr(problem, "J", None)
    if J is not None:
        J_np = _as_numpy(J)
        local_field = J_np @ s
        contrib = -s * local_field  # positive = unhappy
        fig2 = go.Figure(
            data=go.Bar(
                x=np.arange(N),
                y=contrib,
                marker={
                    "color": [_SPIN_UP_COLOR if c > 0 else _SPIN_DOWN_COLOR for c in contrib],
                    "line": {"color": "#0f172a", "width": 0.4},
                },
                hovertemplate="spin %{x}<br>local energy %{y:+.3f}<extra></extra>",
            )
        )
        fig2.add_hline(
            y=0,
            line={"color": hex_to_rgba(p["muted"], 0.55), "dash": "dot"},
        )
        fig2.update_layout(
            **plotly_layout(
                title={"text": "Local energy per spin (positive = frustrated)", "x": 0.5},
                height=240,
                showlegend=False,
                xaxis_title="spin index",
                yaxis_title="−s_i Σ_j J_ij s_j",
            )
        )
        _render(fig2, key="soln_sk_local_energy")


def _ea_2d_arrow_figure(s: np.ndarray, L: int, *, title: str) -> go.Figure:
    """Render an L×L EA slice as ↑↓ markers on a faint lattice grid."""
    p = palette()
    fig = go.Figure()

    # Lattice bonds (vertical + horizontal segments). Plotly likes None-
    # delimited polylines so the whole grid is one trace.
    seg_x: list[float | None] = []
    seg_y: list[float | None] = []
    for r in range(L):
        for c in range(L):
            if c + 1 < L:
                seg_x.extend([c, c + 1, None])
                seg_y.extend([r, r, None])
            if r + 1 < L:
                seg_x.extend([c, c, None])
                seg_y.extend([r, r + 1, None])
    fig.add_trace(
        go.Scatter(
            x=seg_x,
            y=seg_y,
            mode="lines",
            line={"color": hex_to_rgba(p["muted"], 0.30), "width": 1.0},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Position arrows at every lattice site.
    xs = np.tile(np.arange(L), L).astype(float)
    ys = np.repeat(np.arange(L), L).astype(float)
    for tr in _spin_arrow_traces(xs, ys, s, size=18, border=_border_for_theme()):
        fig.add_trace(tr)

    fig.update_layout(
        **plotly_layout(
            title={"text": title, "x": 0.5},
            height=520,
            showlegend=True,
            legend={"x": 0.01, "y": 0.99},
            xaxis={
                "visible": False,
                "showgrid": False,
                "zeroline": False,
                "range": [-0.5, L - 0.5],
            },
            yaxis={
                "visible": False,
                "showgrid": False,
                "zeroline": False,
                "range": [-0.5, L - 0.5],
                "scaleanchor": "x",
                "scaleratio": 1,
                "autorange": "reversed",
            },
        )
    )
    return fig


def _ea_3d_cone_figure(
    s: np.ndarray,
    L: int,
    *,
    J: np.ndarray | None = None,
    title: str,
    show_bonds: bool = True,
    show_frustration: bool = False,
) -> go.Figure:
    """Render a 3D Edwards–Anderson spin field as a rotatable cone scene.

    Each lattice site becomes a Plotly :class:`Cone` arrow pointing along
    ``+z`` (spin up) or ``-z`` (spin down), coloured by sign. Bonds
    between nearest neighbours are drawn as a single low-opacity
    ``Scatter3d`` polyline so the cubic structure is legible without
    cluttering the cone field.

    Args:
        s: flat ``(L**3,)`` array of ±1 spins.
        L: lattice side.
        J: optional ``(L**3, L**3)`` coupling matrix. If supplied and
            ``show_frustration=True``, cones are coloured by their local
            energy contribution ``-s_i · Σ_j J_ij s_j`` (warm = frustrated)
            instead of by spin sign.
        title: chart title (HTML allowed).
    """
    p = palette()
    s = np.asarray(s).astype(float)
    cube = s.reshape(L, L, L)
    xs, ys, zs = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing="ij")
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    zs = zs.reshape(-1)
    spins = cube.reshape(-1)
    up_mask = spins > 0
    down_mask = ~up_mask

    fig = go.Figure()

    if show_bonds:
        # Build all unique nearest-neighbour bonds in a *single* trace by
        # using ``None``-delimited polylines.  Drawing one Scatter3d per
        # bond would create ~3·L^3 traces and obliterate the FPS.
        seg_x: list[float | None] = []
        seg_y: list[float | None] = []
        seg_z: list[float | None] = []
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    if i + 1 < L:
                        seg_x.extend([i, i + 1, None])
                        seg_y.extend([j, j, None])
                        seg_z.extend([k, k, None])
                    if j + 1 < L:
                        seg_x.extend([i, i, None])
                        seg_y.extend([j, j + 1, None])
                        seg_z.extend([k, k, None])
                    if k + 1 < L:
                        seg_x.extend([i, i, None])
                        seg_y.extend([j, j, None])
                        seg_z.extend([k, k + 1, None])
        fig.add_trace(
            go.Scatter3d(
                x=seg_x,
                y=seg_y,
                z=seg_z,
                mode="lines",
                line={"color": hex_to_rgba(p["muted"], 0.18), "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Spin cones — two colour-bound traces (one up, one down) so the
    # legend explains the encoding without needing a continuous bar.
    cone_size = 0.9  # length of each arrow in lattice units
    cone_kwargs = {
        "sizemode": "absolute",
        "sizeref": cone_size,
        "anchor": "tail",
        "showscale": False,
    }

    if up_mask.any():
        fig.add_trace(
            go.Cone(
                x=xs[up_mask],
                y=ys[up_mask],
                z=zs[up_mask] - cone_size / 2,
                u=np.zeros(up_mask.sum()),
                v=np.zeros(up_mask.sum()),
                w=np.full(up_mask.sum(), 1.0),
                colorscale=[[0, _SPIN_UP_COLOR], [1, _SPIN_UP_COLOR]],
                cmin=0,
                cmax=1,
                hovertemplate="(%{x:.0f}, %{y:.0f}, %{z:.0f}) · ↑ +1<extra></extra>",
                name="↑ +1",
                showlegend=True,
                **cone_kwargs,
            )
        )
    if down_mask.any():
        fig.add_trace(
            go.Cone(
                x=xs[down_mask],
                y=ys[down_mask],
                z=zs[down_mask] + cone_size / 2,
                u=np.zeros(down_mask.sum()),
                v=np.zeros(down_mask.sum()),
                w=np.full(down_mask.sum(), -1.0),
                colorscale=[[0, _SPIN_DOWN_COLOR], [1, _SPIN_DOWN_COLOR]],
                cmin=0,
                cmax=1,
                hovertemplate="(%{x:.0f}, %{y:.0f}, %{z:.0f}) · ↓ −1<extra></extra>",
                name="↓ −1",
                showlegend=True,
                **cone_kwargs,
            )
        )

    # Optional frustration overlay: invisible scatter markers coloured
    # by local energy. Cheaper than recolouring the cones (Plotly Cone
    # does not let us mix colorscales per-cone) and still visible in the
    # 3-D scene.
    if show_frustration and J is not None:
        local_energy = -s * (J @ s)
        emax = float(np.abs(local_energy).max() + 1e-9)
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker={
                    "size": 5,
                    "color": local_energy,
                    "colorscale": "RdBu",
                    "cmin": -emax,
                    "cmax": emax,
                    "opacity": 0.9,
                    "colorbar": {
                        "title": "−s·Σ Js  (warm = frustrated)",
                        "thickness": 12,
                        "len": 0.6,
                    },
                },
                hovertemplate=(
                    "site (%{x:.0f}, %{y:.0f}, %{z:.0f})<br>"
                    "local energy %{marker.color:+.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title={"text": title, "x": 0.5},
        height=600,
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene={
            "xaxis": {
                "title": "x",
                "showbackground": False,
                "gridcolor": hex_to_rgba(p["muted"], 0.18),
                "zerolinecolor": hex_to_rgba(p["muted"], 0.30),
                "showspikes": False,
                "range": [-0.5, L - 0.5],
            },
            "yaxis": {
                "title": "y",
                "showbackground": False,
                "gridcolor": hex_to_rgba(p["muted"], 0.18),
                "zerolinecolor": hex_to_rgba(p["muted"], 0.30),
                "showspikes": False,
                "range": [-0.5, L - 0.5],
            },
            "zaxis": {
                "title": "z",
                "showbackground": False,
                "gridcolor": hex_to_rgba(p["muted"], 0.18),
                "zerolinecolor": hex_to_rgba(p["muted"], 0.30),
                "showspikes": False,
                "range": [-0.5, L - 0.5],
            },
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.7, "y": 1.7, "z": 1.0}},
            "bgcolor": "rgba(0,0,0,0)",
        },
        legend={"x": 0.01, "y": 0.99},
    )
    return fig


def render_ea(problem, result, cfg) -> None:
    s = _as_numpy(result.best_sol).astype(float).reshape(-1)
    s = np.where(s >= 0, 1, -1)
    dim = int(getattr(problem, "dim", 3))
    L = int(getattr(problem, "L", round(len(s) ** (1.0 / dim))))
    N = len(s)
    mag = float(s.mean())
    n_up = int((s > 0).sum())
    title_chips = (
        f"<span style='color:{_SPIN_UP_COLOR}'>↑ {n_up}</span>"
        f" · <span style='color:{_SPIN_DOWN_COLOR}'>↓ {N - n_up}</span>"
    )

    if dim == 2 and L * L == N:
        fig = _ea_2d_arrow_figure(
            s, L, title=f"EA spin glass · 2D {L}×{L} · m={mag:+.3f}  {title_chips}"
        )
        _render(fig, key="soln_ea_2d_arrows")
        return

    if dim == 3 and L**3 == N:
        # 3-D cone field — the headline view. Big lattices (L > 16) are
        # downsampled to keep the trace count under a few thousand cones
        # which is comfortable for the WebGL renderer.
        if L > 16:
            stride = int(np.ceil(L / 16))
            cube = s.reshape(L, L, L)[::stride, ::stride, ::stride]
            L_show = cube.shape[0]
            s_show = cube.reshape(-1)
            sub_title = f"(downsampled stride={stride})"
        else:
            L_show = L
            s_show = s
            sub_title = ""

        with st.container():
            colA, colB = st.columns([3, 1])
            with colB:
                show_frustration = st.toggle(
                    "Frustration overlay",
                    value=False,
                    key="ea3d_frustration",
                    help="Colour each site by its local energy contribution "
                    "(positive = frustrated). Requires J access.",
                )
                show_bonds = st.toggle(
                    "Show bonds",
                    value=True,
                    key="ea3d_bonds",
                    help="Faint nearest-neighbour bonds — turn off for a cleaner cone field.",
                )

            J_dense = None
            if show_frustration:
                try:
                    J_attr = getattr(problem, "J", None)
                    if J_attr is not None:
                        J_dense = (
                            J_attr.detach().cpu().numpy()
                            if hasattr(J_attr, "detach")
                            else np.asarray(J_attr)
                        )
                        if L_show != L:
                            # Frustration overlay is meaningless on the
                            # downsampled grid, fall back to spin colouring.
                            J_dense = None
                            show_frustration = False
                except Exception:
                    show_frustration = False

            fig = _ea_3d_cone_figure(
                s_show,
                L_show,
                J=J_dense,
                title=(
                    f"EA spin glass · 3D {L}<sup>3</sup> · m={mag:+.3f}  {title_chips}  {sub_title}"
                ),
                show_bonds=show_bonds,
                show_frustration=show_frustration,
            )
            with colA:
                _render(fig, key="soln_ea_3d_cones")

        with st.expander("Z-slice cross-sections", expanded=False):
            cube = s.reshape(L, L, L)
            max_slices = 6
            z_indices = list(range(L))
            if max_slices < L:
                z_indices = list(np.linspace(0, L - 1, max_slices, dtype=int))
            cols = st.columns(min(3, len(z_indices)))
            for i, z in enumerate(z_indices):
                slice_2d = cube[:, :, z].reshape(-1)
                with cols[i % len(cols)]:
                    fig = _ea_2d_arrow_figure(
                        slice_2d,
                        L,
                        title=f"z = {z}  ·  m_slice = {slice_2d.mean():+.3f}",
                    )
                    fig.update_layout(height=320, margin={"l": 20, "r": 20, "t": 40, "b": 20})
                    _render(fig, key=f"soln_ea_slice_z{z}")
        return

    # Fallback: arrange on a single circle.
    xs, ys = _ring_layout(N)
    fig = go.Figure()
    for tr in _spin_arrow_traces(xs, ys, s, size=14, border=_border_for_theme()):
        fig.add_trace(tr)
    fig.update_layout(
        **plotly_layout(
            title={"text": f"EA spin glass · dim={dim} · N={N}", "x": 0.5},
            height=460,
            showlegend=True,
            **_axes_off(),
        )
    )
    _render(fig, key="soln_ea_ring")


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
    # Dense p-spin glass has no useful spatial layout — reuse the SK ring
    # + local-energy view (J is None for p > 2 so only the ring renders,
    # which is exactly what we want).
    "pspin": render_sk,
    "ea": render_ea,
    # RFIM lives on the same hyper-cubic lattice as EA, so the EA
    # 2D-slice renderer applies as-is.
    "rfim": render_ea,
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
