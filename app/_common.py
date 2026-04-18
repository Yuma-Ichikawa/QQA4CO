"""Shared helpers for the QQA Streamlit app.

Keeps problem construction, previews, and theming in one place so the Home /
Solve / Visualize / Compare pages stay declarative.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import streamlit as st

import qqa

try:
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:  # pragma: no cover - surface a friendlier error
    st.error(
        "The `plotly` package is required for the dashboard. "
        "Add it to `requirements.txt` (or run `pip install plotly`)."
    )
    st.stop()
    raise SystemExit from exc

# Default snippet shown in the Custom-problem editor. It implements a simple
# Sherrington–Kirkpatrick-style spin glass that the user can adapt freely.
DEFAULT_CUSTOM_SNIPPET = '''import torch

# Runs once when the problem is built. Put any fixed data (couplings,
# patterns, matrices, ...) here and keep it out of the hot loop below.
N = 32
g = torch.Generator().manual_seed(0)
J = torch.randn(N, N, generator=g) / (N ** 0.5)
J = (J + J.T) / 2
J.fill_diagonal_(0.0)


def loss_fn(s):
    """Batched energy of a spin-glass. ``s`` has shape (B, N); return (B,)."""
    return -0.5 * torch.einsum("bi,ij,bj->b", s, J, s)
'''


def apply_theme() -> None:
    """Inject a professional dark/glass theme shared by every page."""
    st.markdown(
        """
        <style>
        :root {
            --qqa-bg-1: #05060a;
            --qqa-bg-2: #0b1120;
            --qqa-accent: #38bdf8;
            --qqa-accent2: #a855f7;
            --qqa-text: #e2e8f0;
            --qqa-muted: #94a3b8;
        }
        .stApp {
            background:
                radial-gradient(900px 500px at 10% -10%, rgba(56,189,248,0.18), transparent 60%),
                radial-gradient(900px 500px at 120% 20%, rgba(168,85,247,0.18), transparent 60%),
                linear-gradient(180deg, var(--qqa-bg-1) 0%, var(--qqa-bg-2) 100%);
            color: var(--qqa-text);
        }
        section[data-testid="stSidebar"] {
            background: rgba(11, 18, 32, 0.72);
            backdrop-filter: blur(14px);
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }
        h1, h2, h3, h4 { color: #f8fafc; letter-spacing: -0.01em; }
        h1 {
            background: linear-gradient(120deg, #e0f2fe, #a855f7, #38bdf8);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stMetric { background: rgba(15, 23, 42, 0.55); padding: 0.6rem 0.9rem;
                    border-radius: 14px; border: 1px solid rgba(148,163,184,0.18);
                    backdrop-filter: blur(8px); }
        .stMetric > label { color: var(--qqa-muted) !important; font-weight: 500; }
        .stMetric [data-testid="stMetricValue"] { color: #f8fafc !important; }
        .stButton > button {
            background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
            color: white;
            border: 0;
            border-radius: 999px;
            padding: 0.4rem 1.1rem;
            font-weight: 600;
            box-shadow: 0 6px 22px rgba(56,189,248,0.25);
            transition: transform 120ms ease, box-shadow 120ms ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 28px rgba(168,85,247,0.35);
        }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            background: rgba(15, 23, 42, 0.55); border-radius: 12px;
        }
        code, pre { background: rgba(15, 23, 42, 0.75); color: #bae6fd; }
        .qqa-card {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(148,163,184,0.18);
            padding: 1rem 1.2rem;
            border-radius: 16px;
            backdrop-filter: blur(10px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------


def build_problem(cfg: dict) -> Any:
    """Instantiate a problem from a dashboard config dict.

    Supports all bundled problem families as well as ``kind == "custom"``,
    which compiles a user-provided Python snippet defining ``loss_fn``.
    """
    qqa.fix_seed(cfg["seed"])
    extra = cfg.get("extra", {})
    kind = cfg["kind"]
    device = cfg["device"]

    if kind == "custom":
        source = extra.get("source", DEFAULT_CUSTOM_SNIPPET)
        num_vars = int(extra.get("num_vars", 32))
        variable_kind = extra.get("variable_kind", "spin")
        num_category = extra.get("num_category")
        return qqa.user_problem_from_source(
            source=source,
            num_vars=num_vars,
            variable_kind=variable_kind,
            num_category=num_category,
            name=extra.get("name", "custom"),
            device=device,
        )

    size = int(cfg["size"])
    if kind in {"mis", "maxcut", "maxclique", "coloring"}:
        d = extra.get("graph_d", 3)
        if (size * d) % 2 != 0:
            d = max(2, d - 1) if d > 2 else d + 1
        g = nx.random_regular_graph(d=d, n=size, seed=cfg["seed"])
        if kind == "mis":
            return qqa.MaximumIndependentSet(g, device=device)
        if kind == "maxcut":
            return qqa.MaxCut(g, device=device)
        if kind == "maxclique":
            return qqa.MaxClique(g, device=device)
        if kind == "coloring":
            return qqa.Coloring(g, num_category=extra.get("num_category", 3), device=device)
    if kind == "ising1d":
        return qqa.Ising1D(N=size, device=device)
    if kind == "ea":
        return qqa.EdwardsAnderson(
            L=size, dim=int(extra.get("dim", 3)), seed=cfg["seed"], device=device
        )
    if kind == "sk":
        return qqa.SherringtonKirkpatrick(N=size, seed=cfg["seed"], device=device)
    if kind == "perceptron":
        return qqa.BinaryPerceptron(
            N=size, alpha=float(extra.get("alpha", 0.5)), seed=cfg["seed"], device=device
        )
    if kind == "hopfield":
        return qqa.HopfieldMemory(
            N=size,
            patterns=int(extra.get("patterns", 3)),
            seed=cfg["seed"],
            device=device,
        )
    raise ValueError(f"Unknown problem kind {kind!r}")


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------


def _graph_preview(g: nx.Graph, title: str) -> None:
    pos = nx.spring_layout(g, seed=0)
    edge_x, edge_y = [], []
    for u, v in g.edges:
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])
    node_x = [pos[n][0] for n in g.nodes]
    node_y = [pos[n][1] for n in g.nodes]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"color": "#475569", "width": 1})
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker={"color": "#38bdf8", "size": 8, "line": {"color": "#0ea5e9", "width": 1}},
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"color": "#f8fafc"}},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def _coupling_preview(J: np.ndarray, title: str) -> None:
    fig = go.Figure(data=go.Heatmap(z=J, colorscale="RdBu", zmid=0, colorbar={"title": "J_ij"}))
    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"color": "#f8fafc"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def preview_problem(problem: Any, cfg: dict) -> None:
    kind = cfg["kind"]
    if hasattr(problem, "nx_graph"):
        _graph_preview(problem.nx_graph, f"{kind} graph (n={problem.num_nodes})")
        return
    if hasattr(problem, "J") and problem.J is not None:
        J = problem.J.detach().cpu().numpy()
        _coupling_preview(J, f"{kind} couplings (N={problem.num_spins})")
        return
    if kind == "perceptron":
        xi = problem.xi_signed.detach().cpu().numpy()
        fig = go.Figure(data=go.Heatmap(z=xi, colorscale="RdBu", zmid=0, colorbar={"title": "ξ̂"}))
        fig.update_layout(
            title={
                "text": f"Signed patterns ({problem.num_patterns} × {problem.num_spins})",
                "x": 0.5,
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        return
    if kind == "custom":
        import torch

        st.markdown(
            f"**Custom problem** — `{problem.num_vars}` variables "
            f"({problem.variable_kind}). Preview: single-batch loss evaluation."
        )
        with st.spinner("Evaluating loss on a random sample..."):
            try:
                if problem.variable_kind == "categorical":
                    x = torch.randn(1, problem.num_vars, problem.num_category).softmax(dim=-1)
                elif problem.variable_kind == "spin":
                    x = torch.rand(1, problem.num_vars)
                else:
                    x = torch.rand(1, problem.num_vars)
                val = problem.loss_fn(problem.relaxation(x))
                st.success(
                    f"loss_fn returns tensor shape {tuple(val.shape)}; "
                    f"sample value = {val.item():.4f}"
                )
            except Exception as e:  # pragma: no cover - surfaced in UI
                st.error(f"loss_fn raised: {e}")
        return
    st.info("No preview available for this problem type.")
