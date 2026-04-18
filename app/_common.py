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


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------

_LIGHT_PALETTE = {
    "bg": "#fbfaf6",
    "bg_card": "#ffffff",
    "bg_sidebar": "#f4f1ea",
    "border": "#e4e1d6",
    "text": "#0f172a",
    "muted": "#64748b",
    "accent": "#0f766e",  # deep teal
    "accent2": "#be5a3c",  # warm terracotta
    "grid": "#e5e7eb",
    "palette": ["#0f766e", "#be5a3c", "#1e3a8a", "#b45309", "#6d28d9", "#047857"],
}

_DARK_PALETTE = {
    "bg": "#050914",
    "bg_card": "rgba(15,23,42,0.66)",
    "bg_sidebar": "rgba(11,18,32,0.78)",
    "border": "rgba(148,163,184,0.18)",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "accent": "#38bdf8",
    "accent2": "#a855f7",
    "grid": "rgba(148,163,184,0.18)",
    "palette": ["#38bdf8", "#a855f7", "#f472b6", "#34d399", "#fbbf24", "#60a5fa"],
}


def get_theme() -> str:
    """Return the currently-selected theme (``light`` or ``dark``)."""
    return st.session_state.get("theme", "light")


def theme_toggle_in_sidebar() -> str:
    """Place a compact Light / Dark toggle at the top of the sidebar."""
    current = get_theme()
    with st.sidebar:
        choice = st.radio(
            "Theme",
            options=("light", "dark"),
            index=0 if current == "light" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="_theme_selector",
        )
    st.session_state["theme"] = choice
    return choice


def palette(theme: str | None = None) -> dict:
    """Return the colour palette for ``theme`` (defaults to current)."""
    theme = theme or get_theme()
    return _LIGHT_PALETTE if theme == "light" else _DARK_PALETTE


def plotly_layout(theme: str | None = None, **overrides) -> dict:
    """Plotly layout kwargs consistent with the active theme.

    Every chart in the dashboard passes ``fig.update_layout(**plotly_layout())``
    so colours, fonts and grid lines stay consistent.
    """
    p = palette(theme)
    base = {
        "template": "plotly_white" if (theme or get_theme()) == "light" else "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, -apple-system, sans-serif", "size": 13, "color": p["text"]},
        "title_font": {
            "family": "'Source Serif 4', Georgia, serif",
            "size": 17,
            "color": p["text"],
        },
        "colorway": p["palette"],
        "xaxis": {"gridcolor": p["grid"], "linecolor": p["border"], "zerolinecolor": p["grid"]},
        "yaxis": {"gridcolor": p["grid"], "linecolor": p["border"], "zerolinecolor": p["grid"]},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "bordercolor": p["border"], "borderwidth": 0.5},
        "margin": {"l": 50, "r": 20, "t": 48, "b": 46},
    }
    base.update(overrides)
    return base


def apply_theme() -> None:
    """Inject the active theme's CSS, professional-academic in light mode."""
    theme = get_theme()
    p = palette(theme)

    if theme == "light":
        css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');
        :root {{
            --qqa-bg: {p["bg"]};
            --qqa-card: {p["bg_card"]};
            --qqa-border: {p["border"]};
            --qqa-text: {p["text"]};
            --qqa-muted: {p["muted"]};
            --qqa-accent: {p["accent"]};
            --qqa-accent2: {p["accent2"]};
        }}
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .stApp {{
            background:
                radial-gradient(1200px 420px at 100% -10%, rgba(15,118,110,0.07), transparent 60%),
                radial-gradient(900px 320px at -10% 0%, rgba(190,90,60,0.05), transparent 60%),
                var(--qqa-bg);
            color: var(--qqa-text);
        }}
        section[data-testid="stSidebar"] {{
            background: {p["bg_sidebar"]};
            border-right: 1px solid var(--qqa-border);
        }}
        h1, h2, h3, h4 {{
            font-family: 'Source Serif 4', Georgia, serif;
            color: var(--qqa-text);
            letter-spacing: -0.015em;
            font-weight: 600;
        }}
        h1 {{
            font-size: 2.2rem;
            border-bottom: 1px solid var(--qqa-border);
            padding-bottom: 0.45rem;
            margin-bottom: 0.6rem;
        }}
        h1::before {{
            content: "";
            display: inline-block;
            width: 4px;
            height: 1.4rem;
            background: var(--qqa-accent);
            margin-right: 0.6rem;
            vertical-align: -3px;
            border-radius: 2px;
        }}
        .stCaption, [data-testid="stCaptionContainer"] p {{
            color: var(--qqa-muted) !important;
            font-variant: small-caps;
            letter-spacing: 0.04em;
        }}
        div[data-testid="stMetric"] {{
            background: var(--qqa-card);
            padding: 0.75rem 1rem;
            border-radius: 10px;
            border: 1px solid var(--qqa-border);
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        }}
        div[data-testid="stMetric"] label {{
            color: var(--qqa-muted) !important;
            font-weight: 500;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-weight: 700;
            color: var(--qqa-text) !important;
        }}
        .stButton > button {{
            background: var(--qqa-accent);
            color: #ffffff;
            border: 0;
            border-radius: 6px;
            padding: 0.45rem 1.2rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            transition: all 140ms ease;
        }}
        .stButton > button:hover {{
            background: #0b5a52;
            transform: translateY(-1px);
            box-shadow: 0 3px 10px rgba(15,118,110,0.18);
        }}
        .stButton > button:active {{ transform: translateY(0); }}
        /* ---- BaseWeb slider (light) ----------------------------------- */
        /* Empty track: subtle accent tint so the filled part stands out */
        .stSlider [data-baseweb="slider"] [role="slider"] ~ div,
        .stSlider [data-baseweb="slider"] > div > div > div:first-child {{
            background: rgba(15,118,110,0.18) !important;
        }}
        /* Filled portion of the track (left of thumb) */
        .stSlider [data-baseweb="slider"] > div > div > div:first-child > div {{
            background: var(--qqa-accent) !important;
        }}
        /* Thumb: white puck with accent ring so it's always visible */
        .stSlider [data-baseweb="slider"] [role="slider"] {{
            background: #ffffff !important;
            border: 2px solid var(--qqa-accent) !important;
            box-shadow: 0 1px 3px rgba(15,23,42,0.18) !important;
        }}
        /* Current-value bubble above the thumb */
        .stSlider [data-baseweb="slider"] [role="slider"] > div {{
            color: var(--qqa-accent) !important;
            background: transparent !important;
            font-weight: 600 !important;
        }}
        /* Min / max tick labels at the ends of the slider */
        .stSlider [data-testid="stTickBar"],
        .stSlider [data-testid="stTickBar"] > div {{
            color: var(--qqa-muted) !important;
            background: transparent !important;
        }}
        code, pre {{
            background: #f0ece3;
            color: #334155;
            font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.86rem;
        }}
        .qqa-card {{
            background: var(--qqa-card);
            border: 1px solid var(--qqa-border);
            padding: 1rem 1.25rem;
            border-radius: 10px;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        }}
        .qqa-score {{
            background: linear-gradient(135deg, rgba(15,118,110,0.06), rgba(30,58,138,0.06));
            border: 1px solid var(--qqa-accent);
            border-left: 4px solid var(--qqa-accent);
            padding: 1rem 1.4rem;
            border-radius: 8px;
            margin: 0.6rem 0 1rem 0;
        }}
        .qqa-score .label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--qqa-muted);
        }}
        .qqa-score .value {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--qqa-text);
            line-height: 1.1;
        }}
        .qqa-score .value.infeasible {{ color: #b45309; }}
        .qqa-score .unit {{
            font-size: 1rem;
            color: var(--qqa-muted);
            font-weight: 500;
            margin-left: 0.4rem;
        }}
        .qqa-score .raw {{
            color: var(--qqa-muted);
            font-size: 0.85rem;
            margin-top: 0.3rem;
        }}
        .qqa-badge {{
            display: inline-block;
            padding: 0.12rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .qqa-badge.ok {{ background: #dcfce7; color: #14532d; }}
        .qqa-badge.warn {{ background: #ffedd5; color: #9a3412; }}
        </style>
        """
    else:
        css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');
        :root {{
            --qqa-bg: {p["bg"]};
            --qqa-card: {p["bg_card"]};
            --qqa-border: {p["border"]};
            --qqa-text: {p["text"]};
            --qqa-muted: {p["muted"]};
            --qqa-accent: {p["accent"]};
            --qqa-accent2: {p["accent2"]};
        }}
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{
            background:
                radial-gradient(900px 500px at 10% -10%, rgba(56,189,248,0.18), transparent 60%),
                radial-gradient(900px 500px at 120% 20%, rgba(168,85,247,0.18), transparent 60%),
                linear-gradient(180deg, #05060a 0%, #0b1120 100%);
            color: var(--qqa-text);
        }}
        section[data-testid="stSidebar"] {{
            background: {p["bg_sidebar"]};
            backdrop-filter: blur(14px);
            border-right: 1px solid var(--qqa-border);
        }}
        h1, h2, h3, h4 {{
            font-family: 'Source Serif 4', serif;
            color: #f8fafc;
            letter-spacing: -0.01em;
        }}
        h1 {{
            background: linear-gradient(120deg, #e0f2fe, #a855f7, #38bdf8);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        div[data-testid="stMetric"] {{
            background: rgba(15,23,42,0.55);
            padding: 0.7rem 1rem;
            border-radius: 14px;
            border: 1px solid var(--qqa-border);
            backdrop-filter: blur(8px);
        }}
        div[data-testid="stMetric"] label {{
            color: var(--qqa-muted) !important;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            font-family: 'Source Serif 4', serif;
            color: #f8fafc !important;
        }}
        .stButton > button {{
            background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
            color: white;
            border: 0;
            border-radius: 999px;
            padding: 0.45rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 6px 22px rgba(56,189,248,0.25);
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 10px 28px rgba(168,85,247,0.35);
        }}
        /* ---- BaseWeb slider (dark) ------------------------------------ */
        .stSlider [data-baseweb="slider"] [role="slider"] ~ div,
        .stSlider [data-baseweb="slider"] > div > div > div:first-child {{
            background: rgba(148,163,184,0.25) !important;
        }}
        .stSlider [data-baseweb="slider"] > div > div > div:first-child > div {{
            background: linear-gradient(90deg, #38bdf8, #a855f7) !important;
        }}
        .stSlider [data-baseweb="slider"] [role="slider"] {{
            background: #0b1120 !important;
            border: 2px solid #38bdf8 !important;
            box-shadow: 0 0 0 2px rgba(56,189,248,0.25) !important;
        }}
        .stSlider [data-baseweb="slider"] [role="slider"] > div {{
            color: #e2e8f0 !important;
            background: transparent !important;
        }}
        .stSlider [data-testid="stTickBar"],
        .stSlider [data-testid="stTickBar"] > div {{
            color: var(--qqa-muted) !important;
            background: transparent !important;
        }}
        code, pre {{
            background: rgba(15,23,42,0.75);
            color: #bae6fd;
            font-family: 'JetBrains Mono', monospace;
        }}
        .qqa-card {{
            background: var(--qqa-card);
            border: 1px solid var(--qqa-border);
            padding: 1rem 1.2rem;
            border-radius: 16px;
            backdrop-filter: blur(10px);
        }}
        .qqa-score {{
            background: linear-gradient(135deg, rgba(56,189,248,0.14), rgba(168,85,247,0.14));
            border: 1px solid rgba(56,189,248,0.4);
            border-left: 4px solid var(--qqa-accent);
            padding: 1rem 1.4rem;
            border-radius: 12px;
            margin: 0.6rem 0 1rem 0;
        }}
        .qqa-score .label {{
            font-size: 0.72rem; letter-spacing: 0.12em;
            text-transform: uppercase; color: var(--qqa-muted);
        }}
        .qqa-score .value {{
            font-family: 'Source Serif 4', serif;
            font-size: 2.2rem; font-weight: 700; color: #f8fafc;
        }}
        .qqa-score .value.infeasible {{ color: #fcd34d; }}
        .qqa-score .unit {{
            font-size: 1rem; color: var(--qqa-muted);
            font-weight: 500; margin-left: 0.4rem;
        }}
        .qqa-score .raw {{
            color: var(--qqa-muted); font-size: 0.85rem; margin-top: 0.3rem;
        }}
        .qqa-badge {{
            display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .qqa-badge.ok {{ background: rgba(34,197,94,0.24); color: #a7f3d0; }}
        .qqa-badge.warn {{ background: rgba(245,158,11,0.24); color: #fde68a; }}
        </style>
        """

    st.markdown(css, unsafe_allow_html=True)


def render_score_card(score: dict, raw_loss: float | None = None) -> None:
    """Render the big problem-specific score tile used by the Solve page."""
    if not score:
        return
    feas = score.get("feasible", True)
    badge = (
        '<span class="qqa-badge ok">feasible</span>'
        if feas
        else '<span class="qqa-badge warn">infeasible</span>'
    )
    value = score.get("value", "-")
    value_s = f"{value:.4g}" if isinstance(value, float) else str(value)
    unit = score.get("unit", "")
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    raw_html = f'<div class="raw">raw loss = {raw_loss:.4g}</div>' if raw_loss is not None else ""
    value_cls = "value" if feas else "value infeasible"
    st.markdown(
        f'<div class="qqa-score">'
        f'<div class="label">{score.get("label", "score")} · {badge}</div>'
        f'<div class="{value_cls}">{value_s}{unit_html}</div>'
        f"{raw_html}"
        "</div>",
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

    # -------- new (Phase A) problems ---------------------------------------
    if kind == "knapsack":
        return qqa.Knapsack(
            N=size,
            capacity_ratio=float(extra.get("capacity_ratio", 0.5)),
            seed=cfg["seed"],
            device=device,
        )
    if kind == "number_partition":
        return qqa.NumberPartitioning(
            N=size,
            max_value=int(extra.get("max_value", 100)),
            seed=cfg["seed"],
            device=device,
        )
    if kind == "vertex_cover":
        d = extra.get("graph_d", 3)
        if (size * d) % 2 != 0:
            d = max(2, d - 1) if d > 2 else d + 1
        g = nx.random_regular_graph(d=d, n=size, seed=cfg["seed"])
        return qqa.VertexCover(g, device=device)
    if kind == "graph_bisection":
        d = extra.get("graph_d", 3)
        if (size * d) % 2 != 0:
            d = max(2, d - 1) if d > 2 else d + 1
        g = nx.random_regular_graph(d=d, n=size, seed=cfg["seed"])
        return qqa.GraphBisection(
            g,
            balance_penalty=float(extra.get("balance_penalty", 2.0)),
            device=device,
        )
    if kind == "maxsat3":
        return qqa.MaxSAT3(
            N=size,
            ratio=float(extra.get("ratio", 3.0)),
            seed=cfg["seed"],
            device=device,
        )
    if kind == "tsp":
        return qqa.TSP(
            N=size,
            column_penalty=float(extra.get("column_penalty", 3.0)),
            seed=cfg["seed"],
            device=device,
        )
    if kind == "qap":
        return qqa.QAP(
            N=size,
            column_penalty=float(extra.get("column_penalty", 10.0)),
            seed=cfg["seed"],
            device=device,
        )
    if kind == "nqueens":
        return qqa.NQueens(N=size, device=device)

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
    st.plotly_chart(fig, width="stretch")


def _coupling_preview(J: np.ndarray, title: str) -> None:
    fig = go.Figure(data=go.Heatmap(z=J, colorscale="RdBu", zmid=0, colorbar={"title": "J_ij"}))
    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"color": "#f8fafc"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")


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
        st.plotly_chart(fig, width="stretch")
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
