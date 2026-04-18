"""Shared helpers for the QQA Streamlit app.

Keeps problem construction, previews, and theming in one place so the Home /
Solve / Visualize / Compare pages stay declarative.
"""

from __future__ import annotations

import contextlib
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

# Curated example library shown in the Custom-problem editor.  Each entry
# carries the snippet *and* the metadata the sidebar form needs (variable
# kind, default size, optional category count, one-line description).
# Keeping the metadata next to the snippet means the UI can auto-populate
# the form when a template is loaded — no more "I picked SK but forgot to
# set variable_kind=spin".
CUSTOM_EXAMPLES: dict[str, dict] = {
    "Spin glass · Sherrington–Kirkpatrick (SK)": {
        "kind": "spin",
        "num_vars": 32,
        "num_category": None,
        "description": "Mean-field spin glass with Gaussian couplings. Classic NP-hard benchmark.",
        "source": DEFAULT_CUSTOM_SNIPPET,
    },
    "Spin · Number partitioning": {
        "kind": "spin",
        "num_vars": 32,
        "num_category": None,
        "description": "Split N positive integers into two equal-sum groups (a_i s_i)^2.",
        "source": '''import torch

N = 32
g = torch.Generator().manual_seed(0)
a = torch.randint(1, 100, (N,), generator=g).float()


def loss_fn(s):
    """Minimise the squared imbalance Σ a_i s_i with s ∈ {-1,+1}^N."""
    return (s @ a) ** 2
''',
    },
    "Spin · Ferromagnetic Ising chain": {
        "kind": "spin",
        "num_vars": 64,
        "num_category": None,
        "description": "1-D Ising ferromagnet — instructive sanity check, easy to solve.",
        "source": '''import torch

N = 64
J = 1.0
h = 0.0


def loss_fn(s):
    """1-D ferromagnet: -J Σ s_i s_{i+1} - h Σ s_i, s ∈ {-1,+1}."""
    return -J * (s[:, :-1] * s[:, 1:]).sum(dim=1) - h * s.sum(dim=1)
''',
    },
    "Binary · Weighted MaxCut": {
        "kind": "binary",
        "num_vars": 32,
        "num_category": None,
        "description": "Bipartition vertices to maximise Σ W_ij (x_i + x_j − 2 x_i x_j).",
        "source": '''import torch

N = 32
g = torch.Generator().manual_seed(0)
W = torch.rand(N, N, generator=g)
W = (W + W.T) / 2
W.fill_diagonal_(0.0)


def loss_fn(x):
    """Maximise Σ_{i<j} W_ij (x_i + x_j - 2 x_i x_j) — a weighted MaxCut.

    The annealer minimises, so we negate.
    """
    cut = torch.einsum("ij,bi,bj->b", W, x, 1 - x)
    return -cut
''',
    },
    "Binary · Generic QUBO from a matrix": {
        "kind": "binary",
        "num_vars": 24,
        "num_category": None,
        "description": "Energy x^T Q x — paste your own Q for any QUBO.",
        "source": '''import torch

# Replace Q with your own (N×N) matrix. The energy is x^T Q x.
N = 24
g = torch.Generator().manual_seed(7)
Q = torch.randn(N, N, generator=g)
Q = (Q + Q.T) / 2


def loss_fn(x):
    """Generic QUBO loss for binary x ∈ {0,1}^N."""
    return torch.einsum("ij,bi,bj->b", Q, x, x)
''',
    },
    "Binary · Random 3-SAT clause count": {
        "kind": "binary",
        "num_vars": 30,
        "num_category": None,
        "description": "Minimise unsatisfied clauses on a random 3-SAT formula (M = 90).",
        "source": """import torch

N = 30
M = 90  # clauses
g = torch.Generator().manual_seed(0)
lit = torch.randint(0, 2 * N, (M, 3), generator=g)  # variable-with-sign codes
sign = (lit % 2 == 0).float() * 2.0 - 1.0  # +1 if positive literal, -1 if negated
var = lit // 2


def loss_fn(x):
    # Map x ∈ {0,1} to ±1 literal evaluations.
    spins = 2 * x - 1.0  # (B, N)
    chosen = spins[:, var]  # (B, M, 3)
    eval_lit = chosen * sign  # +1 if literal satisfied
    # Clause unsatisfied iff every literal is -1, i.e. product == -1
    # easier: clause "value" = max over literals; we approximate with mean.
    sat = ((eval_lit + 1) / 2).max(dim=-1).values  # (B, M)
    return (1.0 - sat).sum(dim=-1)
""",
    },
    "Categorical · Toy graph 3-colouring": {
        "kind": "categorical",
        "num_vars": 12,
        "num_category": 3,
        "description": "Minimise edge conflicts on a small ring graph with K=3 colours.",
        "source": '''import torch

N = 12
K = 3
# Ring graph: edges (i, i+1) and the wrap-around (N-1, 0).
edges = torch.tensor([[i, (i + 1) % N] for i in range(N)])


def loss_fn(x):
    """``x`` is one-hot of shape (B, N, K). Penalise endpoints sharing a colour."""
    # (B, |E|, K): inner product per edge per colour.
    overlap = (x[:, edges[:, 0]] * x[:, edges[:, 1]]).sum(dim=-1)
    return overlap.sum(dim=-1)
''',
    },
}


# Keep a flat ``label -> source_str`` map for code that only wants the
# snippet (preview, AppTest, downstream tooling) — saves every caller from
# digging into the metadata dict and keeps backward-compat.
CUSTOM_EXAMPLE_SOURCES: dict[str, str] = {k: v["source"] for k, v in CUSTOM_EXAMPLES.items()}


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


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a ``#RRGGBB`` hex string into a Plotly-safe ``rgba(...)`` string.

    Plotly's property validator rejects the 8-digit ``#RRGGBBAA`` form, so we
    emit the functional ``rgba()`` notation instead (``alpha`` is a float in
    ``[0, 1]``). Non-hex inputs are returned unchanged so callers can safely
    pass already-resolved strings.
    """
    if not isinstance(color, str) or not color.startswith("#"):
        return color
    h = color.lstrip("#")
    if len(h) == 8:  # already hex8 → drop alpha and use caller's alpha instead
        h = h[:6]
    if len(h) != 6:
        return color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


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
            "size": 16,
            "color": p["text"],
        },
        "colorway": p["palette"],
        "xaxis": {"gridcolor": p["grid"], "linecolor": p["border"], "zerolinecolor": p["grid"]},
        "yaxis": {"gridcolor": p["grid"], "linecolor": p["border"], "zerolinecolor": p["grid"]},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "bordercolor": p["border"], "borderwidth": 0.5},
        # Top margin keeps the title clear of Plotly's modebar (which lives
        # in the top-right corner). Was 48, and even modest titles collided.
        "margin": {"l": 56, "r": 28, "t": 64, "b": 50},
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
        /* Hide the auto-generated multipage heading ("streamlit app")
           rendered above our brand block, AND hide the entry-page
           navigation link itself (the brand logo above is the home
           anchor; an extra "streamlit app" link is just noise). */
        [data-testid="stSidebarNav"]::before {{ display: none; }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > div:first-child,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > ul + div:has(>h1),
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] h1:first-of-type,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] h2:first-of-type {{
            display: none !important;
        }}
        /* Drop the first <li> in the nav list — that is the entry page
           link Streamlit derives from the file name (here: "streamlit
           app"). The brand block + the page list under it is enough. */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li:first-child {{
            display: none !important;
        }}
        [data-testid="stSidebarNav"] a {{
            color: var(--qqa-text) !important;
            font-weight: 500;
            border-radius: 6px;
        }}
        [data-testid="stSidebarNav"] a:hover {{
            background: rgba(15,118,110,0.10) !important;
            color: var(--qqa-accent) !important;
        }}
        [data-testid="stSidebarNav"] a span {{ color: inherit !important; }}
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
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            white-space: nowrap;
            overflow: visible;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-weight: 700;
            font-size: 1.35rem;
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
        /* Dark-mode text contrast. The Streamlit theme is declared ``light``
           in config.toml so that ``textColor`` defaults to a dark tone; the
           rules below re-colour every bit of Streamlit-managed text so the
           dark background stays readable. Our intentionally-coloured
           classes (.qqa-badge, .qqa-score .value, ...) are excluded. */
        .stApp div[data-testid="stMarkdownContainer"] p,
        .stApp div[data-testid="stMarkdownContainer"] li,
        .stApp div[data-testid="stMarkdownContainer"] span:not([class*="qqa-"]),
        .stApp div[data-testid="stMarkdownContainer"] strong,
        .stApp div[data-testid="stMarkdownContainer"] em {{
            color: #e8edf7 !important;
        }}
        .stApp a {{ color: #93c5fd !important; }}
        .stApp a:hover {{ color: #bfdbfe !important; }}
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span,
        section[data-testid="stSidebar"] [data-testid="stRadio"] label,
        section[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
        section[data-testid="stSidebar"] [data-testid="stNumberInput"] label,
        section[data-testid="stSidebar"] [data-testid="stSlider"] label,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
            color: #e8edf7 !important;
        }}
        section[data-testid="stSidebar"] svg {{ fill: #e8edf7 !important; }}
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
        /* Same nav-header suppression as in light theme. */
        [data-testid="stSidebarNav"]::before {{ display: none; }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li:first-child {{
            display: none !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > div:first-child,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > ul + div:has(>h1),
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] h1:first-of-type,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] h2:first-of-type {{
            display: none !important;
        }}
        [data-testid="stSidebarNav"] a {{
            color: #e8edf7 !important;
            font-weight: 500;
            border-radius: 6px;
        }}
        [data-testid="stSidebarNav"] a:hover {{
            background: rgba(56,189,248,0.14) !important;
            color: #bae6fd !important;
        }}
        [data-testid="stSidebarNav"] a span {{ color: inherit !important; }}
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Source Serif 4', serif;
            color: #f8fafc !important;
            letter-spacing: -0.01em;
        }}
        h1 {{
            background: linear-gradient(120deg, #e0f2fe, #a855f7, #38bdf8);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        /* Caption / muted text in dark mode stays readable but softer. */
        .stCaption, [data-testid="stCaptionContainer"] p {{
            color: #94a3b8 !important;
            font-variant: small-caps;
            letter-spacing: 0.04em;
        }}
        /* Radio-toggle label ("light" / "dark") on the sidebar. */
        section[data-testid="stSidebar"] [data-testid="stRadio"] label p {{
            color: #e8edf7 !important;
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
            font-size: 0.7rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            white-space: nowrap;
            overflow: visible;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            font-family: 'Source Serif 4', serif;
            font-size: 1.35rem;
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


# ---------------------------------------------------------------------------
# Sidebar brand + paper-link footer (used by every page)
# ---------------------------------------------------------------------------

# URLs surfaced in the UI. Keep in sync with [project.urls] in pyproject.toml.
_PAPER_URL = "https://openreview.net/forum?id=9EfBeXaXf0"
_GITHUB_URL = "https://github.com/Yuma-Ichikawa/QQA4CO"
_DEMO_URL = "https://parallelquasiquantum4co.streamlit.app/"


def sidebar_brand() -> None:
    """Render the QQA brand block at the top of the sidebar.

    Also hides the auto ``streamlit app`` heading injected by Streamlit's
    built-in multipage navigator (CSS takes care of that — this function
    just adds the branded block above the nav).
    """
    theme = get_theme()
    accent = "#0f766e" if theme == "light" else "#38bdf8"
    accent2 = "#be5a3c" if theme == "light" else "#a855f7"
    muted = "#64748b" if theme == "light" else "#94a3b8"
    with st.sidebar:
        st.markdown(
            f"""
            <div class="qqa-brand" style="
                padding: 0.75rem 0.4rem 0.9rem 0.4rem;
                margin-bottom: 0.4rem;
                border-bottom: 1px solid rgba(148,163,184,0.2);
            ">
              <div style="display:flex;align-items:center;gap:0.55rem;">
                <div style="
                    width:32px;height:32px;border-radius:8px;
                    background:linear-gradient(135deg,{accent} 0%,{accent2} 100%);
                    display:flex;align-items:center;justify-content:center;
                    color:#fff;font-weight:700;font-family:'Source Serif 4',serif;
                    box-shadow:0 2px 6px rgba(15,23,42,0.18);
                ">Q</div>
                <div>
                  <div style="
                      font-family:'Source Serif 4',Georgia,serif;
                      font-weight:700;font-size:1.05rem;line-height:1.1;
                  ">QQA4CO</div>
                  <div style="font-size:0.7rem;color:{muted};letter-spacing:0.04em;">
                    Quasi-Quantum Annealing
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def empty_state_card(
    *,
    title: str,
    body: str,
    cta_label: str = "Open Solve",
    cta_page: str = "pages/1_Solve.py",
) -> None:
    """Branded empty-state card with a primary CTA.

    Used on Visualize / Compare when no run is yet available, so the
    user sees a deliberate path forward instead of a bare warning row.
    Falls back gracefully to a markdown-only card if the running
    Streamlit predates ``st.page_link``.
    """
    theme = get_theme()
    p = palette()
    accent = "#0f766e" if theme == "light" else "#38bdf8"
    muted = "#64748b" if theme == "light" else "#94a3b8"
    st.markdown(
        f"""
        <div style="
            border:1px solid {p["border"]};
            background:{p["bg_card"]};
            padding:1.4rem 1.6rem;
            border-radius:12px;
            box-shadow:0 1px 3px rgba(15,23,42,0.05);
        ">
          <div style="
              font-family:'Source Serif 4',Georgia,serif;
              font-weight:600;font-size:1.15rem;color:{p["text"]};
              margin-bottom:0.45rem;
          ">{title}</div>
          <div style="font-size:0.95rem;color:{muted};line-height:1.45;">
            {body}
          </div>
          <div style="height:0.85rem;"></div>
          <div style="
              display:inline-block;padding:0.35rem 0.85rem;border-radius:8px;
              border:1px solid {accent};color:{accent};font-weight:500;
              font-size:0.9rem;
          ">↳ {cta_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if hasattr(st, "page_link"):
        with contextlib.suppress(Exception):
            st.page_link(cta_page, label=cta_label, icon="▶")


def paper_link_footer() -> None:
    """Compact link row rendered at the very bottom of the sidebar."""
    theme = get_theme()
    muted = "#64748b" if theme == "light" else "#94a3b8"
    link = "#0f766e" if theme == "light" else "#93c5fd"
    with st.sidebar:
        st.markdown(
            f"""
            <div style="
                margin-top: 1.2rem;
                padding-top: 0.8rem;
                border-top: 1px solid rgba(148,163,184,0.2);
                font-size: 0.78rem;
                color: {muted};
            ">
              <div style="margin-bottom:0.3rem;letter-spacing:0.04em;
                   text-transform:uppercase;font-size:0.66rem;">
                References
              </div>
              <div style="display:flex;flex-direction:column;gap:0.3rem;">
                <a href="{_PAPER_URL}" target="_blank" style="color:{link};
                   text-decoration:none;">📄  Paper (OpenReview)</a>
                <a href="{_GITHUB_URL}" target="_blank" style="color:{link};
                   text-decoration:none;">⎇  GitHub repository</a>
                <a href="{_DEMO_URL}" target="_blank" style="color:{link};
                   text-decoration:none;">🚀  Hosted live demo</a>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def hero_badges() -> None:
    """Horizontal badge row for the Home hero (paper / repo / demo)."""
    theme = get_theme()
    accent = "#0f766e" if theme == "light" else "#38bdf8"
    border = "#d4d0c3" if theme == "light" else "rgba(148,163,184,0.25)"
    bg = "#ffffff" if theme == "light" else "rgba(15,23,42,0.55)"
    st.markdown(
        f"""
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin:0.3rem 0 0.9rem 0;">
          <a href="{_PAPER_URL}" target="_blank" style="
              display:inline-flex;align-items:center;gap:0.35rem;
              padding:0.26rem 0.7rem;border:1px solid {accent};
              background:{bg};color:{accent};border-radius:999px;
              text-decoration:none;font-size:0.8rem;font-weight:600;
          ">📄 OpenReview paper</a>
          <a href="{_GITHUB_URL}" target="_blank" style="
              display:inline-flex;align-items:center;gap:0.35rem;
              padding:0.26rem 0.7rem;border:1px solid {border};
              background:{bg};color:inherit;border-radius:999px;
              text-decoration:none;font-size:0.8rem;font-weight:600;
          ">⎇ Source on GitHub</a>
          <a href="{_DEMO_URL}" target="_blank" style="
              display:inline-flex;align-items:center;gap:0.35rem;
              padding:0.26rem 0.7rem;border:1px solid {border};
              background:{bg};color:inherit;border-radius:999px;
              text-decoration:none;font-size:0.8rem;font-weight:600;
          ">🚀 Live demo</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    seed = cfg["seed"]
    if kind in {"mis", "maxcut", "maxclique", "coloring", "vertex_cover", "graph_bisection"}:
        return _build_graph_problem(kind, size, seed, device, extra)
    if kind == "ising1d":
        return _safe_call(qqa.Ising1D, N=size, device=device)
    if kind == "ea":
        return _safe_call(
            qqa.EdwardsAnderson,
            L=size,
            dim=int(extra.get("dim", 3)),
            seed=seed,
            device=device,
        )
    if kind == "sk":
        return _safe_call(qqa.SherringtonKirkpatrick, N=size, seed=seed, device=device)
    if kind == "perceptron":
        return _safe_call(
            qqa.BinaryPerceptron,
            N=size,
            alpha=float(extra.get("alpha", 0.5)),
            seed=seed,
            device=device,
        )
    if kind == "hopfield":
        return _safe_call(
            qqa.HopfieldMemory,
            N=size,
            patterns=int(extra.get("patterns", 3)),
            seed=seed,
            device=device,
        )
    if kind == "knapsack":
        return _safe_call(
            qqa.Knapsack,
            N=size,
            capacity_ratio=float(extra.get("capacity_ratio", 0.5)),
            seed=seed,
            device=device,
        )
    if kind == "number_partition":
        return _safe_call(
            qqa.NumberPartitioning,
            N=size,
            max_value=int(extra.get("max_value", 100)),
            seed=seed,
            device=device,
        )
    if kind == "maxsat3":
        return _safe_call(
            qqa.MaxSAT3,
            N=size,
            ratio=float(extra.get("ratio", 3.0)),
            seed=seed,
            device=device,
        )
    if kind == "tsp":
        # Multi-penalty problem: forward every penalty-shaped key from
        # ``extra`` so the dashboard can declare an arbitrary number of
        # penalty terms without touching this dispatcher.
        return _safe_call(
            qqa.TSP,
            N=size,
            seed=seed,
            device=device,
            **_extract_penalty_kwargs(extra, defaults={"row_penalty": 5.0, "col_penalty": 5.0}),
        )
    if kind == "qap":
        return _safe_call(
            qqa.QAP,
            N=size,
            seed=seed,
            device=device,
            **_extract_penalty_kwargs(extra, defaults={"column_penalty": 10.0}),
        )
    if kind == "nqueens":
        return _safe_call(qqa.NQueens, N=size, device=device)

    raise ValueError(f"Unknown problem kind {kind!r}")


def _build_graph_problem(kind: str, size: int, seed: int, device: str, extra: dict):
    """Random-regular-graph problems share a common preamble (degree
    sanitisation + ``nx.random_regular_graph``), so factor it out."""
    d = extra.get("graph_d", 3)
    if (size * d) % 2 != 0:
        d = max(2, d - 1) if d > 2 else d + 1
    g = nx.random_regular_graph(d=d, n=size, seed=seed)
    if kind == "mis":
        return _safe_call(qqa.MaximumIndependentSet, g, device=device)
    if kind == "maxcut":
        return _safe_call(qqa.MaxCut, g, device=device)
    if kind == "maxclique":
        return _safe_call(qqa.MaxClique, g, device=device)
    if kind == "coloring":
        return _safe_call(qqa.Coloring, g, num_category=extra.get("num_category", 3), device=device)
    if kind == "vertex_cover":
        return _safe_call(qqa.VertexCover, g, device=device)
    if kind == "graph_bisection":
        return _safe_call(
            qqa.GraphBisection,
            g,
            balance_penalty=float(extra.get("balance_penalty", 2.0)),
            device=device,
        )
    raise ValueError(f"Unknown graph-problem kind {kind!r}")


# ---------------------------------------------------------------------------
# Constructor dispatch helpers — keep ``build_problem`` and the saved-config
# format decoupled from individual class signatures.
# ---------------------------------------------------------------------------


def _safe_call(cls, *args, **kwargs):
    """Invoke ``cls(*args, **kwargs)`` but silently drop any keyword that
    its ``__init__`` does not accept.

    Why: the Streamlit ``problem_config`` dict is persisted across reruns
    via ``st.session_state``. If a problem class evolves (e.g. TSP renames
    ``column_penalty`` → ``row_penalty``/``col_penalty``) the next page
    refresh would otherwise crash because the old key is still in
    ``extra``. Filtering against the constructor signature on the
    receiving end makes the call boundary forward- and backward-compatible
    by construction.
    """
    import inspect  # noqa: PLC0415 - lazy: only needed here

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return cls(*args, **kwargs)

    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        return cls(*args, **kwargs)

    accepted = set(sig.parameters)
    accepted.discard("self")
    safe = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = set(kwargs) - set(safe)
    if dropped:
        # Surface the drop in the UI without crashing the page. Streamlit
        # may not be initialised in test contexts, so guard the call.
        try:
            import streamlit as _st  # noqa: PLC0415

            _st.caption(
                f"Note — dropped unknown {cls.__name__} kwargs from session "
                f"state: {sorted(dropped)} (signature has changed)."
            )
        except Exception:
            pass
    return cls(*args, **safe)


# Recognised penalty-coefficient suffixes / aliases.  Adding a new
# penalty term to a problem only requires (a) adding a new keyword to the
# class' ``__init__`` and (b) declaring the slider in
# ``streamlit_app.py`` — this dispatcher does **not** need to change.
_PENALTY_SUFFIXES: tuple[str, ...] = ("_penalty", "_weight", "_lambda")
# Legacy keys → modern keys.  Two purposes:
#   * keep saved configs working after a rename;
#   * let users typing `column_penalty` (the old name) still get the
#     intended behaviour (mapped to row + col).
_PENALTY_ALIASES: dict[str, tuple[str, ...]] = {
    "column_penalty": ("row_penalty", "col_penalty"),
}


def _extract_penalty_kwargs(extra: dict, *, defaults: dict[str, float]) -> dict[str, float]:
    """Return a dict of penalty-shaped kwargs, merging ``defaults``,
    explicit ``extra`` keys, dict-form ``penalty_weights``, and legacy
    aliases. Numeric values are coerced to ``float`` so torch is happy.

    Selection rules (later overrides earlier):
        1. ``defaults`` (lowest priority)
        2. legacy aliases in ``extra`` (e.g. ``column_penalty`` mapped to
           both ``row_penalty`` and ``col_penalty``)
        3. explicit penalty-suffixed keys in ``extra`` (override legacy)
        4. ``extra['penalty_weights']`` dict (most explicit ⇒ wins)

    To avoid a stale legacy key drowning a fresh modern key, the legacy
    alias itself is **never** propagated to the output dict; only its
    modern translations are.
    """
    out: dict[str, float] = dict(defaults)
    legacy_targets: set[str] = set()
    for modern_keys in _PENALTY_ALIASES.values():
        legacy_targets.update(modern_keys)

    # 2. legacy aliases (translate, do not propagate the legacy key itself).
    for legacy, modern_keys in _PENALTY_ALIASES.items():
        if legacy in extra:
            try:
                v = float(extra[legacy])
            except (TypeError, ValueError):
                continue
            for k in modern_keys:
                out[k] = v

    # 3. explicit penalty-shaped keys override legacy translations.
    for k, v in extra.items():
        if k in _PENALTY_ALIASES:
            continue  # already handled in step 2
        if any(k.endswith(suf) for suf in _PENALTY_SUFFIXES):
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue

    # 4. structured dict overrides every preceding source.
    pw = extra.get("penalty_weights")
    if isinstance(pw, dict):
        for k, v in pw.items():
            key = k if any(k.endswith(suf) for suf in _PENALTY_SUFFIXES) else f"{k}_penalty"
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                continue

    return out


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
    """Show the coupling matrix without melting the browser.

    Plotly's ``Heatmap`` is dense — every cell becomes a SVG rect. For
    ``N ≳ 1000`` the trace alone is hundreds of MB; for the default EA
    setting (``L=32, dim=3 ⇒ N=32 768``) it instantly OOMs the tab.
    Past ``N_show=256`` we fall back to a *sparse* spy plot of the
    non-zero couplings (still ``O(nnz)`` markers, not ``O(N²)`` rects).
    Spin glasses on a hyper-cubic lattice have ``≈ d·N`` non-zeros, so
    this stays well-behaved even at 32k spins.
    """
    N = J.shape[0]
    # 600^2 ≈ 360k cells — Plotly handles that comfortably; 1k^2 = 1M is
    # already laggy. Anything bigger ⇒ fall back to a sparse view.
    N_show = 600
    if N_show >= N:
        fig = go.Figure(data=go.Heatmap(z=J, colorscale="RdBu", zmid=0, colorbar={"title": "J_ij"}))
        fig.update_layout(
            title={"text": title, "x": 0.5, "font": {"color": "#f8fafc"}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")
        return

    rows, cols = np.nonzero(J)
    if rows.size == 0:
        st.info(f"{title}: coupling matrix is all-zero (N={N}).")
        return
    vals = J[rows, cols]
    vmax = float(np.abs(vals).max())
    fig = go.Figure(
        data=go.Scatter(
            x=cols,
            y=rows,
            mode="markers",
            marker={
                "size": 4,
                "color": vals,
                "colorscale": "RdBu",
                "cmin": -vmax,
                "cmax": vmax,
                "colorbar": {"title": "J_ij", "thickness": 12},
                "line": {"width": 0},
            },
            hovertemplate="i=%{y}, j=%{x}<br>J=%{marker.color:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={
            "text": f"{title} — sparse view ({rows.size} non-zero entries)",
            "x": 0.5,
            "font": {"color": "#f8fafc"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=440,
        xaxis={"title": "j", "scaleanchor": "y", "scaleratio": 1, "autorange": True},
        yaxis={"title": "i", "autorange": "reversed"},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Showing {rows.size} non-zero couplings out of {N * N:,} matrix entries. "
        "A dense heatmap at this size would crash the browser."
    )


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
                # Build a latent of the same shape ``qqa.anneal`` would allocate,
                # then push it through the relaxation's ``forward`` so the value
                # we hand to ``loss_fn`` matches what the optimiser will actually
                # see (continuous spins in [-1, +1] / one-hot simplex / etc.).
                # ``problem.relaxation`` is a *Relaxation instance* — calling it
                # directly raised "'SpinRelaxation' object is not callable".
                relax = problem.relaxation
                if problem.variable_kind == "categorical":
                    latent = torch.rand(1, problem.num_vars, problem.num_category)
                else:
                    latent = torch.rand(1, problem.num_vars)
                x = relax.forward(latent)
                val = problem.loss_fn(x)
                if not torch.is_tensor(val):
                    raise TypeError(
                        f"loss_fn must return a torch.Tensor, got {type(val).__name__}."
                    )
                if val.shape[0] != 1:
                    raise ValueError(
                        f"loss_fn returned shape {tuple(val.shape)} — expected a "
                        "leading batch axis matching the input (B=1 here)."
                    )
                # Also evaluate the discrete projection so users see the kind of
                # number QQA tracks as ``best_obj``.
                with torch.no_grad():
                    val_disc = problem.loss_fn(relax.project(latent))
                st.success(
                    f"loss_fn output shape {tuple(val.shape)} ✓ — relaxed sample "
                    f"= {float(val.flatten()[0]):.4f}, discrete sample "
                    f"= {float(val_disc.flatten()[0]):.4f}."
                )
            except Exception as e:  # pragma: no cover - surfaced in UI
                st.error(f"loss_fn raised: {e}")
        return
    st.info("No preview available for this problem type.")
