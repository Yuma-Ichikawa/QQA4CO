"""Shared helpers for the QQA Streamlit app.

Keeps problem construction, previews, and theming in one place so the Home /
Solve / Visualize / Compare pages stay declarative.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import streamlit as st
from ui.theme import apply_theme as apply_theme  # noqa: F401
from ui.theme import (
    get_theme,
    hex_to_rgba,
    palette,
)
from ui.theme import plotly_layout as plotly_layout  # noqa: F401
from ui.theme import retheme_plotly as retheme_plotly  # noqa: F401
from ui.theme import theme_toggle_in_sidebar as theme_toggle_in_sidebar  # noqa: F401

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
# Sidebar brand + paper-link footer (used by every page)
# ---------------------------------------------------------------------------

# URLs surfaced in the UI. Keep in sync with [project.urls] in pyproject.toml.
_PAPER_URL = "https://openreview.net/forum?id=9EfBeXaXf0"
_GITHUB_URL = "https://github.com/Yuma-Ichikawa/QQA4CO"
_DEMO_URL = "https://parallelquasiquantum4co.streamlit.app/"
_DOI_URL = "https://doi.org/10.5281/zenodo.19648231"


def _page_link_target(target: str) -> str:
    """Return a page path relative to Streamlit's current script.

    Streamlit resolves ``st.page_link`` from ``main_script_path``.  On a
    legacy multipage app that path changes from ``streamlit_app.py`` to the
    selected file below ``pages/``, so a single hard-coded relative path
    cannot work on every page.  Resolve from the live script while keeping a
    public-API-only fallback for older/newer Streamlit releases.
    """
    app_root = Path(__file__).resolve().parent
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        context = get_script_run_ctx(suppress_warning=True)
        script_path = getattr(context, "main_script_path", None)
        if script_path:
            return os.path.relpath(app_root / target, Path(script_path).resolve().parent)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        pass
    return target


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
        # We hide Streamlit's auto-generated multipage navigator with
        # CSS (see ``apply_theme``) and replace it with a manual
        # ``page_link`` block here. Two reasons:
        #   1. The auto navigator labels the entry page by file basename
        #      ("streamlit app"), which is jargony.
        #   2. The auto navigator places itself *above* the brand
        #      block, so the entry-page link sits below the
        #      sub-pages — confusing because the entry page is
        #      conceptually the *first* step (problem selection).
        # The manual block lives directly under the brand and lists
        # pages in the natural left-to-right workflow order:
        # **Problem → Solve → Visualize → Compare → Universal**.
        if hasattr(st, "page_link"):
            with contextlib.suppress(Exception):
                st.markdown(
                    "<div class='qqa-nav' style='margin-top:0.25rem;margin-bottom:0.5rem;'></div>",
                    unsafe_allow_html=True,
                )
            # A missing optional page in an older installed wheel must not
            # suppress every link that follows it. Resolve each entry
            # independently so mixed-version deployments remain navigable.
            links = (
                ("streamlit_app.py", "Problem", "🧩"),
                ("pages/1_Solve.py", "Solve", "▶️"),
                ("pages/2_Visualize.py", "Visualize", "📊"),
                ("pages/3_Compare.py", "Compare", "🔬"),
                ("pages/4_Universal.py", "Universal", "🌐"),
            )
            for page, label, icon in links:
                with contextlib.suppress(Exception):
                    st.page_link(_page_link_target(page), label=label, icon=icon)
            with contextlib.suppress(Exception):
                st.markdown(
                    "<div style='border-bottom:1px solid rgba(148,163,184,0.2);"
                    "margin:0.55rem 0 0.4rem 0;'></div>",
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
            st.page_link(_page_link_target(cta_page), label=cta_label, icon="▶")


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
                <a href="{_DOI_URL}" target="_blank" style="color:{link};
                   text-decoration:none;">🆔  Cite (Zenodo DOI)</a>
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


def render_config_chips(
    cfg: dict,
    *,
    extras: dict | None = None,
) -> None:
    """Render the active problem configuration as a row of chip-style badges.

    A more polished replacement for the bare ``problem: x | size: y | ...``
    caption used at the top of the Solve / Visualize / Compare pages.

    Args:
        cfg: Problem configuration dict, must contain ``kind`` / ``size`` /
            ``device`` / ``seed``.
        extras: Optional ``{label: value}`` dict appended after the standard
            chips (e.g. ``{"polish": "on", "warm-start": "off"}``).
    """
    p = palette()
    accent = p["accent"]
    border = p["border"]
    muted = p["muted"]
    chip_style = (
        "display:inline-flex;align-items:center;gap:0.35rem;"
        f"padding:0.22rem 0.7rem;border:1px solid {border};"
        "background:rgba(148,163,184,0.10);border-radius:999px;"
        "font-size:0.78rem;font-variant-numeric:tabular-nums;"
        "white-space:nowrap;"
    )
    label_style = f"color:{muted};text-transform:uppercase;letter-spacing:0.06em;font-size:0.66rem;"
    val_style = f"color:{accent};font-weight:600;"

    def _chip(label: str, value: object) -> str:
        return (
            f'<span style="{chip_style}">'
            f'<span style="{label_style}">{label}</span>'
            f'<span style="{val_style}">{value}</span>'
            f"</span>"
        )

    chips = [
        _chip("problem", cfg.get("kind", "?")),
        _chip("size", cfg.get("size", "?")),
        _chip("device", cfg.get("device", "?")),
        _chip("seed", cfg.get("seed", "?")),
    ]
    if extras:
        for k, v in extras.items():
            chips.append(_chip(k, v))
    st.markdown(
        '<div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin:0.2rem 0 0.9rem 0;">'
        + "".join(chips)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_score_card(
    score: dict,
    raw_loss: float | None = None,
    **_extra: object,
) -> None:
    """Render the big problem-specific score tile used by the Solve page.

    Parameters
    ----------
    score:
        Output of ``problem.score_summary``. The dict may carry an extra
        ``pre_polish_loss`` key — when present *and* strictly worse than
        ``raw_loss`` we surface a small "before polish" badge so users
        can see how much :func:`qqa.polish.greedy_one_flip` contributed.
    raw_loss:
        Optional raw ``loss_fn`` value (after polish, since ``anneal``
        replaces ``best_obj`` with the polished value).
    **_extra:
        Forward-compatible: silently absorbs any future kwargs from
        callers so an out-of-sync deployment never crashes the page
        with ``TypeError: got an unexpected keyword argument``.
    """
    if not isinstance(score, dict) or not score:
        return
    feas = bool(score.get("feasible", True))
    badge = (
        '<span class="qqa-badge ok">feasible</span>'
        if feas
        else '<span class="qqa-badge warn">infeasible</span>'
    )
    value = score.get("value", "-")
    value_s = f"{value:.4g}" if isinstance(value, int | float) else str(value)
    unit = score.get("unit", "")
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    raw_html = ""
    if isinstance(raw_loss, int | float):
        raw_html = f'<div class="raw">raw loss = {float(raw_loss):.4g}</div>'
    polish_html = ""
    pre_polish = score.get("pre_polish_loss")
    if (
        isinstance(pre_polish, int | float)
        and isinstance(raw_loss, int | float)
        # Only surface the line when polish actually moved the needle.
        and float(pre_polish) > float(raw_loss) + 1e-9
    ):
        delta = float(pre_polish) - float(raw_loss)
        polish_html = (
            f'<div class="raw polish">▴ polish improved by '
            f"<b>{delta:.4g}</b> "
            f'<span class="muted">(pre-polish = {float(pre_polish):.4g})</span></div>'
        )
    value_cls = "value" if feas else "value infeasible"
    st.markdown(
        f'<div class="qqa-score">'
        f'<div class="label">{score.get("label", "score")} · {badge}</div>'
        f'<div class="{value_cls}">{value_s}{unit_html}</div>'
        f"{raw_html}"
        f"{polish_html}"
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
    _validate_problem_extra(kind, extra)

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
    if kind in {
        "mis",
        "maxcut",
        "maxclique",
        "coloring",
        "vertex_cover",
        "graph_bisection",
        "min_dominating_set",
        "bgp",
    }:
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
    if kind == "pspin":
        return _safe_call(
            _require(qqa, "PSpinGlass"),
            N=size,
            p=int(extra.get("p_order", 3)),
            seed=seed,
            device=device,
        )
    if kind == "rfim":
        return _safe_call(
            _require(qqa, "RandomFieldIsing"),
            L=size,
            dim=int(extra.get("dim", 2)),
            J=float(extra.get("coupling_J", 1.0)),
            h_std=float(extra.get("h_std", 1.0)),
            seed=seed,
            device=device,
        )
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
        penalties = _extract_penalty_kwargs(
            extra,
            defaults={"row_penalty": 5.0, "col_penalty": 5.0},
        )
        return _safe_call(
            qqa.TSP,
            N=size,
            seed=seed,
            device=device,
            relaxation=extra.get("relaxation", "sinkhorn"),
            **penalties,
        )
    if kind == "qap":
        return _safe_call(
            qqa.QAP,
            N=size,
            seed=seed,
            device=device,
            **_extract_penalty_kwargs(
                extra,
                defaults={"column_penalty": 10.0},
                aliases={},
            ),
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
    if kind == "min_dominating_set":
        return _safe_call(_require(qqa, "MinimumDominatingSet"), g, device=device)
    if kind == "bgp":
        return _safe_call(
            _require(qqa, "BalancedGraphPartition"),
            g,
            num_category=int(extra.get("num_category", 3)),
            penalty=float(extra.get("balance_penalty", 5e-4)),
            device=device,
        )
    raise ValueError(f"Unknown graph-problem kind {kind!r}")


# ---------------------------------------------------------------------------
# Constructor dispatch helpers — keep ``build_problem`` and the saved-config
# format decoupled from individual class signatures.
# ---------------------------------------------------------------------------

_PROBLEM_EXTRA_KEYS: dict[str, frozenset[str]] = {
    "custom": frozenset({"source", "num_vars", "variable_kind", "num_category", "name"}),
    "mis": frozenset({"graph_d"}),
    "maxcut": frozenset({"graph_d"}),
    "maxclique": frozenset({"graph_d"}),
    "vertex_cover": frozenset({"graph_d"}),
    "min_dominating_set": frozenset({"graph_d"}),
    "coloring": frozenset({"graph_d", "num_category"}),
    "graph_bisection": frozenset({"graph_d", "balance_penalty"}),
    "bgp": frozenset({"graph_d", "num_category", "balance_penalty"}),
    "ising1d": frozenset(),
    "ea": frozenset({"dim"}),
    "sk": frozenset(),
    "pspin": frozenset({"p_order"}),
    "rfim": frozenset({"dim", "coupling_J", "h_std"}),
    "perceptron": frozenset({"alpha"}),
    "hopfield": frozenset({"patterns"}),
    "knapsack": frozenset({"capacity_ratio"}),
    "number_partition": frozenset({"max_value"}),
    "maxsat3": frozenset({"ratio"}),
    "tsp": frozenset(
        {"relaxation", "row_penalty", "col_penalty", "column_penalty", "penalty_weights"}
    ),
    "qap": frozenset({"column_penalty", "penalty_weights"}),
    "nqueens": frozenset(),
}


def _validate_problem_extra(kind: str, extra: dict) -> None:
    """Reject displayed settings that are not consumed by the selected model."""
    if not isinstance(extra, dict):
        raise TypeError("problem_config.extra must be a mapping.")
    allowed = _PROBLEM_EXTRA_KEYS.get(kind)
    if allowed is None:
        return
    unknown = sorted(set(extra) - allowed)
    if unknown:
        raise TypeError(f"Unknown {kind} option(s): {', '.join(unknown)}")


def _require(module: object, attr: str):
    """Return ``getattr(module, attr)`` or raise a friendly ``RuntimeError``.

    The deployed qqa version on Streamlit Cloud sometimes lags behind the
    catalog the UI advertises. When that happens we want a clear, actionable
    message at Run-time instead of an opaque ``AttributeError``.
    """

    obj = getattr(module, attr, None)
    if obj is None:
        version = getattr(module, "__version__", "unknown")
        name = getattr(module, "__name__", "module")
        raise RuntimeError(
            f"{attr!r} is not available in the installed {name} {version}. "
            f"Run `pip install -U {name}` to enable it."
        )
    return obj


def _safe_call(cls, *args, **kwargs):
    """Invoke a constructor after strict keyword-schema validation.

    Persisted UI state is still migrated by the explicit alias layer below,
    but genuinely unknown options are errors. Silently discarding an option
    makes the displayed configuration differ from the model that is solved.
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
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise TypeError(f"Unknown {cls.__name__} option(s): {', '.join(unknown)}")
    return cls(*args, **kwargs)


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


def _extract_penalty_kwargs(
    extra: dict,
    *,
    defaults: dict[str, float],
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, float]:
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
    aliases = _PENALTY_ALIASES if aliases is None else aliases
    out: dict[str, float] = dict(defaults)
    legacy_targets: set[str] = set()
    for modern_keys in aliases.values():
        legacy_targets.update(modern_keys)

    # 2. legacy aliases (translate, do not propagate the legacy key itself).
    for legacy, modern_keys in aliases.items():
        if legacy in extra:
            try:
                v = float(extra[legacy])
            except (TypeError, ValueError):
                continue
            for k in modern_keys:
                out[k] = v

    # 3. explicit penalty-shaped keys override legacy translations.
    for k, v in extra.items():
        if k in aliases:
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
    p = palette()
    is_light = get_theme() == "light"
    edge_color = hex_to_rgba("#475569" if is_light else "#94a3b8", 0.55)
    node_outline = "#0a4d48" if is_light else "#0ea5e9"

    # Map node IDs to a contiguous integer index so spring_layout works
    # even on graphs with non-numeric or non-contiguous labels.
    pos = nx.spring_layout(g, seed=0)
    edge_x, edge_y = [], []
    for u, v in g.edges:
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])
    node_x = [pos[n][0] for n in g.nodes]
    node_y = [pos[n][1] for n in g.nodes]
    degrees = [g.degree(n) for n in g.nodes]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": edge_color, "width": 1},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker={
                "color": degrees,
                "colorscale": "Mint" if is_light else "Tealgrn",
                "size": 11,
                "line": {"color": node_outline, "width": 1.2},
                "colorbar": {"title": "deg", "thickness": 10, "len": 0.6},
            },
            customdata=list(g.nodes),
            hovertemplate="node %{customdata}<br>degree %{marker.color}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"color": p["text"]}},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": p["text"]},
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=400,
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Spring layout · |V| = {g.number_of_nodes()}, "
        f"|E| = {g.number_of_edges()} · node colour = degree"
    )


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
        p = palette()
        fig = go.Figure(
            data=go.Heatmap(
                z=J,
                colorscale="RdBu",
                zmid=0,
                colorbar={"title": "J_ij", "thickness": 12},
                hovertemplate="J(%{y}, %{x}) = %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center", "font": {"color": p["text"]}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": p["text"]},
            height=420,
            margin={"l": 50, "r": 30, "t": 60, "b": 40},
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
    p = palette()
    fig.update_layout(
        title={
            "text": f"{title} — sparse view ({rows.size} non-zero entries)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": p["text"]},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": p["text"]},
        height=440,
        margin={"l": 50, "r": 30, "t": 60, "b": 40},
        xaxis={"title": "j", "scaleanchor": "y", "scaleratio": 1, "autorange": True},
        yaxis={"title": "i", "autorange": "reversed"},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Showing {rows.size} non-zero couplings out of {N * N:,} matrix entries. "
        "A dense heatmap at this size would crash the browser."
    )


# ---------------------------------------------------------------------------
# Per-problem preview registry
# ---------------------------------------------------------------------------
#
# Every problem family supported by the dashboard now has *something* to
# display before a run, even when there is no natural "graph" or
# "coupling matrix" to draw. The dispatch order is:
#
#   1. ``nx_graph`` attribute (graph problems)
#   2. ``J`` attribute (Ising-type problems)
#   3. dedicated per-kind preview (TSP, QAP, Knapsack, …)
#   4. ``custom`` problem editor preview
#   5. concept card (formula + 1-line description) — *never* the bare
#      "no preview available" message that used to land here.
#
# Each preview emits one or more ``st.plotly_chart`` calls plus a
# caption explaining what the user is looking at.


# Concept cards — formula + plain-language description for any problem
# whose preview we cannot draw from instance data alone (e.g. NQueens,
# user-typed problem with a brand-new ``kind`` string).
_PROBLEM_CONCEPTS: dict[str, dict[str, str]] = {
    "tsp": {
        "name": "Travelling Salesman Problem",
        "formula": (r"\min_{\pi \in S_N}\; \sum_{t=0}^{N-1} d\bigl(\pi(t),\pi(t+1)\bigr)"),
        "desc": (
            "Find the shortest closed tour that visits every city exactly "
            "once. We solve it with the **penalty method** on an "
            "$N\\times N$ permutation matrix $x_{t,i}$."
        ),
    },
    "qap": {
        "name": "Quadratic Assignment Problem",
        "formula": r"\min_{\pi}\; \sum_{i,j} F_{ij}\, D_{\pi(i),\pi(j)}",
        "desc": (
            "Assign $N$ facilities to $N$ locations. Cost = flow $F$ "
            "between facilities times distance $D$ between locations."
        ),
    },
    "nqueens": {
        "name": "$N$-Queens",
        "formula": (
            r"\text{place } N \text{ queens on an } N\!\times\!N "
            r"\text{ board, no two attacking}"
        ),
        "desc": (
            "Permutation problem: column $i$ gets exactly one queen at "
            "row $\\pi(i)$, and no two queens share a diagonal."
        ),
    },
    "knapsack": {
        "name": "0/1 Knapsack",
        "formula": (
            r"\max_{x\in\{0,1\}^N}\; \sum_i v_i x_i "
            r"\;\;\text{s.t.}\;\; \sum_i w_i x_i \le C"
        ),
        "desc": (
            "Pick a subset of $N$ items maximising total value while "
            "keeping total weight under capacity $C$."
        ),
    },
    "number_partition": {
        "name": "Number Partitioning",
        "formula": (r"\min_{x\in\{\pm1\}^N}\; \Bigl(\sum_i a_i x_i\Bigr)^2"),
        "desc": (
            "Split a multiset $\\{a_1,\\dots,a_N\\}$ of positive numbers "
            "into two subsets whose sums are as equal as possible."
        ),
    },
    "maxsat3": {
        "name": "MAX-3-SAT",
        "formula": (r"\max_{x\in\{0,1\}^N}\; \#\{\text{satisfied 3-CNF clauses}\}"),
        "desc": (
            "Random 3-CNF instance with $\\alpha N$ clauses. The phase "
            "transition sits near $\\alpha\\approx 4.27$."
        ),
    },
    "hopfield": {
        "name": "Hopfield Memory",
        "formula": (
            r"H(x) = -\tfrac12 \sum_{ij} J_{ij} x_i x_j,\;\;"
            r"J_{ij} = \tfrac{1}{P}\sum_\mu \xi^\mu_i \xi^\mu_j"
        ),
        "desc": ("Retrieve one of $P$ stored binary patterns by minimising the Hopfield energy."),
    },
    "perceptron": {
        "name": "Binary Perceptron",
        "formula": (
            r"\text{find } x\in\{\pm1\}^N \text{ s.t. } "
            r"\langle x, \xi^\mu\rangle \ge 0\ \forall \mu"
        ),
        "desc": (
            "Find a binary weight vector that classifies $\\alpha N$ random patterns correctly."
        ),
    },
    "ising1d": {
        "name": "1-D Ising Chain",
        "formula": (r"H(s) = -\sum_{i} J\, s_i s_{i+1}\;\; (s_i \in \{\pm 1\})"),
        "desc": "Periodic ferromagnetic chain — the textbook spin model.",
    },
    "ea": {
        "name": "Edwards–Anderson Spin Glass",
        "formula": (
            r"H(s) = -\sum_{\langle i,j \rangle} J_{ij} s_i s_j,\;\;"
            r"J_{ij}\sim\mathcal{N}(0,1)"
        ),
        "desc": (
            "$d$-dimensional hyper-cubic lattice with i.i.d. Gaussian "
            "couplings. NP-hard ground-state problem."
        ),
    },
    "sk": {
        "name": "Sherrington–Kirkpatrick Spin Glass",
        "formula": (
            r"H(s) = -\tfrac{1}{\sqrt{N}}\!\sum_{i<j} J_{ij} s_i s_j,\;\;"
            r"J_{ij}\!\sim\!\mathcal N(0,1)"
        ),
        "desc": "Mean-field spin glass — every pair of spins interacts.",
    },
}


def _concept_card(kind: str, *, problem: Any | None = None) -> None:
    """Render a formula + description card for a problem whose instance
    data has no natural visual representation. This is the *fallback*
    view but it is always informative — never the bare "no preview"
    message of old."""
    info = _PROBLEM_CONCEPTS.get(
        kind,
        {
            "name": kind.replace("_", " ").title(),
            "formula": "",
            "desc": "User-defined combinatorial optimisation problem.",
        },
    )
    p = palette()
    muted = "#64748b" if get_theme() == "light" else "#94a3b8"
    accent = "#0f766e" if get_theme() == "light" else "#38bdf8"

    st.markdown(
        f"""
        <div style="
            border:1px solid {p["border"]};
            background:{p["bg_card"]};
            padding:1.2rem 1.4rem;
            border-radius:12px;
            box-shadow:0 1px 3px rgba(15,23,42,0.05);
        ">
          <div style="
              display:flex;align-items:center;gap:0.5rem;
              font-family:'Source Serif 4',Georgia,serif;
              font-weight:600;font-size:1.1rem;color:{p["text"]};
              margin-bottom:0.2rem;
          ">
            <span style="
                display:inline-block;width:6px;height:18px;border-radius:3px;
                background:{accent};
            "></span>
            {info["name"]}
          </div>
          <div style="font-size:0.85rem;color:{muted};margin-bottom:0.55rem;">
            Concept overview · instance details below
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if info["formula"]:
        st.latex(info["formula"])
    st.markdown(info["desc"])


def _figure_layout(title: str, *, height: int = 380, **extra) -> dict:
    """Common Plotly layout dict for previews — keeps every chart on the
    Home page typographically consistent."""
    return {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "height": height,
        "margin": {"l": 50, "r": 30, "t": 60, "b": 40},
        **extra,
    }


def _tsp_preview(problem: Any) -> None:
    coords = _as_np(problem.coords)
    distance = _as_np(problem.distance)
    p = palette()
    accent = "#0f766e" if get_theme() == "light" else "#38bdf8"
    cols = st.columns([3, 2])
    with cols[0]:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers+text",
                marker={
                    "size": 14,
                    "color": accent,
                    "line": {"color": p["border"], "width": 1.5},
                },
                text=[str(i) for i in range(len(coords))],
                textposition="top center",
                hovertemplate="city %{text}<br>(%{x:.2f}, %{y:.2f})<extra></extra>",
            )
        )
        fig.update_layout(
            **_figure_layout(
                f"City coordinates · N = {problem.N}",
                xaxis={"title": "x", "scaleanchor": "y"},
                yaxis={"title": "y"},
            )
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Cities are placed on the unit square; the optimiser searches for the shortest closed tour."
        )
    with cols[1]:
        fig = go.Figure(
            data=go.Heatmap(
                z=distance,
                colorscale="Mint",
                colorbar={"title": "d(i,j)", "thickness": 12},
                hovertemplate="d(%{y}, %{x}) = %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(**_figure_layout("Distance matrix", height=380))
        st.plotly_chart(fig, width="stretch")


def _qap_preview(problem: Any) -> None:
    F = _as_np(problem.F)
    D = _as_np(problem.D)
    cols = st.columns(2)
    with cols[0]:
        fig = go.Figure(
            data=go.Heatmap(
                z=F,
                colorscale="Sunsetdark",
                colorbar={"title": "F", "thickness": 12},
                hovertemplate="F(%{y},%{x}) = %{z:.2f}<extra></extra>",
            )
        )
        fig.update_layout(**_figure_layout(f"Flow F · {F.shape[0]}×{F.shape[1]}"))
        st.plotly_chart(fig, width="stretch")
        st.caption("How much material moves between facility pairs.")
    with cols[1]:
        fig = go.Figure(
            data=go.Heatmap(
                z=D,
                colorscale="Mint",
                colorbar={"title": "D", "thickness": 12},
                hovertemplate="D(%{y},%{x}) = %{z:.2f}<extra></extra>",
            )
        )
        fig.update_layout(**_figure_layout(f"Distance D · {D.shape[0]}×{D.shape[1]}"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Distance between every pair of locations.")


def _knapsack_preview(problem: Any) -> None:
    weights = _as_np(problem.weights)
    values = _as_np(problem.values)
    capacity = float(problem.capacity)
    p = palette()
    accent_v = "#0f766e" if get_theme() == "light" else "#38bdf8"
    accent_w = "#be5a3c" if get_theme() == "light" else "#a855f7"
    idx = np.argsort(-(values / np.maximum(weights, 1e-9)))  # by value-density
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"item {i}" for i in idx],
            y=values[idx],
            name="value",
            marker_color=accent_v,
            hovertemplate="item %{x}<br>value=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[f"item {i}" for i in idx],
            y=weights[idx],
            name="weight",
            marker_color=accent_w,
            opacity=0.85,
            hovertemplate="item %{x}<br>weight=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=capacity,
        line={"color": p["text"], "dash": "dash", "width": 1.5},
        annotation_text=f"capacity = {capacity:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        **_figure_layout(
            f"Knapsack instance · N = {problem.N}, total weight = {weights.sum():.1f}, "
            f"capacity = {capacity:.1f}",
            barmode="group",
            xaxis={"title": "items (sorted by value-density)"},
            yaxis={"title": "weight / value"},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        )
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Bars compare each item's weight against its value; the dashed line is the total weight budget."
    )


def _number_partition_preview(problem: Any) -> None:
    values = _as_np(problem.values).astype(float)
    target = values.sum() / 2.0
    accent = "#0f766e" if get_theme() == "light" else "#38bdf8"
    fig = go.Figure(
        data=go.Bar(
            x=[f"a_{i}" for i in range(len(values))],
            y=values,
            marker_color=accent,
            hovertemplate="a_%{x} = %{y}<extra></extra>",
        )
    )
    fig.add_hline(
        y=target,
        line={"dash": "dash", "color": "#94a3b8"},
        annotation_text=f"target sum / 2 = {target:.1f}",
        annotation_position="top right",
    )
    fig.update_layout(
        **_figure_layout(
            f"Number-partitioning instance · N = {len(values)}, total = {values.sum():.0f}",
            xaxis={"title": "items"},
            yaxis={"title": "value"},
        )
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Goal: find ±1 signs whose signed sum is as close to zero as possible.")


def _maxsat3_preview(problem: Any) -> None:
    cv = _as_np(problem.clause_vars)
    cs = _as_np(problem.clause_signs)
    M = problem.num_clauses
    N = problem.N
    n_show = min(12, M)
    rows = []
    for c in range(n_show):
        lits = []
        for k in range(3):
            v = int(cv[c, k])
            sign = int(cs[c, k])
            lits.append(f"{'¬' if sign < 0 else ''}x{v}")
        rows.append("(" + " ∨ ".join(lits) + ")")
    body = " ∧ ".join(rows)
    if n_show < M:
        body += f" ∧ … ({M - n_show} more clauses)"
    p = palette()
    st.markdown(
        f"""
        <div style="
            border:1px solid {p["border"]};
            background:{p["bg_card"]};
            padding:1rem 1.2rem;border-radius:10px;
            font-family:'JetBrains Mono', 'SF Mono', Consolas, monospace;
            font-size:0.85rem;line-height:1.6;color:{p["text"]};
            white-space:pre-wrap;word-break:break-word;
        ">{body}</div>
        """,
        unsafe_allow_html=True,
    )
    # Variable-incidence histogram — a cheap structure indicator that
    # tells the user whether the instance is balanced.
    var_freq = np.bincount(cv.reshape(-1), minlength=N)
    fig = go.Figure(
        data=go.Bar(
            x=np.arange(N),
            y=var_freq,
            marker_color="#0f766e" if get_theme() == "light" else "#38bdf8",
            hovertemplate="x_%{x} appears in %{y} clauses<extra></extra>",
        )
    )
    fig.update_layout(
        **_figure_layout(
            f"Variable incidence · N = {N}, M = {M}, ratio α ≈ {M / N:.2f}",
            xaxis={"title": "variable index"},
            yaxis={"title": "appearances"},
            height=300,
        )
    )
    st.plotly_chart(fig, width="stretch")


def _nqueens_preview(problem: Any) -> None:
    N = int(problem.N)
    light = "#f1f5f9" if get_theme() == "light" else "#1e293b"
    dark = "#cbd5e1" if get_theme() == "light" else "#334155"
    z = np.indices((N, N)).sum(axis=0) % 2
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            colorscale=[[0, light], [1, dark]],
            showscale=False,
            hoverinfo="skip",
        )
    )
    # Decorative queens on the perimeter to hint at "place N queens".
    fig.add_trace(
        go.Scatter(
            x=[-1] * N,
            y=list(range(N)),
            mode="text",
            text=["♛"] * N,
            textfont={"size": 22, "color": "#0f766e" if get_theme() == "light" else "#38bdf8"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        **_figure_layout(
            f"{N}×{N} chessboard — place {N} non-attacking queens",
            height=420,
            xaxis={"visible": False, "range": [-2, N - 0.5], "scaleanchor": "y"},
            yaxis={"visible": False, "range": [-0.5, N - 0.5]},
        )
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Search space: {N}! permutations — even N=12 already has half a billion candidates."
    )


def _hopfield_preview(problem: Any) -> None:
    patterns = _as_np(problem.patterns)
    P, N = patterns.shape
    fig = go.Figure(
        data=go.Heatmap(
            z=patterns,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "ξ", "thickness": 12},
            hovertemplate="pattern %{y}, spin %{x} = %{z:+.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_figure_layout(
            f"Stored patterns · {P} patterns × {N} spins",
            xaxis={"title": "spin index"},
            yaxis={"title": "pattern index"},
        )
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Hopfield retrieves one of these patterns by minimising its associative-memory energy."
    )


def as_numpy(x) -> np.ndarray:
    """Return a CPU ``numpy`` view of a torch tensor / array / scalar.

    Shared across the app (problem previews, solution viz). Lives here
    rather than at each import site so behaviour stays consistent if we
    ever need to e.g. detach non-leaf autograd graphs or handle BF16.
    """
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# Backwards-compatible alias — callers inside ``_common.py`` still use
# ``_as_np``. Keeping the alias avoids a large patch and preserves the
# "private helper" reading for module-internal call sites.
_as_np = as_numpy


_DEDICATED_PREVIEWS: dict[str, callable] = {
    "tsp": _tsp_preview,
    "qap": _qap_preview,
    "knapsack": _knapsack_preview,
    "number_partition": _number_partition_preview,
    "maxsat3": _maxsat3_preview,
    "nqueens": _nqueens_preview,
    "hopfield": _hopfield_preview,
}


def preview_problem(problem: Any, cfg: dict) -> None:
    kind = cfg["kind"]
    # Always lead with the concept card so the user sees *what* the
    # problem is, not just an instance dump. The instance-specific
    # plot follows below.
    _concept_card(kind, problem=problem)

    if hasattr(problem, "nx_graph"):
        _graph_preview(problem.nx_graph, f"{kind} graph (n={problem.num_nodes})")
        return
    if kind in _DEDICATED_PREVIEWS:
        try:
            _DEDICATED_PREVIEWS[kind](problem)
        except Exception as e:
            st.warning(f"Could not draw the {kind} instance preview: {e}")
        return
    if hasattr(problem, "J") and problem.J is not None:
        J = problem.J.detach().cpu().numpy()
        _coupling_preview(J, f"{kind} couplings (N={problem.num_spins})")
        return
    if kind == "perceptron":
        xi = problem.xi_signed.detach().cpu().numpy()
        fig = go.Figure(data=go.Heatmap(z=xi, colorscale="RdBu", zmid=0, colorbar={"title": "ξ̂"}))
        fig.update_layout(
            **_figure_layout(
                f"Signed patterns ({problem.num_patterns} × {problem.num_spins})",
                height=400,
            )
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
    # Catch-all: a brand-new ``kind`` that the registry doesn't know
    # about. The concept card was already rendered at the top of
    # ``preview_problem``; here we just surface a polite, branded note
    # rather than the old generic "no preview" message.
    p = palette()
    muted = "#64748b" if get_theme() == "light" else "#94a3b8"
    st.markdown(
        f"""
        <div style="
            border:1px dashed {p["border"]};padding:0.85rem 1rem;
            border-radius:10px;background:{p["bg_card"]};
            color:{muted};font-size:0.9rem;line-height:1.5;
        ">
          The instance-level visualisation for <code>{kind}</code> is
          not yet available, but the concept card above describes the
          problem and the solver still works as usual. Submit
          <em>Run QQA</em> in <b>Solve</b> to see the post-anneal
          solution view.
        </div>
        """,
        unsafe_allow_html=True,
    )
