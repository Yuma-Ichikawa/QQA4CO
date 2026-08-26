"""Theme, CSS, and Plotly styling for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

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


def retheme_plotly(fig):
    """Reskin a Plotly figure so its palette matches the active theme.

    Every page mirrors this pattern — wrap any ``viz`` figure in
    ``retheme_plotly(...)`` before ``st.plotly_chart``. Errors from older
    figure objects that don't expose ``update_layout`` are intentionally
    swallowed so a single bad chart never breaks the whole page.
    """
    import contextlib  # noqa: PLC0415 - keep optional import local

    with contextlib.suppress(Exception):
        fig.update_layout(**plotly_layout())
    return fig


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
        /* Hide Streamlit's entire auto-generated multipage navigator.
           We render our own labelled nav (Problem → Solve → Visualize
           → Compare) inside ``sidebar_brand()`` so the workflow order
           reads naturally and the entry page has a sensible label. */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
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
            background:
              radial-gradient(120% 150% at 0% 0%, rgba(56,189,248,0.10) 0%, transparent 60%),
              linear-gradient(135deg, rgba(15,118,110,0.06), rgba(30,58,138,0.06));
            border: 1px solid var(--qqa-accent);
            border-left: 5px solid var(--qqa-accent);
            padding: 1.1rem 1.4rem;
            border-radius: 14px;
            margin: 0.6rem 0 1rem 0;
            box-shadow: 0 6px 24px -12px rgba(56,189,248,0.32),
                        0 1px 2px rgba(15,23,42,0.06);
        }}
        .qqa-score .label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--qqa-muted);
        }}
        .qqa-score .value {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 2.4rem;
            font-weight: 700;
            color: var(--qqa-text);
            line-height: 1.1;
            letter-spacing: -0.01em;
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
            font-variant-numeric: tabular-nums;
        }}
        .qqa-score .raw.polish {{
            color: #047857;
            font-weight: 500;
            margin-top: 0.45rem;
        }}
        .qqa-score .raw .muted {{ color: var(--qqa-muted); font-weight: 400; }}
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
        /* Same auto-nav suppression as in light theme. */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
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
            background:
              radial-gradient(120% 150% at 0% 0%, rgba(56,189,248,0.18) 0%, transparent 60%),
              linear-gradient(135deg, rgba(56,189,248,0.14), rgba(168,85,247,0.14));
            border: 1px solid rgba(56,189,248,0.4);
            border-left: 5px solid var(--qqa-accent);
            padding: 1.1rem 1.4rem;
            border-radius: 14px;
            margin: 0.6rem 0 1rem 0;
            box-shadow: 0 14px 36px -16px rgba(56,189,248,0.50),
                        0 1px 2px rgba(0,0,0,0.30);
        }}
        .qqa-score .label {{
            font-size: 0.72rem; letter-spacing: 0.12em;
            text-transform: uppercase; color: var(--qqa-muted);
        }}
        .qqa-score .value {{
            font-family: 'Source Serif 4', serif;
            font-size: 2.4rem; font-weight: 700; color: #f8fafc;
            letter-spacing: -0.01em;
        }}
        .qqa-score .value.infeasible {{ color: #fcd34d; }}
        .qqa-score .unit {{
            font-size: 1rem; color: var(--qqa-muted);
            font-weight: 500; margin-left: 0.4rem;
        }}
        .qqa-score .raw {{
            color: var(--qqa-muted); font-size: 0.85rem; margin-top: 0.3rem;
            font-variant-numeric: tabular-nums;
        }}
        .qqa-score .raw.polish {{
            color: #6ee7b7;
            font-weight: 500;
            margin-top: 0.45rem;
        }}
        .qqa-score .raw .muted {{ color: var(--qqa-muted); font-weight: 400; }}
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
