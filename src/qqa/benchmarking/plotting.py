"""Polished benchmark-report renderer for the QQA4CO CO suite.

Turns any ``scripts/bench_discs.py --output results.json`` payload into
a single publication-quality image that communicates, at a glance,
*how a method performs across every benchmark family shipped with the
suite* — DISCS (``maxcut`` / ``mis`` / ``maxclique`` / ``normcut``),
``coloring``, ``mis-rrg``, ``ea3d``, ``balanced-partition``.

Layout
------

.. code-block:: text

    +-----------------------------------------------------------------+
    |   HEADER   run metadata (backend · suite · device · hp)         |
    +-----------------------------------------------------------------+
    |        |                                                        |
    | RADAR  |   KPI BAND (families · instances · feasibility · ApR)  |
    |        +--------------------------------------------------------+
    |        |   PER-SUBSET horizontal bars (sorted within family)    |
    +--------+--------------------------------------------------------+
    |  FEASIBILITY bars      |   PER-INSTANCE violin + strip           |
    +-----------------------------------------------------------------+
    |                     FOOTER  citation hint                       |
    +-----------------------------------------------------------------+

The palette is a consistent per-family mapping so that the same family
gets the same colour on every panel (radar spoke, bar chunk, violin
hue). A/B/C comparisons across methods get distinct hatching and a
legend above the chart so the family colours stay meaningful.

CLI
---
::

    python scripts/plot_benchmarks.py results.json --output bench.png
    python scripts/plot_benchmarks.py r1.json r2.json --labels baseline tuned \\
        --output cmp.png
    python scripts/plot_benchmarks.py results.json --format svg
    python scripts/plot_benchmarks.py results.json --theme dark

Only depends on matplotlib + numpy (already core QQA4CO deps).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# --------------------------------------------------------------------------- #
# Report normalisation                                                        #
# --------------------------------------------------------------------------- #


class SubsetRow(dict):
    """Dict-shaped row so tests can keep using ``row['feasibility']``."""

    __slots__ = ()

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def load_report(path: Path) -> dict[str, Any]:
    """Parse a ``bench_discs.py`` JSON payload into a plot-friendly shape."""
    with open(path) as fh:
        raw = json.load(fh)

    families: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"mean_ratio": math.nan, "subsets": [], "instance_ratios": []}
    )
    for agg in raw.get("results", []):
        fam = agg.get("problem", "?")
        key = _subset_key(agg)
        ratios = [r.get("ratio") for r in agg.get("instances", []) if r.get("ratio") is not None]
        subset_ratio = agg.get("mean_ratio")
        feas_rate = (
            float(agg.get("n_feasible", 0)) / float(agg.get("n", 1))
            if agg.get("n")
            else float("nan")
        )
        row = SubsetRow(
            family=fam,
            subset=key,
            mean_ratio=float(subset_ratio) if subset_ratio is not None else math.nan,
            feasibility=feas_rate,
            n=int(agg.get("n", 0)),
            instance_ratios=[float(x) for x in ratios],
        )
        families[fam]["subsets"].append(row)
        families[fam]["instance_ratios"].extend(row["instance_ratios"])

    for info in families.values():
        weights, values = [], []
        for s in info["subsets"]:
            if s["n"] > 0 and not math.isnan(s["mean_ratio"]):
                weights.append(s["n"])
                values.append(s["mean_ratio"])
        info["mean_ratio"] = float(np.average(values, weights=weights)) if values else math.nan

    return {
        "label": _default_label(raw),
        "backend": raw.get("backend", "?"),
        "suite": raw.get("suite", "?"),
        "device": raw.get("device", "?"),
        "qqa_hp": raw.get("qqa_hp", {}),
        "families": dict(families),
    }


def _subset_key(agg: dict[str, Any]) -> str:
    gt = agg.get("graph_type") or ""
    sub = agg.get("subset") or ""
    if gt and sub:
        return f"{gt}/{sub}"
    return gt or sub or "default"


def _default_label(raw: dict[str, Any]) -> str:
    backend = raw.get("backend", "?")
    suite = raw.get("suite", "?")
    return f"{backend}:{suite}"


# --------------------------------------------------------------------------- #
# Palette + family order                                                      #
# --------------------------------------------------------------------------- #

#: Stable per-family colour so the same family gets the same hue everywhere.
_FAMILY_COLORS = {
    "mis": "#1f77b4",
    "maxcut": "#2ca02c",
    "maxclique": "#17becf",
    "normcut": "#9467bd",
    "coloring": "#e377c2",
    "mis-rrg": "#ff7f0e",
    "ea3d": "#d62728",
    "balanced-partition": "#8c564b",
}
_FAMILY_ORDER = list(_FAMILY_COLORS)
_METHOD_HATCHES = ("", "///", "\\\\\\", "xxx", "...", "+++", "|||")

_THEMES = {
    "light": {
        "style": "seaborn-v0_8-whitegrid",
        "bg": "white",
        "fg": "#222222",
        "muted": "#555555",
        "panel": "#f6f7f9",
        "divider": "#d0d4da",
    },
    "dark": {
        "style": "dark_background",
        "bg": "#0f1115",
        "fg": "#f5f5f7",
        "muted": "#c0c4cc",
        "panel": "#1c1f26",
        "divider": "#333842",
    },
}


def _ordered_family_axes(reports: list[dict]) -> list[str]:
    present: set[str] = set()
    for r in reports:
        present.update(r["families"].keys())
    axes = [f for f in _FAMILY_ORDER if f in present]
    axes += sorted(present - set(_FAMILY_ORDER))
    return axes


def _family_color(fam: str) -> Any:
    if fam in _FAMILY_COLORS:
        return _FAMILY_COLORS[fam]
    return plt.get_cmap("tab10")(hash(fam) % 10)


# --------------------------------------------------------------------------- #
# Panels                                                                      #
# --------------------------------------------------------------------------- #


def _draw_header(fig, reports: list[dict], theme: dict, title: str | None) -> None:
    ax = fig.add_axes((0.0, 0.93, 1.0, 0.07))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    main = title if title is not None else "QQA4CO benchmark report"
    ax.text(
        0.015,
        0.72,
        main,
        fontsize=18,
        fontweight="bold",
        color=theme["fg"],
        va="center",
    )
    sub_bits = []
    for r in reports:
        hp = r.get("qqa_hp", {}) or {}
        bits = [
            f"backend={r.get('backend', '?')}",
            f"suite={r.get('suite', '?')}",
            f"device={r.get('device', '?')}",
        ]
        if hp:
            hp_str = ", ".join(f"{k}={hp[k]}" for k in sorted(hp) if hp[k] is not None)
            if len(hp_str) > 120:
                hp_str = hp_str[:117] + "…"
            bits.append(f"hp=[{hp_str}]")
        sub_bits.append(f"{r['label']}  —  " + "  ·  ".join(bits))
    ax.text(
        0.015,
        0.30,
        "\n".join(sub_bits),
        fontsize=8.8,
        color=theme["muted"],
        va="center",
    )
    ax.plot([0.01, 0.99], [0.01, 0.01], color=theme["divider"], linewidth=1.0, clip_on=False)


def _kpi_band(
    fig, box: tuple[float, float, float, float], reports: list[dict], theme: dict
) -> None:
    ax = fig.add_axes(box)
    ax.set_axis_off()

    # Gather stats for the *first* report (primary method); show
    # deltas against other reports in smaller font.
    primary = reports[0]
    total_inst = sum(s.n for info in primary["families"].values() for s in info["subsets"])
    total_feas = 0
    feas_den = 0
    ratios: list[float] = []
    families_with_ratio = 0
    for info in primary["families"].values():
        for s in info["subsets"]:
            if not math.isnan(s.feasibility):
                total_feas += s.feasibility * s.n
                feas_den += s.n
            if not math.isnan(s.mean_ratio):
                ratios.extend([s.mean_ratio] * max(s.n, 1))
        if not math.isnan(info["mean_ratio"]):
            families_with_ratio += 1

    feas_pct = (total_feas / feas_den * 100.0) if feas_den else float("nan")
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")

    kpis = [
        ("families", f"{len(primary['families'])}", f"{families_with_ratio} with ApR"),
        ("instances", f"{total_inst}", f"{feas_den} with feasibility"),
        ("mean ApR", _fmt_ratio(mean_ratio), "weighted by n"),
        ("feasible", _fmt_pct(feas_pct), f"across {feas_den} replicas"),
    ]
    n = len(kpis)
    pad = 0.01
    card_w = (1.0 - pad * (n + 1)) / n
    card_h = 0.88
    for i, (label, value, sub) in enumerate(kpis):
        x0 = pad + i * (card_w + pad)
        y0 = 0.05
        box_patch = FancyBboxPatch(
            (x0, y0),
            card_w,
            card_h,
            boxstyle="round,pad=0.01,rounding_size=0.035",
            linewidth=0.8,
            edgecolor=theme["divider"],
            facecolor=theme["panel"],
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_artist(box_patch)
        ax.text(
            x0 + 0.015,
            y0 + card_h - 0.14,
            label.upper(),
            transform=ax.transAxes,
            fontsize=8.5,
            color=theme["muted"],
            fontweight="bold",
        )
        ax.text(
            x0 + card_w / 2,
            y0 + card_h / 2 - 0.02,
            value,
            transform=ax.transAxes,
            fontsize=22,
            color=theme["fg"],
            fontweight="bold",
            ha="center",
            va="center",
        )
        ax.text(
            x0 + card_w / 2,
            y0 + 0.11,
            sub,
            transform=ax.transAxes,
            fontsize=7.5,
            color=theme["muted"],
            ha="center",
            va="center",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _fmt_ratio(x: float) -> str:
    if math.isnan(x):
        return "—"
    return f"{x:.3f}"


def _fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "—"
    return f"{x:.1f}%"


def _panel_radar(ax, reports: list[dict], theme: dict) -> None:
    families = _ordered_family_axes(reports)
    if not families:
        _empty(ax, "no data", theme)
        return

    theta = np.linspace(0, 2 * np.pi, len(families), endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])

    # Background grid with concentric reference ring at 1.0.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for level, alpha in ((0.25, 0.25), (0.5, 0.35), (0.75, 0.45), (1.0, 0.6)):
        ax.plot(
            np.linspace(0, 2 * np.pi, 200),
            [level] * 200,
            color=theme["divider"],
            linewidth=0.8 if level != 1.0 else 1.2,
            linestyle="-" if level == 1.0 else "--",
            alpha=alpha,
        )

    for method_idx, r in enumerate(reports):
        vals = [r["families"].get(f, {}).get("mean_ratio", math.nan) for f in families]
        plot_vals = [0.0 if (v is None or math.isnan(v)) else max(0.0, min(1.6, v)) for v in vals]
        plot_vals.append(plot_vals[0])
        color = _method_color(method_idx, theme)
        ax.plot(
            theta_closed,
            plot_vals,
            linewidth=2.1,
            label=r["label"],
            color=color,
            marker="o",
            markersize=4.5,
        )
        ax.fill(theta_closed, plot_vals, alpha=0.15, color=color)

    # Axis cosmetics.
    ax.set_xticks(theta)
    ax.set_xticklabels([])
    # Slightly inflate the radial range so the family labels sit just
    # outside the plotted polygons, not on top of them.
    rmax = max(1.1, *_flatten_mean_ratios(reports, 1.0))
    ax.set_ylim(0, rmax * 1.12)
    # Show family labels with their family colour so the reader can
    # cross-reference with per-subset / feasibility panels.
    for ang, fam in zip(theta, families, strict=True):
        # Horizontal alignment follows the quadrant to avoid clipping.
        ang_deg = np.degrees(ang) % 360
        if 85 <= ang_deg <= 95 or 265 <= ang_deg <= 275:
            ha = "center"
        elif 95 < ang_deg < 265:
            ha = "right"
        else:
            ha = "left"
        ax.text(
            ang,
            rmax * 1.05,
            fam,
            ha=ha,
            va="center",
            fontsize=10,
            color=_family_color(fam),
            fontweight="bold",
        )
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.5, color=theme["muted"])
    ax.set_rlabel_position(180 / max(len(families), 1))
    ax.set_title(
        "Mean approximation ratio per family",
        fontsize=11.5,
        pad=16,
        color=theme["fg"],
        loc="center",
    )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(theme["divider"])


def _method_color(i: int, theme: dict) -> str:
    # Professional, colour-blind friendly palette (Okabe-Ito).
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]
    return palette[i % len(palette)]


def _flatten_mean_ratios(reports: list[dict], clip_min: float) -> list[float]:
    vals: list[float] = [clip_min]
    for r in reports:
        for info in r["families"].values():
            v = info.get("mean_ratio")
            if v is not None and not math.isnan(v):
                vals.append(v)
    return vals


def _panel_horizontal_bars(
    ax,
    reports: list[dict],
    theme: dict,
    *,
    field: str,
    title: str,
    subtitle: str,
    reference: float | None = 1.0,
) -> None:
    families = _ordered_family_axes(reports)
    rows: list[tuple[str, str]] = []  # (family, subset) pairs
    family_bands: list[tuple[int, int, str]] = []  # (start_idx, end_idx, fam)
    start = 0
    for fam in families:
        seen: set[str] = set()
        subsets = []
        for r in reports:
            for s in r["families"].get(fam, {}).get("subsets", []):
                if s.subset not in seen:
                    subsets.append(s.subset)
                    seen.add(s.subset)
        for sub in subsets:
            rows.append((fam, sub))
        if subsets:
            family_bands.append((start, start + len(subsets), fam))
            start += len(subsets)

    if not rows:
        _empty(ax, "no subsets", theme)
        return

    y = np.arange(len(rows))
    h = 0.82 / max(1, len(reports))

    # Family colour bands as light background stripes.
    for a, b, fam in family_bands:
        ax.axhspan(a - 0.5, b - 0.5, color=_family_color(fam), alpha=0.07, zorder=0)

    for i, r in enumerate(reports):
        values = []
        for fam, sub in rows:
            match = _find_subset(r, fam, sub)
            v = getattr(match, field, math.nan) if match else math.nan
            values.append(0.0 if (v is None or math.isnan(v)) else v)
        color = _method_color(i, theme)
        offset = (i - (len(reports) - 1) / 2.0) * h
        ax.barh(
            y + offset,
            values,
            height=h * 0.92,
            label=r["label"],
            color=color,
            edgecolor=theme["bg"],
            linewidth=0.6,
            hatch=_METHOD_HATCHES[i % len(_METHOD_HATCHES)],
            zorder=3,
        )

    # y-tick labels are "family/subset" with the family part recoloured
    # inline via per-tick colour so the family mapping stays visible
    # even though we removed the separate family column.
    labels = [f"{fam}/{sub}" if sub else fam for fam, sub in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.8)
    for tick, (fam, _) in zip(ax.get_yticklabels(), rows, strict=True):
        tick.set_color(_family_color(fam))
    ax.invert_yaxis()
    if reference is not None:
        ax.axvline(
            reference,
            color=theme["muted"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            zorder=2,
        )
    # Title goes well above the axes so it never collides with the subtitle.
    ax.set_title(title, fontsize=11.5, color=theme["fg"], loc="left", pad=22)
    ax.text(
        0.0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        fontsize=8.5,
        color=theme["muted"],
        va="bottom",
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["divider"])
    ax.tick_params(axis="x", colors=theme["muted"], labelsize=8)
    ax.grid(axis="x", alpha=0.35, zorder=1)


def _find_subset(r: dict, fam: str, sub: str) -> SubsetRow | None:
    for s in r["families"].get(fam, {}).get("subsets", []):
        if s.subset == sub:
            return s
    return None


def _panel_violin(ax, reports: list[dict], theme: dict) -> None:
    families = _ordered_family_axes(reports)
    if not families:
        _empty(ax, "no per-instance ratios", theme)
        return

    width = 0.8 / max(1, len(reports))
    x = 0.0
    xticks, xtick_labels = [], []
    any_data = False

    for fam in families:
        xticks.append(x + 0.4)
        xtick_labels.append(fam)
        for i, r in enumerate(reports):
            ratios = r["families"].get(fam, {}).get("instance_ratios", [])
            pos = x + i * width + width / 2
            if len(ratios) >= 2:
                parts = ax.violinplot(
                    [ratios],
                    positions=[pos],
                    widths=width * 0.85,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(_family_color(fam))
                    body.set_alpha(0.35)
                    body.set_edgecolor(_method_color(i, theme))
                    body.set_linewidth(0.8)
                any_data = True
            # Always draw a strip of individual points (small, jittered).
            if ratios:
                rng = np.random.default_rng(7 + i)
                jitter = rng.uniform(-width * 0.18, width * 0.18, size=len(ratios))
                ax.scatter(
                    np.full(len(ratios), pos) + jitter,
                    ratios,
                    s=7,
                    alpha=0.6,
                    color=_method_color(i, theme),
                    edgecolor="none",
                    zorder=3,
                )
                any_data = True
            if ratios:
                # Median line on top of the violin
                median = float(np.median(ratios))
                ax.hlines(
                    median,
                    pos - width * 0.35,
                    pos + width * 0.35,
                    colors=theme["fg"],
                    linewidth=1.6,
                    zorder=4,
                )
        x += 1.0

    if not any_data:
        _empty(ax, "no per-instance ratios in this run", theme)
        return

    ax.axhline(1.0, color=theme["muted"], linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xticks(xticks)
    # Rotate x-labels when many families to keep long names readable.
    rotation = 22 if len(families) >= 6 else 0
    ax.set_xticklabels(
        xtick_labels, fontsize=9, rotation=rotation, ha="right" if rotation else "center"
    )
    for tick, fam in zip(ax.get_xticklabels(), xtick_labels, strict=True):
        tick.set_color(_family_color(fam))
        tick.set_fontweight("bold")
    ax.set_title(
        "Per-instance ratio distribution",
        fontsize=11.5,
        color=theme["fg"],
        loc="left",
        pad=22,
    )
    ax.text(
        0.0,
        1.015,
        "Violins (where n ≥ 2) with median bars and jittered strip of every replica.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=theme["muted"],
        va="bottom",
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["divider"])
    ax.tick_params(axis="y", colors=theme["muted"], labelsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=1)


def _empty(ax, msg: str, theme: dict) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=theme["muted"], fontsize=10)


# --------------------------------------------------------------------------- #
# Top-level composition                                                       #
# --------------------------------------------------------------------------- #


def render(
    reports: list[dict],
    *,
    title: str | None = None,
    theme: str = "light",
) -> plt.Figure:
    """Build the full report figure."""
    th = _THEMES.get(theme, _THEMES["light"])
    plt.rcdefaults()
    plt.style.use(th["style"])
    rc = {
        "figure.facecolor": th["bg"],
        "axes.facecolor": th["bg"],
        "axes.edgecolor": th["divider"],
        "axes.labelcolor": th["fg"],
        "text.color": th["fg"],
        "xtick.color": th["muted"],
        "ytick.color": th["muted"],
        "legend.facecolor": th["panel"],
        "legend.edgecolor": th["divider"],
        "font.family": "DejaVu Sans",
        "font.size": 10,
    }
    with plt.rc_context(rc):  # type: ignore[arg-type]  # Matplotlib stubs require literal keys.
        fig = plt.figure(figsize=(17.0, 12.0), dpi=130, facecolor=th["bg"])

        # HEADER strip (top 7%).
        _draw_header(fig, reports, th, title)

        # KPI band — sits just under the header, spans full width.
        _kpi_band(fig, (0.02, 0.82, 0.96, 0.09), reports, th)

        # RADAR — bottom-left of the upper band. Leave a margin above
        # the axes for the "Mean approximation ratio per family" title
        # so it doesn't collide with the KPI band, and generous room
        # below so the bottom family label stays clear of the
        # feasibility panel title.
        ax_radar = fig.add_axes((0.02, 0.44, 0.36, 0.31), projection="polar")
        _panel_radar(ax_radar, reports, th)

        # PER-SUBSET horizontal bars — upper-right, same band as radar.
        ax_subset = fig.add_axes((0.50, 0.46, 0.47, 0.33))
        _panel_horizontal_bars(
            ax_subset,
            reports,
            th,
            field="mean_ratio",
            title="Mean approximation ratio per subset",
            subtitle="Higher = better. Dashed = published optimum. Colour = method.",
            reference=1.0,
        )

        # FEASIBILITY horizontal bars — lower-left.
        ax_feas = fig.add_axes((0.06, 0.08, 0.40, 0.30))
        _panel_horizontal_bars(
            ax_feas,
            reports,
            th,
            field="feasibility",
            title="Feasibility rate per subset",
            subtitle="Share of replicas satisfying the hard constraints.",
            reference=1.0,
        )

        # VIOLIN — lower-right.
        ax_violin = fig.add_axes((0.55, 0.08, 0.42, 0.30))
        _panel_violin(ax_violin, reports, th)

        # METHOD legend — at the very bottom, spanning both lower panels.
        handles, labels = ax_subset.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.005),
                ncol=min(len(labels), 5),
                frameon=True,
                facecolor=th["panel"],
                edgecolor=th["divider"],
                fontsize=10,
            )

        # FOOTER — citation hint.
        fig.text(
            0.5,
            0.035,
            "dataset  ·  huggingface.co/datasets/Yuma-Ichikawa/qqa4co-bench      "
            "runner  ·  scripts/bench_discs.py      viz  ·  scripts/plot_benchmarks.py",
            ha="center",
            va="center",
            fontsize=8,
            color=th["muted"],
        )

    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("results", nargs="+", type=Path, help="bench_discs.py JSON output(s)")
    p.add_argument(
        "--labels", nargs="+", default=None, help="label per input (default: backend:suite)"
    )
    p.add_argument("--output", type=Path, default=None, help="output image path (PNG/SVG/PDF)")
    p.add_argument("--format", default=None, help="force output format (overrides extension)")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--theme", choices=sorted(_THEMES), default="light", help="light or dark theme")
    p.add_argument(
        "--title", default=None, help="figure title (default: 'QQA4CO benchmark report')"
    )
    args = p.parse_args(argv)

    reports = [load_report(path) for path in args.results]
    if args.labels:
        if len(args.labels) != len(reports):
            p.error("--labels count must match number of result files")
        for r, lab in zip(reports, args.labels, strict=True):
            r["label"] = lab

    fig = render(reports, title=args.title, theme=args.theme)

    if args.output is not None:
        out = args.output
        if args.format:
            out = out.with_suffix(f".{args.format}")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"wrote {out}")

    plt.close(fig)
    return 0
