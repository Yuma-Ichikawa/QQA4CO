"""Visualise a ``scripts/bench_discs.py --output results.json`` payload.

Renders four panels so that method improvements are easy to spot at a
glance across every benchmark family shipped with QQA4CO
(``DISCS`` / ``coloring`` / ``mis-rrg`` / ``ea3d`` / ``balanced-partition``):

1. **Radar chart** — one axis per benchmark family, value = mean
   approximation ratio (missing families render at ``0.0`` so a radar
   spoke that doesn't cover one axis is immediately visible).
2. **Bar chart** — per *subset* mean ratio, grouped and coloured by
   family. Gives you fine-grained "which subsets did I win/lose on?"
   information that the radar summary hides.
3. **Feasibility bar** — feasibility rate per subset (share of replicas
   that satisfy the hard constraints). Penalised QUBO families
   (``mis``, ``maxclique``, ``coloring``, ``mis-rrg``) can report a
   high objective *and* an infeasibility rate > 0; this panel surfaces
   that failure mode.
4. **Box plot of per-instance ratios** — distribution of approximation
   ratios within each family. Lets you see variance, skew, and
   per-instance outliers that a mean hides.

If multiple ``results.json`` files are passed, every file is rendered
as a separate series so A/B comparisons are a one-liner:

    python scripts/plot_benchmarks.py \
        baseline.json my_method.json --labels baseline mine \
        --output bench_report.png

CLI
---
::

    python scripts/plot_benchmarks.py results.json --output bench.png
    python scripts/plot_benchmarks.py r1.json r2.json --output cmp.png \
        --labels baseline tuned
    python scripts/plot_benchmarks.py results.json --format svg --show

The script deliberately avoids any new dependency: it only needs
matplotlib + numpy, which are already core QQA4CO dependencies.
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

# --------------------------------------------------------------------------- #
# Public data model                                                           #
# --------------------------------------------------------------------------- #

# A "report" is our loader-agnostic view of what we need from the JSON.
# This makes the rest of the file easy to unit-test: the test can build
# a synthetic report without going through argparse + file I/O.
#
# report = {
#     "label":  "my-method",
#     "families": {
#         "mis":      {
#             "mean_ratio": 0.98,
#             "subsets":  [{"subset": "satlib/uf", "mean_ratio": 0.985,
#                           "feasibility": 1.0, "n": 500,
#                           "instance_ratios": [0.98, 0.99, ...]}],
#         },
#         ...
#     },
# }
#
# Notes:
# * Families with no ratio (e.g. ea3d with unknown ground state,
#   balanced-partition) carry NaN in ``mean_ratio`` and an empty
#   ``instance_ratios`` list; the plots skip those axes.


def load_report(path: Path) -> dict[str, Any]:
    """Parse a ``bench_discs.py`` JSON payload into our normalised shape."""
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
        families[fam]["subsets"].append(
            {
                "subset": key,
                "mean_ratio": float(subset_ratio) if subset_ratio is not None else math.nan,
                "feasibility": feas_rate,
                "n": int(agg.get("n", 0)),
                "instance_ratios": [float(x) for x in ratios],
            }
        )
        families[fam]["instance_ratios"].extend(float(x) for x in ratios)

    for info in families.values():
        weights, values = [], []
        for s in info["subsets"]:
            if s["n"] > 0 and not math.isnan(s["mean_ratio"]):
                weights.append(s["n"])
                values.append(s["mean_ratio"])
        info["mean_ratio"] = float(np.average(values, weights=weights)) if values else math.nan

    return {
        "label": _default_label(raw),
        "families": dict(families),
        "raw": raw,
    }


def _subset_key(agg: dict[str, Any]) -> str:
    gt = agg.get("graph_type") or ""
    sub = agg.get("subset") or ""
    if gt and sub:
        return f"{gt}/{sub}"
    return gt or sub or "default"


def _default_label(raw: dict[str, Any]) -> str:
    suite = raw.get("suite", "?")
    backend = raw.get("backend", "?")
    return f"{backend}:{suite}"


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #


def _ordered_family_axes(reports: list[dict]) -> list[str]:
    # Keep a stable, documented family order so A/B plots are aligned.
    preferred = [
        "mis",
        "maxcut",
        "maxclique",
        "normcut",
        "coloring",
        "mis-rrg",
        "ea3d",
        "balanced-partition",
    ]
    present: set[str] = set()
    for r in reports:
        present.update(r["families"].keys())
    axes = [f for f in preferred if f in present]
    axes += sorted(present - set(preferred))
    return axes


def _radar(ax, reports: list[dict]) -> None:
    families = _ordered_family_axes(reports)
    if not families:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return

    theta = np.linspace(0, 2 * np.pi, len(families), endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])

    for r in reports:
        vals = [r["families"].get(f, {}).get("mean_ratio", math.nan) for f in families]
        plot_vals = [0.0 if (v is None or math.isnan(v)) else max(0.0, min(1.5, v)) for v in vals]
        plot_vals.append(plot_vals[0])
        ax.plot(theta_closed, plot_vals, linewidth=2, label=r["label"])
        ax.fill(theta_closed, plot_vals, alpha=0.15)

    ax.set_xticks(theta)
    ax.set_xticklabels(families, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.set_ylim(0.0, max(1.05, *_flatten_mean_ratios(reports, 1.0)))
    ax.set_title("mean approximation ratio (higher = better)", fontsize=10)
    ax.grid(True, alpha=0.4)


def _flatten_mean_ratios(reports: list[dict], clip_min: float) -> list[float]:
    vals: list[float] = [clip_min]
    for r in reports:
        for info in r["families"].values():
            v = info.get("mean_ratio")
            if v is not None and not math.isnan(v):
                vals.append(v)
    return vals


def _subset_bar(ax, reports: list[dict]) -> None:
    families = _ordered_family_axes(reports)
    labels, blocks = [], []
    for fam in families:
        seen: set[str] = set()
        for r in reports:
            for s in r["families"].get(fam, {}).get("subsets", []):
                if s["subset"] not in seen:
                    labels.append(f"{fam}/{s['subset']}")
                    seen.add(s["subset"])
                    blocks.append((fam, s["subset"]))

    if not labels:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no subsets", ha="center", va="center")
        return

    x = np.arange(len(blocks))
    width = 0.8 / max(1, len(reports))
    for i, r in enumerate(reports):
        values = []
        for fam, subset in blocks:
            match = next(
                (s for s in r["families"].get(fam, {}).get("subsets", []) if s["subset"] == subset),
                None,
            )
            v = match["mean_ratio"] if match else math.nan
            values.append(0.0 if (v is None or math.isnan(v)) else v)
        ax.bar(x + i * width - 0.4 + width / 2, values, width, label=r["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("mean approximation ratio")
    ax.set_title("per-subset mean approximation ratio (higher = better)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def _feasibility_bar(ax, reports: list[dict]) -> None:
    families = _ordered_family_axes(reports)
    labels, blocks = [], []
    for fam in families:
        seen: set[str] = set()
        for r in reports:
            for s in r["families"].get(fam, {}).get("subsets", []):
                if s["subset"] not in seen:
                    labels.append(f"{fam}/{s['subset']}")
                    seen.add(s["subset"])
                    blocks.append((fam, s["subset"]))

    if not labels:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no subsets", ha="center", va="center")
        return

    x = np.arange(len(blocks))
    width = 0.8 / max(1, len(reports))
    for i, r in enumerate(reports):
        values = []
        for fam, subset in blocks:
            match = next(
                (s for s in r["families"].get(fam, {}).get("subsets", []) if s["subset"] == subset),
                None,
            )
            v = match["feasibility"] if match else math.nan
            values.append(0.0 if (v is None or math.isnan(v)) else v)
        ax.bar(x + i * width - 0.4 + width / 2, values, width, label=r["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("feasibility rate")
    ax.set_title(
        "share of feasible solutions (1.0 = all replicas satisfy constraints)", fontsize=10
    )
    ax.grid(axis="y", alpha=0.3)


def _instance_box(ax, reports: list[dict]) -> None:
    families = _ordered_family_axes(reports)
    if not families:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return

    positions = []
    data = []
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(reports), 1)))
    x = 0.0
    xticks, xtick_labels = [], []
    width = 0.8 / max(1, len(reports))
    for fam in families:
        xticks.append(x + 0.4)
        xtick_labels.append(fam)
        for i, r in enumerate(reports):
            ratios = r["families"].get(fam, {}).get("instance_ratios", [])
            positions.append(x + i * width + width / 2)
            data.append(ratios if ratios else [math.nan])
        x += 1.0

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width * 0.9,
        patch_artist=True,
        showfliers=True,
    )
    for idx, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[idx % len(reports)])
        box.set_alpha(0.6)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("per-instance approximation ratio")
    ax.set_title("per-instance ratio distribution (median + IQR + outliers)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)


# --------------------------------------------------------------------------- #
# Top-level                                                                   #
# --------------------------------------------------------------------------- #


def render(
    reports: list[dict],
    *,
    title: str | None = None,
) -> plt.Figure:
    """Build the 2x2 figure shared by ``--output`` and ``--show``."""
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(2, 2)

    ax_radar = fig.add_subplot(gs[0, 0], projection="polar")
    ax_subset = fig.add_subplot(gs[0, 1])
    ax_feas = fig.add_subplot(gs[1, 0])
    ax_box = fig.add_subplot(gs[1, 1])

    _radar(ax_radar, reports)
    _subset_bar(ax_subset, reports)
    _feasibility_bar(ax_feas, reports)
    _instance_box(ax_box, reports)

    ax_subset.legend(loc="lower right", fontsize=8)
    ax_feas.legend(loc="lower right", fontsize=8)

    if title is None:
        title = " vs ".join(r["label"] for r in reports)
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
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
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--show", action="store_true", help="open an interactive window")
    p.add_argument("--title", default=None, help="figure suptitle (default: 'A vs B vs ...')")
    args = p.parse_args(argv)

    reports = [load_report(path) for path in args.results]
    if args.labels:
        if len(args.labels) != len(reports):
            p.error("--labels count must match number of result files")
        for r, lab in zip(reports, args.labels, strict=True):
            r["label"] = lab

    fig = render(reports, title=args.title)

    if args.show:
        matplotlib.use("TkAgg", force=True)
        plt.show()

    if args.output is not None:
        out = args.output
        if args.format:
            out = out.with_suffix(f".{args.format}")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
