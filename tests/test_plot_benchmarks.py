"""Smoke + contract tests for ``scripts/plot_benchmarks.py``.

We keep the assertions conservative (image existence, Axes count,
legend wiring, ``load_report`` field contract) — not pixel-level — so
the tests stay stable across matplotlib versions while still catching:

* breakage of the JSON -> report normaliser,
* accidental removal / rename of one of the four panels,
* regressions in multi-file (A/B) mode label assignment.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # deterministic, headless CI

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
plot = importlib.import_module("plot_benchmarks")


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _synth_payload() -> dict:
    """Minimal ``bench_discs.py`` JSON covering 3 families with mixed ratios."""
    return {
        "backend": "qqa",
        "suite": "demo",
        "device": "cpu",
        "results": [
            {
                "problem": "mis",
                "graph_type": "er",
                "subset": "800",
                "n": 3,
                "n_feasible": 3,
                "mean_objective": 43.0,
                "mean_ratio": 1.04,
                "instances": [
                    {
                        "instance": 0,
                        "id": "0001",
                        "objective": 43,
                        "feasible": True,
                        "best_known": 41,
                        "ratio": 1.049,
                        "wall_s": 1.0,
                    },
                    {
                        "instance": 1,
                        "id": "0002",
                        "objective": 42,
                        "feasible": True,
                        "best_known": 40,
                        "ratio": 1.050,
                        "wall_s": 1.0,
                    },
                    {
                        "instance": 2,
                        "id": "0003",
                        "objective": 43,
                        "feasible": True,
                        "best_known": 42,
                        "ratio": 1.024,
                        "wall_s": 1.0,
                    },
                ],
            },
            {
                "problem": "coloring",
                "graph_type": "myciel",
                "subset": "",
                "n": 2,
                "n_feasible": 1,
                "mean_objective": 0.5,
                "mean_ratio": None,
                "instances": [
                    {
                        "instance": 0,
                        "id": "myciel3",
                        "objective": 0,
                        "feasible": True,
                        "best_known": 0.0,
                        "ratio": None,
                        "wall_s": 0.5,
                    },
                    {
                        "instance": 1,
                        "id": "myciel4",
                        "objective": 1,
                        "feasible": False,
                        "best_known": 0.0,
                        "ratio": None,
                        "wall_s": 0.5,
                    },
                ],
            },
            {
                "problem": "ea3d",
                "graph_type": "gaussian",
                "subset": "L4",
                "n": 2,
                "n_feasible": 2,
                "mean_objective": -95.6,
                "mean_ratio": 0.89,
                "instances": [
                    {
                        "instance": 0,
                        "id": "0001",
                        "objective": -95.6,
                        "feasible": True,
                        "best_known": -100.0,
                        "ratio": 0.956,
                        "wall_s": 0.5,
                    },
                    {
                        "instance": 1,
                        "id": "0002",
                        "objective": -90.0,
                        "feasible": True,
                        "best_known": -100.0,
                        "ratio": 0.9,
                        "wall_s": 0.5,
                    },
                ],
            },
        ],
    }


@pytest.fixture
def payload_path(tmp_path: Path) -> Path:
    p = tmp_path / "results.json"
    with p.open("w") as fh:
        json.dump(_synth_payload(), fh)
    return p


# --------------------------------------------------------------------------- #
# load_report                                                                 #
# --------------------------------------------------------------------------- #


def test_load_report_normalises_every_family(payload_path: Path):
    rep = plot.load_report(payload_path)
    assert set(rep["families"]) == {"mis", "coloring", "ea3d"}

    mis = rep["families"]["mis"]
    assert math.isclose(mis["mean_ratio"], 1.04, rel_tol=1e-2)
    assert len(mis["subsets"]) == 1
    assert len(mis["instance_ratios"]) == 3

    coloring = rep["families"]["coloring"]
    # coloring has no ratios -> mean_ratio NaN but feasibility reflects 1/2.
    assert math.isnan(coloring["mean_ratio"])
    assert coloring["instance_ratios"] == []
    assert coloring["subsets"][0]["feasibility"] == 0.5

    ea3d = rep["families"]["ea3d"]
    # Family mean == subset mean (single-subset family) == 0.89 from fixture.
    assert math.isclose(ea3d["mean_ratio"], 0.89, rel_tol=1e-2)
    assert ea3d["subsets"][0]["feasibility"] == 1.0
    assert len(ea3d["instance_ratios"]) == 2


def test_load_report_label_default(payload_path: Path):
    rep = plot.load_report(payload_path)
    assert rep["label"] == "qqa:demo"


# --------------------------------------------------------------------------- #
# render                                                                      #
# --------------------------------------------------------------------------- #


def test_render_produces_expected_panels(payload_path: Path):
    rep = plot.load_report(payload_path)
    fig = plot.render([rep])
    try:
        # Polished layout: header strip + KPI band + 4 data panels + footer.
        # The exact number of chrome axes may evolve; what matters is:
        #   * there is exactly one polar (radar) axes, and
        #   * there are at least 4 data panels (radar + 3 cartesian data).
        polar_axes = [ax for ax in fig.axes if ax.name == "polar"]
        assert len(polar_axes) == 1
        assert len(polar_axes[0].lines) >= 1

        # At least four total panels once chrome is counted.
        assert len(fig.axes) >= 4

        # A cartesian panel drawing bars should carry legend handles.
        bar_axes = [
            ax for ax in fig.axes if ax.name != "polar" and any(p.get_label() for p in ax.patches)
        ]
        assert bar_axes, "no bar-style panel found"
    finally:
        plt.close(fig)


def test_render_ab_comparison_preserves_both_labels(payload_path: Path, tmp_path: Path):
    # Fabricate a second run by halving every ratio: this simulates a
    # worse baseline so A/B rendering has 2 radar spokes, 2 bar series.
    data = _synth_payload()
    for agg in data["results"]:
        if agg["mean_ratio"] is not None:
            agg["mean_ratio"] *= 0.5
        for inst in agg["instances"]:
            if inst["ratio"] is not None:
                inst["ratio"] *= 0.5
    p2 = tmp_path / "baseline.json"
    p2.write_text(json.dumps(data))

    r1 = plot.load_report(payload_path)
    r2 = plot.load_report(p2)
    r1["label"] = "method"
    r2["label"] = "baseline"

    fig = plot.render([r1, r2])
    try:
        # Either axis-level legend or a figure-level legend is fine — the
        # contract is that *both* method labels are wired somewhere.
        legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
        legends += list(fig.legends)
        labels: set[str] = set()
        for leg in legends:
            labels.update(t.get_text() for t in leg.get_texts())
        assert {"method", "baseline"} <= labels
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def test_cli_writes_image(payload_path: Path, tmp_path: Path):
    out = tmp_path / "out.png"
    rc = plot.main([str(payload_path), "--output", str(out)])
    assert rc == 0
    assert out.is_file()
    # PNG magic = 89 50 4E 47 0D 0A 1A 0A
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_rejects_mismatched_labels(payload_path: Path, tmp_path: Path):
    out = tmp_path / "out.png"
    with pytest.raises(SystemExit):
        plot.main([str(payload_path), "--labels", "one", "two", "--output", str(out)])
