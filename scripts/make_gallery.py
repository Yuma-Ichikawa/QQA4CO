"""Regenerate the visualization gallery shipped under ``docs/assets/gallery/``.

Each run is deliberately small so the whole script finishes in a few minutes on
a laptop CPU. The PNGs are referenced from ``docs/visualization.md`` and
should be re-generated deterministically after
any behavioural change to :func:`qqa.anneal` or :mod:`qqa.visualization`.

Run with::

    uv run python scripts/make_gallery.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import networkx as nx

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import qqa  # noqa: E402
from qqa import visualization as viz  # noqa: E402
from qqa.callbacks import PopulationTracker  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "gallery"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUT.parents[2])}")


def _run(
    problem,
    *,
    sol_size: int,
    epochs: int,
    lr: float = 1.0,
    temp: float = 1e-3,
    min_bg: float = -3.0,
    max_bg: float = 0.1,
    curve_rate: int = 4,
    div_param: float = 0.2,
    stride: int | None = None,
):
    tracker = PopulationTracker(stride=max(1, stride or max(1, epochs // 80)), record_x=True)
    result = qqa.anneal(
        problem,
        sol_size=sol_size,
        learning_rate=lr,
        temp=temp,
        schedule=qqa.LinearBGSchedule(min_bg=min_bg, max_bg=max_bg),
        curve_rate=curve_rate,
        div_param=div_param,
        num_epochs=epochs,
        verbose=False,
        callbacks=[tracker],
    )
    return result, tracker


def _dump_set(kind: str, problem, result, tracker, *, skip_solution: bool = False) -> None:
    fig, _ = viz.plot_history(result, title=f"{kind} — dynamics", show=False)
    _save(fig, f"history_{kind}.png")

    fig, _ = viz.plot_best_trajectory(result, title=f"{kind} — best objective", show=False)
    _save(fig, f"best_{kind}.png")

    if not skip_solution:
        try:
            fig, _ = viz.plot_solution_heatmap(
                result, problem=problem, title=f"{kind} — best solution", show=False
            )
            _save(fig, f"solution_{kind}.png")
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"  [skip] solution heatmap for {kind}: {e}")

    try:
        fig, _ = viz.plot_population_evolution(
            tracker, title=f"{kind} — parallel population", backend="matplotlib", show=False
        )
        _save(fig, f"population_{kind}.png")
    except Exception as e:  # pragma: no cover - diagnostic only
        print(f"  [skip] population plot for {kind}: {e}")


def _schedule_figure() -> None:
    fig, _ = viz.plot_schedule(
        qqa.LinearBGSchedule(-3.0, 0.1),
        num_epochs=1000,
        title="LinearBGSchedule(-3.0 → 0.1)",
        show=False,
    )
    _save(fig, "schedule_default.png")


def gallery() -> None:
    qqa.fix_seed(0)

    print("[gallery] MIS (random regular, N=40, d=3)")
    g = nx.random_regular_graph(d=3, n=40, seed=0)
    problem = qqa.MaximumIndependentSet(g, penalty=2)
    result, tracker = _run(problem, sol_size=64, epochs=900)
    _dump_set("mis", problem, result, tracker)

    print("[gallery] MaxCut (Erdős-Rényi, N=40, p=0.15)")
    g = nx.erdos_renyi_graph(n=40, p=0.15, seed=1)
    problem = qqa.MaxCut(g)
    result, tracker = _run(problem, sol_size=64, epochs=900)
    _dump_set("maxcut", problem, result, tracker)

    print("[gallery] Coloring (random regular, N=30, d=4, K=3)")
    g = nx.random_regular_graph(d=4, n=30, seed=2)
    problem = qqa.Coloring(g, num_category=3)
    result, tracker = _run(problem, sol_size=64, epochs=1200, curve_rate=4, div_param=0.2)
    _dump_set("coloring", problem, result, tracker, skip_solution=True)

    print("[gallery] Ising 1D ferromagnetic (N=32, J=1, periodic)")
    problem = qqa.Ising1D(N=32, J=1.0, h=0.0, periodic=True)
    result, tracker = _run(problem, sol_size=64, epochs=600, curve_rate=2)
    _dump_set("ising1d", problem, result, tracker)

    print("[gallery] Edwards-Anderson 3D (L=4, seed=0)")
    problem = qqa.EdwardsAnderson(L=4, dim=3, seed=0)
    result, tracker = _run(problem, sol_size=128, epochs=1500, curve_rate=2)
    _dump_set("ea3d", problem, result, tracker)

    print("[gallery] Sherrington-Kirkpatrick (N=80, seed=0)")
    problem = qqa.SherringtonKirkpatrick(N=80, seed=0)
    result, tracker = _run(problem, sol_size=128, epochs=1500, curve_rate=2)
    _dump_set("sk", problem, result, tracker)

    print("[gallery] BinaryPerceptron (N=40, alpha=0.4)")
    problem = qqa.BinaryPerceptron(N=40, alpha=0.4, seed=0, sharpness=10.0)
    result, tracker = _run(problem, sol_size=128, epochs=1200, curve_rate=2)
    _dump_set("perceptron", problem, result, tracker)

    print("[gallery] Hopfield memory (N=64, P=3)")
    problem = qqa.HopfieldMemory(N=64, patterns=3, seed=0)
    result, tracker = _run(problem, sol_size=128, epochs=1000, curve_rate=2)
    _dump_set("hopfield", problem, result, tracker)

    print("[gallery] Default annealing schedule figure")
    _schedule_figure()

    print(f"[gallery] done — {len(list(OUT.glob('*.png')))} figures in {OUT}")


if __name__ == "__main__":  # pragma: no cover
    gallery()
