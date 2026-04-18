"""Tests for :mod:`qqa.visualization` across both backends."""

from __future__ import annotations

import importlib
import warnings

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pytest

import qqa
from qqa import visualization as viz

matplotlib.use("Agg")


def _have_plotly() -> bool:
    try:
        importlib.import_module("plotly.graph_objects")
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def small_result():
    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=16, seed=0)
    problem = qqa.MaximumIndependentSet(g)
    return qqa.anneal(problem, sol_size=16, num_epochs=120, verbose=False), problem


def test_plot_history_matplotlib(small_result):
    result, _ = small_result
    fig, axs = viz.plot_history(result, backend="matplotlib", show=False)
    assert isinstance(fig, plt.Figure)
    assert len(axs) == 3
    plt.close(fig)


def test_plot_best_trajectory_matplotlib(small_result):
    result, _ = small_result
    fig, _ = viz.plot_best_trajectory(result, backend="matplotlib", show=False)
    plt.close(fig)


def test_plot_schedule_matplotlib():
    sched = qqa.LinearBGSchedule(min_bg=-2.0, max_bg=0.1)
    fig, _ = viz.plot_schedule(sched, num_epochs=50, backend="matplotlib", show=False)
    plt.close(fig)


@pytest.mark.skipif(not _have_plotly(), reason="plotly not installed")
def test_plot_history_plotly(small_result):
    import plotly.graph_objects as go

    result, _ = small_result
    fig = viz.plot_history(result, backend="plotly", show=False)
    assert isinstance(fig, go.Figure)


@pytest.mark.skipif(not _have_plotly(), reason="plotly not installed")
def test_plot_run_comparison_plotly(small_result):
    import plotly.graph_objects as go

    result, _ = small_result
    fig = viz.plot_run_comparison([result, result], labels=["a", "b"], backend="plotly", show=False)
    assert isinstance(fig, go.Figure)


@pytest.mark.skipif(not _have_plotly(), reason="plotly not installed")
def test_plot_parallel_coordinates_plotly():
    import plotly.graph_objects as go

    df = {"min_bg": [-2, -1], "max_bg": [0.0, 0.1], "best_obj": [-5.0, -4.0]}
    fig = viz.plot_parallel_coordinates(df, objective="best_obj", backend="plotly", show=False)
    assert isinstance(fig, go.Figure)


def test_plot_solution_heatmap_matplotlib(small_result):
    result, problem = small_result
    fig, _ = viz.plot_solution_heatmap(result, problem=problem, backend="matplotlib", show=False)
    plt.close(fig)


def test_fallback_warns_when_plotly_missing(monkeypatch, small_result):
    if _have_plotly():
        pytest.skip("plotly is installed; fallback path not exercised here")
    result, _ = small_result
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig, _ = viz.plot_history(result, backend="plotly", show=False)
        assert any("plotly" in str(x.message).lower() for x in w)
    plt.close(fig)
