"""Regenerate the shipped example notebooks from short Python snippets.

Run with: ``uv run python scripts/_generate_notebooks.py``.
Safe to re-run; overwrites examples/*.ipynb deterministically.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES.mkdir(exist_ok=True)


def make_nb(title: str, subtitle: str, cells: list[tuple[str, str]]):
    nb = nbf.v4.new_notebook()
    header = nbf.v4.new_markdown_cell(f"# {title}\n\n{subtitle}")
    nb.cells.append(header)
    for kind, src in cells:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    return nb


COMMON_IMPORTS = """\
import networkx as nx
import matplotlib.pyplot as plt
import qqa
from qqa import visualization as viz

qqa.fix_seed(0)
print('QQA version:', qqa.__version__)
"""


def save(path: Path, nb):
    with open(path, "w") as fh:
        nbf.write(nb, fh)


def nb01():
    body = [
        ("md", "## Setup"),
        ("code", COMMON_IMPORTS),
        ("md", "## Problem: Maximum Independent Set on a random regular graph"),
        (
            "code",
            "g = nx.random_regular_graph(d=3, n=50, seed=0)\n"
            "problem = qqa.MaximumIndependentSet(g, penalty=2)\n"
            "print(f'N = {problem.num_nodes}, |E| = {g.number_of_edges()}')",
        ),
        ("md", "## Run QQA"),
        (
            "code",
            "result = qqa.anneal(problem, sol_size=100, num_epochs=1500, verbose=False)\n"
            "print(f'MIS size (lower bound): {-int(result.best_obj)}')\n"
            "print(f'runtime: {result.runtime:.2f}s')",
        ),
        ("md", "## Visualise the dynamics"),
        ("code", "viz.plot_history(result, show=False);"),
    ]
    return make_nb(
        "QQA 01 – Maximum Independent Set",
        "Solve MIS on a random 3-regular graph using the unified `qqa.anneal` API.",
        body,
    )


def nb02():
    body = [
        ("code", COMMON_IMPORTS),
        (
            "code",
            "g = nx.random_regular_graph(d=3, n=40, seed=0)\n"
            "problem = qqa.Coloring(g, num_category=3)",
        ),
        (
            "code",
            "result = qqa.anneal(problem, sol_size=100, num_epochs=2000, verbose=False)\n"
            "print(f'conflict count: {int(result.best_obj)}')",
        ),
        ("code", "viz.plot_best_trajectory(result, show=False);"),
    ]
    return make_nb(
        "QQA 02 – Graph coloring",
        "Three-coloring of a random 3-regular graph.",
        body,
    )


def nb03():
    body = [
        ("code", COMMON_IMPORTS),
        (
            "code",
            "g = nx.erdos_renyi_graph(n=60, p=0.15, seed=0)\n"
            "for u, v in g.edges:\n    g[u][v]['weight'] = 1.0\n"
            "problem = qqa.MaxCut(g)",
        ),
        (
            "code",
            "result = qqa.anneal(problem, sol_size=100, num_epochs=1500, verbose=False)\n"
            "cut = -float(result.best_obj) / 2\n"
            "print(f'approx cut size: {cut:.2f}')",
        ),
        ("code", "viz.plot_history(result, show=False);"),
    ]
    return make_nb(
        "QQA 03 – Max-Cut",
        "Max-Cut on an Erdős–Rényi graph.",
        body,
    )


def nb04():
    body = [
        ("code", COMMON_IMPORTS),
        (
            "md",
            "## 3D Edwards–Anderson spin glass\n\n"
            "A $L^3$ cubic lattice with independent $\\mathcal{N}(0, 1)$ couplings on "
            "nearest-neighbour bonds.",
        ),
        (
            "code",
            "problem = qqa.EdwardsAnderson(L=4, dim=3, seed=0)\n"
            "print(f'N = {problem.num_spins}, couplings shape = {tuple(problem.J.shape)}')",
        ),
        (
            "code",
            "result = qqa.anneal(problem, sol_size=200, num_epochs=2000, verbose=False)\n"
            "print(f'E_0 / N ≈ {result.best_obj / problem.num_spins:.4f}')",
        ),
        ("code", "viz.plot_best_trajectory(result, show=False);"),
    ]
    return make_nb(
        "QQA 04 – Edwards–Anderson 3D",
        "Ground-state energy estimation for the classic 3D EA model.",
        body,
    )


def nb05():
    body = [
        ("code", COMMON_IMPORTS),
        (
            "md",
            "## Sherrington–Kirkpatrick model\n\n"
            "All-to-all couplings $J_{ij}\\sim\\mathcal{N}(0, 1/N)$. "
            "The Parisi ground-state energy density is $e_0 \\approx -0.7632$.",
        ),
        (
            "code",
            "problem = qqa.SherringtonKirkpatrick(N=100, seed=0)\n"
            "result = qqa.anneal(problem, sol_size=200, num_epochs=2000, verbose=False)\n"
            "print(f'E_0 / N = {result.best_obj / 100:.4f}  (target ≈ -0.7632)')",
        ),
        ("code", "viz.plot_history(result, show=False);"),
    ]
    return make_nb(
        "QQA 05 – Sherrington–Kirkpatrick",
        "Mean-field spin-glass ground-state energy.",
        body,
    )


def nb06():
    body = [
        ("code", COMMON_IMPORTS),
        (
            "md",
            "## Discrete (binary) perceptron\n\n"
            "Teacher–student setup: the teacher generates labels, the student must "
            "find binary weights that reproduce them.",
        ),
        (
            "code",
            "problem = qqa.BinaryPerceptron(N=50, alpha=0.5, seed=0, sharpness=10.0)\n"
            "print(f'patterns: {problem.num_patterns}')",
        ),
        (
            "code",
            "result = qqa.anneal(problem, sol_size=200, num_epochs=2000, verbose=False)\n"
            "s_best = result.best_sol[0]\n"
            "print(f'surrogate loss: {result.best_obj:.3f}')\n"
            "print(f'exact errors : {int(problem.error_count(s_best.unsqueeze(0)))}')",
        ),
        ("code", "viz.plot_best_trajectory(result, show=False);"),
    ]
    return make_nb(
        "QQA 06 – Binary perceptron",
        "Teacher/student binary perceptron learning as an optimization problem.",
        body,
    )


def nb07():
    body = [
        ("code", COMMON_IMPORTS),
        (
            "md",
            "## Hopfield associative memory\n\n"
            "Store $P$ random $\\pm 1$ patterns via Hebbian couplings and check that "
            "QQA recovers a stored pattern.",
        ),
        (
            "code",
            "problem = qqa.HopfieldMemory(N=80, patterns=3, seed=0)\n"
            "result = qqa.anneal(problem, sol_size=128, num_epochs=1500, verbose=False)\n"
            "print(f'energy: {result.best_obj:.3f}  (target ≈ {-80/2:.3f} per pattern)')",
        ),
        (
            "code",
            "s_best = result.best_sol[0]\n"
            "m = problem.overlap(s_best.unsqueeze(0))\n"
            "print(f'overlap with stored patterns: {m[0].tolist()}')",
        ),
    ]
    return make_nb(
        "QQA 07 – Hopfield memory",
        "Associative memory ground-state recovery.",
        body,
    )


def nb08():
    body = [
        ("code", COMMON_IMPORTS),
        (
            "md",
            "## Parallel benchmark with hyper-parameter sweep\n\n"
            "Run multiple QQA configurations on the same problem and inspect the "
            "parallel-coordinates plot of the results.",
        ),
        (
            "code",
            "import itertools, pandas as pd\n"
            "problem = qqa.SherringtonKirkpatrick(N=80, seed=0)\n"
            "rows = []\n"
            "for mb, Mb, dp in itertools.product([-3, -2, -1], [0.0, 0.1], [0.0, 0.1]):\n"
            "    r = qqa.anneal(problem, sol_size=64, num_epochs=600,\n"
            "                   min_bg=mb, max_bg=Mb, div_param=dp, verbose=False)\n"
            "    rows.append({'min_bg': mb, 'max_bg': Mb, 'div_param': dp,\n"
            "                 'best_obj_per_N': float(r.best_obj) / 80})\n"
            "df = pd.DataFrame(rows)\n"
            "df",
        ),
        (
            "code",
            "try:\n"
            "    fig = viz.plot_parallel_coordinates(df, objective='best_obj_per_N', show=False)\n"
            "    fig\n"
            "except Exception as e:\n"
            "    print('Plotly not installed; install with pip install qqa[plotly]')\n"
            "    print(e)",
        ),
    ]
    return make_nb(
        "QQA 08 – Parallel benchmark",
        "Sweep QQA hyper-parameters and visualise the outcome.",
        body,
    )


def main() -> None:
    builders = {
        "01_maximum_independent_set.ipynb": nb01,
        "02_graph_coloring.ipynb": nb02,
        "03_max_cut.ipynb": nb03,
        "04_edwards_anderson_3d.ipynb": nb04,
        "05_sherrington_kirkpatrick.ipynb": nb05,
        "06_binary_perceptron.ipynb": nb06,
        "07_hopfield_memory.ipynb": nb07,
        "08_parallel_benchmark.ipynb": nb08,
    }
    for name, fn in builders.items():
        save(EXAMPLES / name, fn())
        print("wrote", name)


if __name__ == "__main__":
    main()
