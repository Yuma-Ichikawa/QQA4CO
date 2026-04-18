"""Regenerate the shipped example notebooks from short Python snippets.

Run with: ``uv run python scripts/_generate_notebooks.py``.
Safe to re-run; overwrites examples/*.ipynb deterministically.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES.mkdir(exist_ok=True)

REPO_SLUG = "Yuma-Ichikawa/QQA4CO"
COLAB_BADGE_TEMPLATE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/{slug}/blob/main/examples/{nb})"
)


def _colab_install_cell() -> str:
    """First code cell — installs the package on Colab only (no-op locally)."""
    return (
        "# Install QQA on Google Colab (no-op if already installed locally)\n"
        "import importlib.util\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        "if importlib.util.find_spec('qqa') is None:\n"
        "    subprocess.check_call(\n"
        "        [\n"
        "            sys.executable,\n"
        "            '-m',\n"
        "            'pip',\n"
        "            'install',\n"
        "            '--quiet',\n"
        f"            'qqa @ git+https://github.com/{REPO_SLUG}.git',\n"
        "        ]\n"
        "    )"
    )


def make_nb(title: str, subtitle: str, cells: list[tuple[str, str]], *, nb_filename: str):
    nb = nbf.v4.new_notebook()
    badge = COLAB_BADGE_TEMPLATE.format(slug=REPO_SLUG, nb=nb_filename)
    header = nbf.v4.new_markdown_cell(f"# {title}\n\n{badge}\n\n{subtitle}")
    nb.cells.append(header)
    nb.cells.append(nbf.v4.new_code_cell(_colab_install_cell()))
    for kind, src in cells:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": []},
    }
    return nb


COMMON_IMPORTS = """\
import matplotlib.pyplot as plt
import networkx as nx

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
        nb_filename="01_maximum_independent_set.ipynb",
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
        nb_filename="02_graph_coloring.ipynb",
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
        nb_filename="03_max_cut.ipynb",
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
        nb_filename="04_edwards_anderson_3d.ipynb",
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
        nb_filename="05_sherrington_kirkpatrick.ipynb",
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
            "# ``best_sol`` has shape (N,) for single-instance problems; add a batch dim\n"
            "s_best = result.best_sol.unsqueeze(0)\n"
            "print(f'surrogate loss: {result.best_obj:.3f}')\n"
            "print(f'exact errors : {int(problem.error_count(s_best))}')",
        ),
        ("code", "viz.plot_best_trajectory(result, show=False);"),
    ]
    return make_nb(
        "QQA 06 – Binary perceptron",
        "Teacher/student binary perceptron learning as an optimization problem.",
        body,
        nb_filename="06_binary_perceptron.ipynb",
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
            "s_best = result.best_sol.unsqueeze(0)\n"
            "m = problem.overlap(s_best)\n"
            "print(f'overlap with stored patterns: {m[0].tolist()}')",
        ),
    ]
    return make_nb(
        "QQA 07 – Hopfield memory",
        "Associative memory ground-state recovery.",
        body,
        nb_filename="07_hopfield_memory.ipynb",
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
        nb_filename="08_parallel_benchmark.ipynb",
    )


def nb00():
    """One-click Google Colab quickstart: every problem, one short cell each."""
    body = [
        (
            "md",
            "This notebook walks through every problem family shipped with QQA on "
            "Google Colab. It installs `qqa` from GitHub, detects CUDA if "
            "available, and runs a small `qqa.anneal` job per problem with an "
            "inline `viz.plot_history` / `viz.plot_best_trajectory` figure. "
            "The whole notebook finishes in ~2 minutes on a free CPU Colab "
            "runtime and ~30s on a GPU runtime.",
        ),
        ("md", "## Setup"),
        (
            "code",
            "import matplotlib.pyplot as plt\n"
            "import networkx as nx\n"
            "import torch\n\n"
            "import qqa\n"
            "from qqa import visualization as viz\n\n"
            "qqa.fix_seed(0)\n"
            "print('QQA version:', qqa.__version__)\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "print('device:', device)",
        ),
        ("md", "## 1. Maximum Independent Set"),
        (
            "code",
            "g = nx.random_regular_graph(d=3, n=50, seed=0)\n"
            "problem = qqa.MaximumIndependentSet(g, penalty=2, device=device)\n"
            "r = qqa.anneal(problem, sol_size=64, num_epochs=1000, device=device, verbose=False)\n"
            "print(f'MIS size >= {-int(r.best_obj)}  ({r.runtime:.2f}s)')\n"
            "viz.plot_history(r, show=False);",
        ),
        ("md", "## 2. Graph coloring (K=3)"),
        (
            "code",
            "g = nx.random_regular_graph(d=3, n=40, seed=0)\n"
            "problem = qqa.Coloring(g, num_category=3, device=device)\n"
            "r = qqa.anneal(problem, sol_size=64, num_epochs=1500, device=device, verbose=False)\n"
            "print(f'conflicts: {int(round(r.best_obj))}')\n"
            "viz.plot_best_trajectory(r, show=False);",
        ),
        ("md", "## 3. Max-Cut"),
        (
            "code",
            "g = nx.erdos_renyi_graph(n=40, p=0.2, seed=0)\n"
            "problem = qqa.MaxCut(g, device=device)\n"
            "r = qqa.anneal(problem, sol_size=64, num_epochs=1000, device=device, verbose=False)\n"
            "print(f'cut value >= {-float(r.best_obj):.2f}')\n"
            "viz.plot_history(r, show=False);",
        ),
        ("md", "## 4. 1D Ising ferromagnet"),
        (
            "code",
            "problem = qqa.Ising1D(N=32, J=1.0, periodic=True, device=device)\n"
            "r = qqa.anneal(problem, sol_size=64, num_epochs=600, device=device, verbose=False)\n"
            "print(f'E = {float(r.best_obj):.3f}  (target: -32)')\n"
            "viz.plot_best_trajectory(r, show=False);",
        ),
        ("md", "## 5. Edwards–Anderson 3D spin glass"),
        (
            "code",
            "problem = qqa.EdwardsAnderson(L=4, dim=3, seed=0, device=device)\n"
            "r = qqa.anneal(problem, sol_size=128, num_epochs=1500, device=device, verbose=False)\n"
            "print(f'E / N = {float(r.best_obj) / problem.num_spins:.4f}')\n"
            "viz.plot_history(r, show=False);",
        ),
        ("md", "## 6. Sherrington–Kirkpatrick"),
        (
            "code",
            "problem = qqa.SherringtonKirkpatrick(N=100, seed=0, device=device)\n"
            "r = qqa.anneal(problem, sol_size=128, num_epochs=1500, device=device, verbose=False)\n"
            "print(f'e_0 = {float(r.best_obj) / 100:.4f}  (Parisi: -0.7632)')\n"
            "viz.plot_best_trajectory(r, show=False);",
        ),
        ("md", "## 7. Binary perceptron"),
        (
            "code",
            "problem = qqa.BinaryPerceptron(N=30, alpha=0.5, seed=0, sharpness=10.0, device=device)\n"
            "r = qqa.anneal(problem, sol_size=128, num_epochs=1500, device=device, verbose=False)\n"
            "s_best = problem.relaxation.project(r.best_sol).unsqueeze(0)\n"
            "print(f'min errors = {int(problem.error_count(s_best).min())}')\n"
            "viz.plot_best_trajectory(r, show=False);",
        ),
        ("md", "## 8. Hopfield memory"),
        (
            "code",
            "problem = qqa.HopfieldMemory(N=64, patterns=3, seed=0, device=device)\n"
            "r = qqa.anneal(problem, sol_size=128, num_epochs=1000, device=device, verbose=False)\n"
            "s_best = problem.relaxation.project(r.best_sol).unsqueeze(0)\n"
            "overlap = problem.overlap(s_best).abs().max().item()\n"
            "print(f'max overlap with stored pattern: {overlap:.3f}')\n"
            "viz.plot_history(r, show=False);",
        ),
        ("md", "## 9. Parallel MIS (`MaximumIndependentSetInstance`)"),
        (
            "code",
            "N, degrees = 60, [2, 3, 4, 5]\n"
            "graphs = [nx.random_regular_graph(d=d, n=N, seed=d) for d in degrees]\n"
            "problem = qqa.MaximumIndependentSetInstance(graphs, max_node=N, penalty=2, device=device)\n"
            "r = qqa.anneal(problem, sol_size=64, num_epochs=800, device=device, verbose=False)\n"
            "for d, obj in zip(degrees, r.best_obj, strict=False):\n"
            "    print(f'  degree={d}: MIS >= {-int(round(float(obj)))}')",
        ),
        ("md", "## Custom loss via `UserProblem`"),
        (
            "code",
            "import torch\n\n"
            "N = 40\n"
            "g = torch.Generator().manual_seed(0)\n"
            "J = torch.randn(N, N, generator=g) / (N ** 0.5)\n"
            "J = (J + J.T) / 2\n"
            "J.fill_diagonal_(0.0)\n"
            "problem = qqa.UserProblem(\n"
            "    num_vars=N, variable_kind='spin',\n"
            "    loss_fn=lambda s: -0.5 * torch.einsum('bi,ij,bj->b', s, J, s),\n"
            ")\n"
            "r = qqa.anneal(problem, sol_size=128, num_epochs=1000, verbose=False)\n"
            "print(f'custom spin-glass e_0 = {float(r.best_obj) / N:.4f}')",
        ),
    ]
    return make_nb(
        "QQA 00 – Colab quickstart",
        (
            "One-notebook tour of every built-in problem, designed to run on a "
            "free Google Colab CPU runtime."
        ),
        body,
        nb_filename="00_colab_quickstart.ipynb",
    )


def main() -> None:
    builders = {
        "00_colab_quickstart.ipynb": nb00,
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
