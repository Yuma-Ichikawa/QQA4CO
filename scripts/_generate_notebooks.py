"""Regenerate the shipped example notebooks from short Python snippets.

Run with: ``uv run python scripts/_generate_notebooks.py``.
Safe to re-run; overwrites examples/*.ipynb deterministically.

Generated code uses double quotes throughout so the notebooks pass
``ruff format --check`` without a second pass.
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
    """First code cell — installs ``qqa`` on Colab only.

    Prefers PyPI (small, cached, stable), falls back to ``git+`` for
    users wanting the ``main`` branch. A no-op when ``qqa`` is already
    importable (local / CI runs skip the pip invocation entirely).
    """
    return (
        """\
# Install QQA on Google Colab (no-op if already installed locally).
# We prefer the released wheel on PyPI; users who want bleeding-edge
# ``main`` can set QQA_INSTALL_FROM_GIT=1 before running this cell.
import importlib.util
import os
import subprocess
import sys

if importlib.util.find_spec("qqa") is None:
    if os.environ.get("QQA_INSTALL_FROM_GIT") == "1":
        spec = "qqa @ git+https://github.com/"""
        + REPO_SLUG
        + """.git"
    else:
        spec = "qqa"
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", spec]
    )"""
    )


def _det_id(nb_filename: str, idx: int) -> str:
    """Return the same compact IDs used by the ``nbstripout`` commit hook.

    ``nbformat`` otherwise assigns random IDs, while ``nbstripout`` normalises
    them to their ordinal position.  Generate that canonical form directly so
    regeneration and pre-commit are byte-stable.
    """
    del nb_filename
    return str(idx)


def make_nb(title: str, subtitle: str, cells: list[tuple[str, str]], *, nb_filename: str):
    nb = nbf.v4.new_notebook()
    badge = COLAB_BADGE_TEMPLATE.format(slug=REPO_SLUG, nb=nb_filename)
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n{badge}\n\n{subtitle}",
            id=_det_id(nb_filename, 0),
        )
    )
    nb.cells.append(nbf.v4.new_code_cell(_colab_install_cell(), id=_det_id(nb_filename, 1)))
    for i, (kind, src) in enumerate(cells, start=2):
        cell_id = _det_id(nb_filename, i)
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src, id=cell_id))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src, id=cell_id))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": []},
    }
    return nb


COMMON_IMPORTS = """\
import matplotlib.pyplot as plt  # noqa: F401
import networkx as nx

import qqa
from qqa import visualization as viz

qqa.fix_seed(0)
print("QQA version:", qqa.__version__)
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
            'print(f"N = {problem.num_nodes}, |E| = {g.number_of_edges()}")',
        ),
        ("md", "## Run QQA"),
        (
            "code",
            "result = qqa.anneal(problem, sol_size=100, num_epochs=1500, verbose=False)\n"
            'print(f"MIS size (lower bound): {-int(result.best_obj)}")\n'
            'print(f"runtime: {result.runtime:.2f}s")',
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
            'print(f"conflict count: {int(result.best_obj)}")',
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
            'for u, v in g.edges:\n    g[u][v]["weight"] = 1.0\n'
            "problem = qqa.MaxCut(g)",
        ),
        (
            "code",
            "result = qqa.anneal(problem, sol_size=100, num_epochs=1500, verbose=False)\n"
            "cut = -float(result.best_obj) / 2\n"
            'print(f"approx cut size: {cut:.2f}")',
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
            'print(f"N = {problem.num_spins}, couplings shape = {tuple(problem.J.shape)}")',
        ),
        (
            "code",
            "result = qqa.anneal(problem, sol_size=200, num_epochs=2000, verbose=False)\n"
            'print(f"E_0 / N ≈ {result.best_obj / problem.num_spins:.4f}")',
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
            'print(f"E_0 / N = {result.best_obj / 100:.4f}  (target ≈ -0.7632)")',
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
            'print(f"patterns: {problem.num_patterns}")',
        ),
        (
            "code",
            "result = qqa.anneal(problem, sol_size=200, num_epochs=2000, verbose=False)\n"
            "# ``best_sol`` has shape (N,) for single-instance problems; add a batch dim\n"
            "s_best = result.best_sol.unsqueeze(0)\n"
            'print(f"surrogate loss: {result.best_obj:.3f}")\n'
            'print(f"exact errors : {int(problem.error_count(s_best))}")',
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
            'print(f"energy: {result.best_obj:.3f}  (target ≈ {-80 / 2:.3f} per pattern)")',
        ),
        (
            "code",
            "s_best = result.best_sol.unsqueeze(0)\n"
            "m = problem.overlap(s_best)\n"
            'print(f"overlap with stored patterns: {m[0].tolist()}")',
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
            "import itertools\n"
            "\n"
            "import pandas as pd\n"
            "\n"
            "problem = qqa.SherringtonKirkpatrick(N=80, seed=0)\n"
            "rows = []\n"
            "for mb, Mb, dp in itertools.product([-3, -2, -1], [0.0, 0.1], [0.0, 0.1]):\n"
            "    r = qqa.anneal(\n"
            "        problem,\n"
            "        sol_size=64,\n"
            "        num_epochs=600,\n"
            "        min_bg=mb,\n"
            "        max_bg=Mb,\n"
            "        div_param=dp,\n"
            "        verbose=False,\n"
            "    )\n"
            "    rows.append(\n"
            "        {\n"
            '            "min_bg": mb,\n'
            '            "max_bg": Mb,\n'
            '            "div_param": dp,\n'
            '            "best_obj_per_N": float(r.best_obj) / 80,\n'
            "        }\n"
            "    )\n"
            "df = pd.DataFrame(rows)\n"
            "df",
        ),
        (
            "code",
            "try:\n"
            '    fig = viz.plot_parallel_coordinates(df, objective="best_obj_per_N", show=False)\n'
            "except Exception as e:\n"
            '    print("Plotly not installed; install with pip install qqa[plotly]")\n'
            "    print(e)\n"
            "else:\n"
            "    fig.show()",
        ),
    ]
    return make_nb(
        "QQA 08 – Parallel benchmark",
        "Sweep QQA hyper-parameters and visualise the outcome.",
        body,
        nb_filename="08_parallel_benchmark.ipynb",
    )


def nb12():
    """Natural-language front door for QQA, SCIP, Pareto, and black-box runs."""
    body = [
        (
            "md",
            "## Why this notebook?\n\n"
            "The same `qqa.ask(...)` call compiles an ordinary-language decision "
            "problem into a strict, reviewable model and routes it locally. The LLM "
            "never executes code or chooses an arbitrary solver. Keep API keys in an "
            "environment variable or a hidden prompt—never in the notebook.",
        ),
        (
            "code",
            "import getpass\n"
            "import os\n"
            "\n"
            "import qqa\n"
            "\n"
            'print("QQA version:", qqa.__version__)\n'
            'if not os.environ.get("QQA_LLM_API_KEY"):\n'
            '    key = getpass.getpass("Compatible API key (leave blank for offline cells): ")\n'
            "    if key:\n"
            '        os.environ["QQA_LLM_API_KEY"] = key\n'
            'if os.environ.get("QQA_LLM_API_KEY") and not os.environ.get("QQA_LLM_BASE_URL"):\n'
            '    base_url = input("OpenAI-compatible base URL: ").strip()\n'
            "    if base_url:\n"
            '        os.environ["QQA_LLM_BASE_URL"] = base_url\n'
            'if os.environ.get("QQA_LLM_API_KEY") and not os.environ.get("QQA_LLM_MODEL"):\n'
            '    model_id = input("Model ID: ").strip()\n'
            "    if model_id:\n"
            '        os.environ["QQA_LLM_MODEL"] = model_id\n'
            "live_api_ready = all(\n"
            "    os.environ.get(name)\n"
            '    for name in ("QQA_LLM_API_KEY", "QQA_LLM_BASE_URL", "QQA_LLM_MODEL")\n'
            ")\n"
            'print("Live API profile ready:", live_api_ready)',
        ),
        (
            "md",
            "## 1. Review and solve without an API\n\n"
            "A validated JSON model is ideal for reproducible production runs. This "
            "mixed binary/integer/real example works without credentials.",
        ),
        (
            "code",
            "production_spec = {\n"
            '    "name": "production-plan",\n'
            '    "variables": [\n'
            '        {"name": "open", "kind": "binary", "lower": 0, "upper": 1, "size": 2},\n'
            '        {"name": "lots", "kind": "integer", "lower": 0, "upper": 12, "size": 2},\n'
            '        {"name": "overtime", "kind": "real", "lower": 0, "upper": 16, "size": 1},\n'
            "    ],\n"
            '    "objectives": [\n'
            "        {\n"
            '            "name": "weekly_cost",\n'
            '            "direction": "min",\n'
            '            "expression": "1400*open[0] + 1100*open[1] + 460*lots[0] + 510*lots[1] + 38*square(overtime)",\n'
            '            "unit": "USD",\n'
            "        }\n"
            "    ],\n"
            '    "constraints": [\n'
            "        {\n"
            '            "name": "demand", "expression": "8*lots[0] + 7*lots[1] + overtime",\n'
            '            "sense": ">=", "rhs": 105, "weight": 1000, "scale": 105, "tolerance": 0.05,\n'
            "        },\n"
            "        {\n"
            '            "name": "link_a", "expression": "lots[0] - 12*open[0]",\n'
            '            "sense": "<=", "rhs": 0, "weight": 500, "scale": 12, "tolerance": 0.01,\n'
            "        },\n"
            "        {\n"
            '            "name": "link_b", "expression": "lots[1] - 10*open[1]",\n'
            '            "sense": "<=", "rhs": 0, "weight": 500, "scale": 10, "tolerance": 0.01,\n'
            "        },\n"
            "    ],\n"
            '    "notes": "",\n'
            "}\n"
            'plan = qqa.plan_spec(production_spec, solver="qqa")\n'
            'plan.to_dict()["routing"]',
        ),
        (
            "code",
            "answer = qqa.execute_plan(\n"
            '    plan, sol_size=128, num_epochs=800, device="auto", seed=7\n'
            ")\n"
            "answer.result.score",
        ),
        (
            "md",
            "## 2. Natural language → QQA + SCIP\n\n"
            'With `qqa[scip]` installed, `solver="auto"` uses QQA exploration '
            "followed by SCIP certification for compatible single-objective models.",
        ),
        (
            "code",
            'single_request = """\n'
            "Plan production at two plants. Opening decisions are binary, production\n"
            "lots are bounded integers, and overtime is continuous. Minimize fixed,\n"
            "lot, and quadratic overtime costs while meeting demand and linking each\n"
            "plant's production to its opening decision. Use the numerical bounds and\n"
            "coefficients from the reviewed production example above.\n"
            '"""\n'
            "if live_api_ready:\n"
            "    single = qqa.ask(\n"
            "        single_request,\n"
            '        solver="auto",\n'
            '        device="auto",\n'
            "        sol_size=128,\n"
            "        num_epochs=800,\n"
            "        scip_time_limit=30,\n"
            "    )\n"
            '    display(single.plan.to_dict()["routing"])\n'
            "    display(single.result.score)\n"
            "else:\n"
            '    print("Set the QQA_LLM_* API profile to run this live translation.")',
        ),
        (
            "md",
            "## 3. Natural language → one-run Pareto front\n\n"
            "Multiple objectives are preserved as separate goals. Parallel reference "
            "directions recover a nondominated archive in one run.",
        ),
        (
            "code",
            'pareto_request = """\n'
            "Allocate integer production lots and continuous overtime while deciding\n"
            "which plants open. Simultaneously minimize total cost, carbon emissions,\n"
            "and unmet-demand risk. Keep each objective separate and enforce capacity,\n"
            "activation, and demand constraints. Give every variable an explicit,\n"
            "realistic finite bound and record assumptions.\n"
            '"""\n'
            "if live_api_ready:\n"
            "    pareto = qqa.ask(\n"
            "        pareto_request,\n"
            '        solver="auto",\n'
            "        sol_size=256,\n"
            "        num_epochs=1000,\n"
            '        device="auto",\n'
            "    )\n"
            '    display(pareto.plan.to_dict()["routing"])\n'
            "    qqa.plot_pareto(pareto.result)\n"
            "    qqa.plot_pareto_diagnostics(pareto.result)\n"
            "else:\n"
            '    print("Set the QQA_LLM_* API profile to run this live translation.")',
        ),
        (
            "md",
            "## 4. Natural language → budget-aware black-box optimisation\n\n"
            "Mentioning an expensive simulator or black-box experiment makes `auto` "
            "select batch surrogate optimisation. The safe expression is evaluated "
            "point by point, with no gradients exposed to the optimiser.",
        ),
        (
            "code",
            'blackbox_request = """\n'
            "Treat reactor tuning as an expensive black-box experiment. Choose an\n"
            "integer reactor count from 1 to 8 and real temperature from 300 to 500.\n"
            "Minimize (reactors-4)^2 + ((temperature-410)/30)^2 with at most 96\n"
            "parallelizable evaluations, subject to reactors*temperature <= 2800.\n"
            '"""\n'
            "if live_api_ready:\n"
            "    blackbox = qqa.ask(\n"
            "        blackbox_request,\n"
            '        solver="auto",\n'
            "        budget=96,\n"
            "        batch_size=8,\n"
            "        workers=8,\n"
            '        device="auto",\n'
            "    )\n"
            '    display(blackbox.plan.to_dict()["routing"])\n'
            "    display(blackbox.result.best_point)\n"
            "    qqa.plot_blackbox(blackbox.result)\n"
            "else:\n"
            '    print("Set the QQA_LLM_* API profile to run this live translation.")',
        ),
        (
            "md",
            "## CLI equivalents\n\n"
            "The provider-neutral profile is read from `QQA_LLM_API_KEY`, "
            "`QQA_LLM_BASE_URL`, and `QQA_LLM_MODEL`. The key never appears in the "
            "command line, generated model, result JSON, or report.",
        ),
        (
            "code",
            'print("""qqa ask "Minimize (x-2)^2 for real x in [-5,5]" --plan-only --show-model\n'
            "qqa ask --file realistic-request.txt --solver auto --device auto \\\\\n"
            "  --output-plan plan.json --output-result result.json --report result.html\n"
            "qqa ask --spec plan-model.json --solver qqa --device auto\n"
            'qqa gui  # open the Ask QQA tab""")',
        ),
    ]
    return make_nb(
        "QQA 12 – Natural-language optimization",
        (
            "One safe entry point for QQA, QQA+SCIP, one-run Pareto fronts, and "
            "budget-aware black-box optimization."
        ),
        body,
        nb_filename="12_natural_language_optimization_colab.ipynb",
    )


def nb00():
    """One-click Google Colab quickstart: every problem, one short cell each."""
    body = [
        (
            "md",
            "This notebook walks through every problem family shipped with QQA on "
            "Google Colab. It installs `qqa` **from PyPI**, detects CUDA if "
            "available, and runs a small `qqa.anneal` job per problem with an "
            "inline `viz.plot_history` / `viz.plot_best_trajectory` figure. "
            "The whole notebook finishes in ~2 minutes on a free CPU Colab "
            "runtime and ~30s on a GPU runtime. Set "
            "`QQA_INSTALL_FROM_GIT=1` before the install cell to track the "
            "`main` branch instead.",
        ),
        ("md", "## Setup"),
        (
            "code",
            "import matplotlib.pyplot as plt  # noqa: F401\n"
            "import networkx as nx\n"
            "import torch\n"
            "\n"
            "import qqa\n"
            "from qqa import visualization as viz\n"
            "\n"
            "qqa.fix_seed(0)\n"
            'print("QQA version:", qqa.__version__)\n'
            'device = "cuda" if torch.cuda.is_available() else "cpu"\n'
            'print("device:", device)',
        ),
        ("md", "## 1. Maximum Independent Set"),
        (
            "code",
            "g = nx.random_regular_graph(d=3, n=50, seed=0)\n"
            "problem = qqa.MaximumIndependentSet(g, penalty=2, device=device)\n"
            "r = qqa.anneal(problem, sol_size=64, num_epochs=1000, device=device, verbose=False)\n"
            'print(f"MIS size >= {-int(r.best_obj)}  ({r.runtime:.2f}s)")\n'
            "viz.plot_history(r, show=False);",
        ),
        ("md", "## 2. Graph coloring (K=3)"),
        (
            "code",
            "g = nx.random_regular_graph(d=3, n=40, seed=0)\n"
            "problem = qqa.Coloring(g, num_category=3, device=device)\n"
            "r = qqa.anneal(problem, sol_size=64, num_epochs=1500, device=device, verbose=False)\n"
            'print(f"conflicts: {int(round(r.best_obj))}")\n'
            "viz.plot_best_trajectory(r, show=False);",
        ),
        ("md", "## 3. Max-Cut"),
        (
            "code",
            "g = nx.erdos_renyi_graph(n=40, p=0.2, seed=0)\n"
            "problem = qqa.MaxCut(g, device=device)\n"
            "r = qqa.anneal(problem, sol_size=64, num_epochs=1000, device=device, verbose=False)\n"
            'print(f"cut value >= {-float(r.best_obj):.2f}")\n'
            "viz.plot_history(r, show=False);",
        ),
        ("md", "## 4. 1D Ising ferromagnet"),
        (
            "code",
            "problem = qqa.Ising1D(N=32, J=1.0, periodic=True, device=device)\n"
            "r = qqa.anneal(problem, sol_size=64, num_epochs=600, device=device, verbose=False)\n"
            'print(f"E = {float(r.best_obj):.3f}  (target: -32)")\n'
            "viz.plot_best_trajectory(r, show=False);",
        ),
        ("md", "## 5. Edwards–Anderson 3D spin glass"),
        (
            "code",
            "problem = qqa.EdwardsAnderson(L=4, dim=3, seed=0, device=device)\n"
            "r = qqa.anneal(problem, sol_size=128, num_epochs=1500, device=device, verbose=False)\n"
            'print(f"E / N = {float(r.best_obj) / problem.num_spins:.4f}")\n'
            "viz.plot_history(r, show=False);",
        ),
        ("md", "## 6. Sherrington–Kirkpatrick"),
        (
            "code",
            "problem = qqa.SherringtonKirkpatrick(N=100, seed=0, device=device)\n"
            "r = qqa.anneal(problem, sol_size=128, num_epochs=1500, device=device, verbose=False)\n"
            'print(f"e_0 = {float(r.best_obj) / 100:.4f}  (Parisi: -0.7632)")\n'
            "viz.plot_best_trajectory(r, show=False);",
        ),
        ("md", "## 7. Binary perceptron"),
        (
            "code",
            "problem = qqa.BinaryPerceptron(N=30, alpha=0.5, seed=0, sharpness=10.0, device=device)\n"
            "r = qqa.anneal(problem, sol_size=128, num_epochs=1500, device=device, verbose=False)\n"
            "s_best = problem.relaxation.project(r.best_sol).unsqueeze(0)\n"
            'print(f"min errors = {int(problem.error_count(s_best).min())}")\n'
            "viz.plot_best_trajectory(r, show=False);",
        ),
        ("md", "## 8. Hopfield memory"),
        (
            "code",
            "problem = qqa.HopfieldMemory(N=64, patterns=3, seed=0, device=device)\n"
            "r = qqa.anneal(problem, sol_size=128, num_epochs=1000, device=device, verbose=False)\n"
            "s_best = problem.relaxation.project(r.best_sol).unsqueeze(0)\n"
            "overlap = problem.overlap(s_best).abs().max().item()\n"
            'print(f"max overlap with stored pattern: {overlap:.3f}")\n'
            "viz.plot_history(r, show=False);",
        ),
        ("md", "## 9. Parallel MIS (`MaximumIndependentSetInstance`)"),
        (
            "code",
            "N, degrees = 60, [2, 3, 4, 5]\n"
            "graphs = [nx.random_regular_graph(d=d, n=N, seed=d) for d in degrees]\n"
            "problem = qqa.MaximumIndependentSetInstance(\n"
            "    graphs, max_node=N, penalty=2, device=device\n"
            ")\n"
            "r = qqa.anneal(problem, sol_size=64, num_epochs=800, device=device, verbose=False)\n"
            "for d, obj in zip(degrees, r.best_obj, strict=False):\n"
            '    print(f"  degree={d}: MIS >= {-int(round(float(obj)))}")',
        ),
        ("md", "## Custom loss via `UserProblem`"),
        (
            "code",
            "import torch\n"
            "\n"
            "N = 40\n"
            "g = torch.Generator().manual_seed(0)\n"
            "J = torch.randn(N, N, generator=g) / (N**0.5)\n"
            "J = (J + J.T) / 2\n"
            "J.fill_diagonal_(0.0)\n"
            "problem = qqa.UserProblem(\n"
            "    num_vars=N,\n"
            '    variable_kind="spin",\n'
            '    loss_fn=lambda s: -0.5 * torch.einsum("bi,ij,bj->b", s, J, s),\n'
            ")\n"
            "r = qqa.anneal(problem, sol_size=128, num_epochs=1000, verbose=False)\n"
            'print(f"custom spin-glass e_0 = {float(r.best_obj) / N:.4f}")',
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


def _apply_ruff_format() -> None:
    """Run ``ruff format`` on the freshly written notebooks.

    Keeps the on-disk output byte-identical to ``ruff format --check``
    so CI or downstream format checks see no drift.
    """
    import shutil
    import subprocess

    ruff = shutil.which("ruff")
    if ruff is None:
        return  # fall back silently — ruff is only a dev dep
    subprocess.run([ruff, "format", "--quiet", str(EXAMPLES)], check=False)


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
        "12_natural_language_optimization_colab.ipynb": nb12,
    }
    for name, fn in builders.items():
        save(EXAMPLES / name, fn())
        print("wrote", name)
    _apply_ruff_format()


if __name__ == "__main__":
    main()
