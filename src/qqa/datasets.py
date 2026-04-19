"""Dataset loaders for benchmark instances shipped with the repository.

All loaders resolve paths relative to the repository root (the directory that
contains ``data/``). You can override the base directory with the
``QQA_DATA_DIR`` environment variable, or pass an explicit ``path=`` argument.

Only ``data/mis/er-small`` and ``data/mis/er-large`` are shipped in this
repository. The other loaders (SAT, Twitter, RB, BA, OptSicom) are provided
for completeness; obtain those datasets separately and point ``QQA_DATA_DIR``
(or the ``path`` argument) at them.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from qqa.problems import (
    MaxClique,
    MaxCliqueInstance,
    MaxCut,
    MaxCutInstance,
    MaximumIndependentSet,
    MaximumIndependentSetInstance,
)

_THIS = Path(__file__).resolve()


def _default_data_dir() -> Path:
    """Resolve the on-disk benchmark dataset directory.

    Priority order:
    1. ``$QQA_DATA_DIR`` if set — explicit user override.
    2. ``<repo_root>/data`` if this module lives inside the source tree
       (``src/qqa/datasets.py`` => ``parents[2] == repo_root``).
    3. ``./data`` next to the current working directory — a sensible
       fallback for wheel installs from PyPI where no source tree exists.

    Loaders raise a clear ``FileNotFoundError`` when the resolved
    directory does not contain the requested benchmark, so callers always
    get an actionable message ("set QQA_DATA_DIR or pass path=") rather
    than an opaque ``listdir`` error.
    """

    env = os.environ.get("QQA_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # When installed in editable / source mode, ``parents[2]`` is the
    # repository root and ``parents[2] / "data"`` ships ``mis/er-small``.
    repo_data = _THIS.parents[2] / "data"
    if repo_data.is_dir():
        return repo_data
    # Wheel install fallback — let the user opt in via cwd/data.
    return Path.cwd() / "data"


DATA_DIR: Path = _default_data_dir()


def _resolve(path: str | os.PathLike | None, default_subpath: str) -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
    else:
        resolved = _default_data_dir() / default_subpath
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Benchmark directory {resolved!s} does not exist. "
            "Set the QQA_DATA_DIR environment variable to point at the "
            "QQA4CO repository's ``data/`` directory, or pass an explicit "
            "``path=`` argument to this loader."
        )
    return resolved


def _load_pickle(p: Path):
    with open(p, "rb") as fh:
        return pickle.load(fh)


# ----------------------------------------------------------------------------
# MIS
# ----------------------------------------------------------------------------


def mis_er_small(
    penalty: float = 3.0,
    problem_type: str = "list",
    device: str | torch.device = "cpu",
    path: str | os.PathLike | None = None,
):
    """Load the small Erdős-Rényi MIS benchmark (~700-800 nodes)."""
    root = _resolve(path, "mis/er-small")
    graphs = [_load_pickle(root / f) for f in sorted(os.listdir(root))]
    if problem_type == "all":
        return MaximumIndependentSetInstance(graphs, 800, penalty=penalty, device=device)
    return [MaximumIndependentSet(g, penalty=penalty, device=device) for g in graphs]


def mis_er_large(
    penalty: float = 3.0,
    problem_type: str = "list",
    device: str | torch.device = "cpu",
    path: str | os.PathLike | None = None,
):
    """Load the large Erdős-Rényi MIS benchmark (~9000-11000 nodes)."""
    root = _resolve(path, "mis/er-large")
    graphs = [_load_pickle(root / f) for f in sorted(os.listdir(root))]
    if problem_type == "all":
        return MaximumIndependentSetInstance(graphs, 10915, penalty=penalty, device=device)
    return [MaximumIndependentSet(g, penalty=penalty, device=device) for g in graphs]


def mis_sat(
    group: str = "all",
    problem_type: str = "list",
    device: str | torch.device = "cpu",
    path: str | os.PathLike | None = None,
):
    """Load SAT-based MIS graphs (external dataset)."""
    root = _resolve(path, "mis/SAT_graphs_ver2")
    files = sorted(os.listdir(root))
    if group == "first":
        subset = files[: len(files) // 2]
    elif group == "second":
        subset = files[len(files) // 2 :]
    else:
        subset = files
    graphs = [nx.from_numpy_array(np.load(root / f)) for f in subset]
    if problem_type == "all":
        return MaximumIndependentSetInstance(graphs, 1347, penalty=1, device=device)
    return [MaximumIndependentSet(g, penalty=1, device=device) for g in graphs]


# ----------------------------------------------------------------------------
# Max Clique
# ----------------------------------------------------------------------------


def mcq_twitter(
    problem_type: str = "list",
    device: str | torch.device = "cpu",
    path: str | os.PathLike | None = None,
):
    root = _resolve(path, "maxclique/twitter")
    graphs: list[nx.Graph] = []
    for f in sorted(os.listdir(root)):
        data = _load_pickle(root / f)
        graphs.extend(data[0])
    if problem_type == "all":
        return MaxCliqueInstance(graphs, 247, device=device)
    return [MaxClique(g, device=device) for g in graphs]


def mcq_RB(
    group: str = "all",
    problem_type: str = "list",
    device: str | torch.device = "cpu",
    path: str | os.PathLike | None = None,
):
    root = _resolve(path, "maxclique/RB_test")
    files = sorted(os.listdir(root))
    if group == "first":
        subset = files[: len(files) // 2]
    elif group == "second":
        subset = files[len(files) // 2 :]
    else:
        subset = files
    graphs: list[nx.Graph] = []
    for f in subset:
        data = _load_pickle(root / f)
        for g in data:
            graphs.append(g[1])
    if problem_type == "all":
        return MaxCliqueInstance(graphs, 475, device=device)
    return [MaxClique(g, device=device) for g in graphs]


# ----------------------------------------------------------------------------
# Max Cut
# ----------------------------------------------------------------------------


_MAX_NODES_BA = [1100, 150, 20, 300, 40, 600, 75]


def mct_ba(
    case: int,
    problem_type: str = "list",
    device: str | torch.device = "cpu",
    path: str | os.PathLike | None = None,
):
    base = _resolve(path, "maxcut/maxcut-ba")
    folder = sorted(os.listdir(base))[case]
    root = base / folder
    graphs = [_load_pickle(root / f)[0] for f in sorted(os.listdir(root))]
    if problem_type == "all":
        return MaxCutInstance(graphs, _MAX_NODES_BA[case], device=device)
    return [MaxCut(g, device=device) for g in graphs]


def mct_er(
    case: int,
    problem_type: str = "list",
    device: str | torch.device = "cpu",
    path: str | os.PathLike | None = None,
):
    base = _resolve(path, "maxcut/maxcut-er")
    folder = sorted(os.listdir(base))[case]
    root = base / folder
    graphs = [_load_pickle(root / f)[0] for f in sorted(os.listdir(root))]
    if problem_type == "all":
        return MaxCutInstance(graphs, _MAX_NODES_BA[case], device=device)
    return [MaxCut(g, device=device) for g in graphs]


def mct_opt(
    problem_type: str = "list",
    device: str | torch.device = "cpu",
    path: str | os.PathLike | None = None,
):
    root = _resolve(path, "maxcut/optsicom")
    graphs = [_load_pickle(root / f)[0] for f in sorted(os.listdir(root))]
    if problem_type == "all":
        return MaxCutInstance(graphs, 125, device=device)
    return [MaxCut(g, device=device) for g in graphs]


# ----------------------------------------------------------------------------
# Known-best values (optional helpers)
# ----------------------------------------------------------------------------


def best_twitter(path: str | os.PathLike | None = None) -> np.ndarray:
    root = _resolve(path, "maxclique/twitter")
    bests: list[float] = []
    for f in sorted(os.listdir(root)):
        data = _load_pickle(root / f)
        bests.extend(data[1])
    return np.asarray(bests)


def best_RB(path: str | os.PathLike | None = None) -> np.ndarray:
    root = _resolve(path, "maxclique/RB_test")
    bests: list[float] = []
    for f in sorted(os.listdir(root)):
        data = _load_pickle(root / f)
        for item in data:
            bests.append(item[0])
    return np.asarray(bests)


def best_ba(case: int, path: str | os.PathLike | None = None) -> np.ndarray:
    base = _resolve(path, "maxcut/maxcut-ba")
    folder = sorted(os.listdir(base))[case]
    root = base / folder
    bests = [_load_pickle(root / f)[1][0] for f in sorted(os.listdir(root))]
    return np.asarray(bests)


def best_er(case: int, path: str | os.PathLike | None = None) -> np.ndarray:
    base = _resolve(path, "maxcut/maxcut-er")
    folder = sorted(os.listdir(base))[case]
    root = base / folder
    bests = [_load_pickle(root / f)[1][0] for f in sorted(os.listdir(root))]
    return np.asarray(bests)


def best_opt(path: str | os.PathLike | None = None) -> np.ndarray:
    root = _resolve(path, "maxcut/optsicom")
    bests = [_load_pickle(root / f)[1][0] for f in sorted(os.listdir(root))]
    return np.asarray(bests)
