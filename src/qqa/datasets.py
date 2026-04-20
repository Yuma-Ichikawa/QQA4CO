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

import json
import os
import pickle
from dataclasses import dataclass
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
    NormalizedCut,
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


# ============================================================================ #
# DISCS unified suite (data/discs/<problem>/<graph_type>/<subset>/...)         #
# ============================================================================ #
#
# This block consumes the layout produced by ``scripts/setup_discs_data.sh``
# (which calls ``scripts/convert_discs_to_qqa.py``). Each subset directory
# contains:
#
#   * ``{0001,...}.gpickle`` — one ``pickle.dump(networkx.Graph)`` per file.
#   * ``manifest.jsonl``     — one JSON record per graph carrying the
#                              ``best_known`` objective and provenance fields.
#
# All four DISCS CO problem families (MaxCut / MIS / MaxClique / NormCut) share
# the same on-disk schema, so a single helper drives every loader below.
# ---------------------------------------------------------------------------- #


def _default_discs_root() -> Path:
    return _default_data_dir() / "discs"


@dataclass(frozen=True)
class DiscsBenchmark:
    """A loaded DISCS subset.

    Attributes
    ----------
    problems:
        List of concrete ``COProblem`` instances ready to feed into
        ``qqa.anneal`` / ``qqa.simulated_annealing`` / ``qqa.population_annealing``.
    best_known:
        ``np.ndarray`` of length ``len(problems)``. Entries are ``np.nan`` for
        instances whose source did not carry a known optimum.
    manifest:
        Raw manifest records (1 dict per instance) — useful for surfacing
        ``num_nodes``, ``source``, etc. in benchmark dashboards.
    subset_dir:
        Path to the directory we read from.
    """

    problems: list
    best_known: np.ndarray
    manifest: list[dict]
    subset_dir: Path

    def __len__(self) -> int:
        return len(self.problems)


def _read_manifest(subset_dir: Path) -> list[dict]:
    manifest = subset_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"DISCS manifest not found at {manifest}. Did you run `./scripts/setup_discs_data.sh`?"
        )
    records: list[dict] = []
    with open(manifest) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise FileNotFoundError(f"DISCS manifest at {manifest} is empty.")
    return records


def _resolve_discs_subset(
    problem: str,
    graph_type: str,
    subset: str | None,
    *,
    root: str | os.PathLike | None,
) -> tuple[Path, list[dict]]:
    """Locate the subset directory + manifest for one (problem, graph_type, subset)."""
    base = Path(root).expanduser().resolve() if root is not None else _default_discs_root()
    if not base.is_dir():
        raise FileNotFoundError(
            f"DISCS unified data root {base} does not exist. "
            "Run `./scripts/setup_discs_data.sh` (or set QQA_DATA_DIR)."
        )
    candidates: list[Path] = []
    type_dir = base / problem / graph_type
    if not type_dir.is_dir():
        raise FileNotFoundError(
            f"DISCS subset directory {type_dir} not found. "
            f"Available problem dirs: {sorted(p.name for p in base.iterdir() if p.is_dir())}"
        )
    if subset is not None:
        candidates = [type_dir / subset]
    else:
        candidates = [p for p in sorted(type_dir.iterdir()) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No DISCS subsets under {type_dir}. "
            f"Available subsets: {sorted(p.name for p in type_dir.iterdir() if p.is_dir())}"
        )

    records: list[dict] = []
    for cand in candidates:
        records.extend(_read_manifest(cand))
    # The single subset case keeps a unique subset_dir; multi-subset returns the
    # parent type_dir to make traceback hints predictable.
    return (candidates[0] if len(candidates) == 1 else type_dir), records


def _load_graphs_from_manifest(
    subset_dir: Path,
    records: list[dict],
    *,
    limit: int | None = None,
) -> tuple[list[nx.Graph], np.ndarray, list[dict]]:
    if limit is not None:
        records = records[:limit]
    graphs: list[nx.Graph] = []
    bests: list[float] = []
    for rec in records:
        # When records come from a multi-subset query the per-record subset
        # directory differs; reconstruct it relative to the unified root.
        per_dir = subset_dir if subset_dir.name == rec["subset"] else subset_dir / rec["subset"]
        graph_path = per_dir / rec["file"]
        with open(graph_path, "rb") as fh:
            graphs.append(pickle.load(fh))
        bests.append(rec["best_known"] if rec["best_known"] is not None else np.nan)
    return graphs, np.asarray(bests, dtype=np.float64), records


def discs_maxcut(
    graph_type: str = "ba",
    subset: str | None = None,
    *,
    device: str | torch.device = "cpu",
    limit: int | None = None,
    root: str | os.PathLike | None = None,
    parallel: bool = False,
) -> DiscsBenchmark:
    """Load DISCS MaxCut instances.

    Parameters
    ----------
    graph_type
        ``"ba"`` (Barabási–Albert), ``"er"`` (Erdős–Rényi), or ``"optsicom"``.
    subset
        Subset name within the graph type (e.g. ``"200"``, ``"1024-1100"``,
        ``"b"``). If ``None``, every subset under ``graph_type`` is loaded
        and concatenated.
    device, limit, root
        Standard knobs (limit caps total instances; root overrides
        ``QQA_DATA_DIR/discs``).
    parallel
        If ``True``, pack every graph into a single :class:`MaxCutInstance`
        problem (padded to ``max_node`` and masked) so ``qqa.anneal`` solves
        them all in one batched call. The returned ``DiscsBenchmark.problems``
        is then a length-1 list. See :doc:`docs/problems` for details.
    """
    sdir, records = _resolve_discs_subset("maxcut", graph_type, subset, root=root)
    graphs, bests, recs = _load_graphs_from_manifest(sdir, records, limit=limit)
    if parallel:
        problems = [MaxCutInstance(graphs, device=device)]
    else:
        problems = [MaxCut(g, device=device) for g in graphs]
    return DiscsBenchmark(problems, bests, recs, sdir)


def discs_mis(
    graph_type: str = "satlib",
    subset: str | None = None,
    *,
    penalty: float = 2.0,
    device: str | torch.device = "cpu",
    limit: int | None = None,
    root: str | os.PathLike | None = None,
    parallel: bool = False,
) -> DiscsBenchmark:
    """Load DISCS Maximum Independent Set instances.

    ``graph_type`` ∈ ``{"satlib", "er", "er_density"}``.

    Set ``parallel=True`` to fold every instance into a single
    :class:`MaximumIndependentSetInstance` (one batched solve via
    ``qqa.anneal``). All instances share ``penalty``; for heterogeneous
    penalties build the ``Instance`` class manually.
    """
    sdir, records = _resolve_discs_subset("mis", graph_type, subset, root=root)
    graphs, bests, recs = _load_graphs_from_manifest(sdir, records, limit=limit)
    if parallel:
        problems = [MaximumIndependentSetInstance(graphs, penalty=penalty, device=device)]
    else:
        problems = [MaximumIndependentSet(g, penalty=penalty, device=device) for g in graphs]
    return DiscsBenchmark(problems, bests, recs, sdir)


def discs_maxclique(
    graph_type: str = "rb",
    subset: str | None = None,
    *,
    penalty: float = 3.0,
    device: str | torch.device = "cpu",
    limit: int | None = None,
    root: str | os.PathLike | None = None,
    parallel: bool = False,
) -> DiscsBenchmark:
    """Load DISCS Maximum Clique instances. ``graph_type`` ∈ ``{"rb", "twitter"}``.

    Set ``parallel=True`` to fold every instance into a single
    :class:`MaxCliqueInstance`.
    """
    sdir, records = _resolve_discs_subset("maxclique", graph_type, subset, root=root)
    graphs, bests, recs = _load_graphs_from_manifest(sdir, records, limit=limit)
    if parallel:
        problems = [MaxCliqueInstance(graphs, penalty=penalty, device=device)]
    else:
        problems = [MaxClique(g, penalty=penalty, device=device) for g in graphs]
    return DiscsBenchmark(problems, bests, recs, sdir)


def discs_normcut(
    graph_type: str = "nets",
    subset: str | None = None,
    *,
    num_category: int = 2,
    device: str | torch.device = "cpu",
    limit: int | None = None,
    root: str | os.PathLike | None = None,
) -> DiscsBenchmark:
    """Load DISCS Normalized Cut instances.

    ``graph_type`` ∈ ``{"nets", "gap_rand"}``.
    Note: DISCS itself ships only ``best_known=None`` for NormCut (it is a
    minimisation with no known optimum); approximation ratios in benchmarks
    are typically reported relative to DISCS' best-found values.
    """
    sdir, records = _resolve_discs_subset("normcut", graph_type, subset, root=root)
    graphs, bests, recs = _load_graphs_from_manifest(sdir, records, limit=limit)
    problems = [NormalizedCut(g, num_category=num_category, device=device) for g in graphs]
    return DiscsBenchmark(problems, bests, recs, sdir)


def list_discs_subsets(root: str | os.PathLike | None = None) -> dict[str, dict[str, list[str]]]:
    """Return ``{problem: {graph_type: [subset, ...]}}`` for the unified suite."""
    base = Path(root).expanduser().resolve() if root is not None else _default_discs_root()
    out: dict[str, dict[str, list[str]]] = {}
    if not base.is_dir():
        return out
    for prob_dir in sorted(base.iterdir()):
        if not prob_dir.is_dir() or prob_dir.name.startswith("_"):
            continue
        out[prob_dir.name] = {}
        for type_dir in sorted(prob_dir.iterdir()):
            if not type_dir.is_dir():
                continue
            subsets = sorted(s.name for s in type_dir.iterdir() if s.is_dir())
            out[prob_dir.name][type_dir.name] = subsets
    return out
