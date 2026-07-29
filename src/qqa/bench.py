"""One-line benchmark API for third-party users.

The implementation lives in :mod:`qqa.benchmarking`, which is shipped in
both wheels and source distributions.  The files under ``scripts/`` are thin
cluster-friendly wrappers around the same implementation, so Python and CLI
entry points have identical behaviour after ``pip install qqa``.

Typical flow::

    from qqa import bench

    bench.list_suites()               # {'maxcut': {'er': ['200', '400']}, ...}
    payload = bench.run(                # solve + return a dict
        "mis-satlib",
        backend="qqa",
        instances=3,
        output="bench_results/mine.json",
    )
    bench.plot(                         # render the benchmark report image
        ["bench_results/mine.json", "bench_results/sa.json"],
        labels=["QQA", "SA"],
        output="bench_results/report.png",
    )

CLI equivalents (``qqa bench-run``, ``qqa bench-plot``, and
``qqa bench-list``) are documented in the CLI reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_RESULTS_DIR",
    "list_suites",
    "resolve_suite",
    "run",
    "plot",
    "bench_discs_main",
    "plot_benchmarks_main",
]

#: Default directory for benchmark artifacts (``results.json`` + plots).
#:
#: We keep it at the repository root so both the raw JSON and the
#: rendered figure can be committed / shared / referenced from a CI
#: artefact in a single place.
DEFAULT_RESULTS_DIR = Path("bench_results")


def _load_bench_discs():
    """Lazily import the packaged runner.

    Keeping this tiny seam preserves the existing test/mocking contract while
    avoiding import-time dataset discovery for users who only import
    :mod:`qqa.bench`.
    """
    from qqa.benchmarking import runner

    return runner


def _load_plot_benchmarks():
    """Lazily import the packaged report renderer."""
    from qqa.benchmarking import plotting

    return plotting


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def list_suites() -> dict[str, dict[str, list[str]]]:
    """Return ``{family: {graph_type: [subset, ...]}}`` for the data on disk.

    Raises
    ------
    SystemExit
        If no benchmark data has been downloaded yet. Run
        ``make bench-all-setup`` (or ``./scripts/setup_benchmarks.sh``)
        first.
    """
    mod = _load_bench_discs()
    return mod._build_catalog()


def resolve_suite(suite: str) -> list[tuple[str, str, str | None]]:
    """Expand a suite identifier to a list of ``(family, graph_type, subset)``."""
    mod = _load_bench_discs()
    return mod._resolve_suite(suite)


def run(
    suite: str = "all",
    *,
    backend: str = "qqa",
    instances: int | None = None,
    sol_size: int = 20,
    num_epochs: int = 500,
    device: str = "auto",
    seed: int = 0,
    output: Path | str | None = None,
    parallel: bool = False,
    penalty: float | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run a benchmark suite and return the payload that would be written to JSON.

    Parameters
    ----------
    suite:
        Same syntax as ``scripts/bench_discs.py --suite`` —
        ``all``, a family (``mis``, ``coloring``, ``ea3d`` ...),
        a ``<family>-<graph_type>`` prefix or a fully qualified
        ``<family>-<graph_type>-<subset>`` triple.
    backend:
        ``qqa`` (default), ``sa`` or ``pa``.
    instances:
        Per-subset cap. Omit to run every instance on disk.
    output:
        If set, the JSON payload is dumped here. When the path is
        relative we resolve it under :data:`DEFAULT_RESULTS_DIR` so
        third parties can keep results together without thinking about
        where to put them.
    parallel:
        Only for ``qqa`` + ``mis``/``maxcut``/``maxclique`` — pack every
        instance of a subset into a single batched ``*Instance`` problem
        and solve in one ``qqa.anneal`` call.
    extra_args:
        Raw CLI arguments forwarded to ``bench_discs.py`` (e.g.
        ``['--learning-rate', '1.0']`` to override PQQA hyperparameters).

    Returns
    -------
    dict
        The payload dict (same shape as the JSON file):
        ``{"backend", "suite", "device", "sol_size", "num_epochs",
        "seed", "parallel", "qqa_hp", "results": [...]}``.
    """
    out_path = _normalise_output(output)
    mod = _load_bench_discs()

    argv: list[str] = [
        "--suite",
        suite,
        "--backend",
        backend,
        "--sol-size",
        str(sol_size),
        "--num-epochs",
        str(num_epochs),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if instances is not None:
        argv += ["--instances", str(instances)]
    if parallel:
        argv += ["--parallel"]
    if penalty is not None:
        argv += ["--penalty", str(penalty)]
    if out_path is not None:
        argv += ["--output", str(out_path)]
    if extra_args:
        argv += list(extra_args)

    rc = mod.main(argv)
    if rc != 0:
        raise SystemExit(f"qqa.bench.run: suite '{suite}' failed with rc={rc}")

    if out_path is not None and out_path.is_file():
        with out_path.open() as fh:
            return json.load(fh)
    # If no output path was requested we still want to return something
    # meaningful; re-run the resolver and warn the user that no JSON was
    # produced.
    return {"suite": suite, "results": []}


def plot(
    results: list[Path | str] | Path | str,
    *,
    labels: list[str] | None = None,
    output: Path | str | None = None,
    title: str | None = None,
    theme: str = "light",
    dpi: int = 160,
    fmt: str | None = None,
) -> Path | None:
    """Render a polished benchmark-report image from ``bench_discs.py`` JSON.

    Parameters
    ----------
    results:
        Either a single path or a list of paths to ``results.json``
        files. Up to ~5 overlays render cleanly; beyond that the bars
        get crowded.
    labels:
        Optional per-report label; defaults to ``"<backend>:<suite>"``.
    output:
        Where to write the image. Relative paths are resolved under
        :data:`DEFAULT_RESULTS_DIR`. Extension controls the format
        (``.png``, ``.svg``, ``.pdf``); use ``fmt`` to force.
    theme:
        ``"light"`` (default) or ``"dark"``.

    Returns
    -------
    Path or None
        The written file when ``output`` is set.
    """
    mod = _load_plot_benchmarks()
    if isinstance(results, (str, Path)):
        results = [results]
    out = _normalise_output(output)

    argv: list[str] = [str(p) for p in results]
    if labels:
        argv += ["--labels", *labels]
    if out is not None:
        argv += ["--output", str(out)]
    if fmt:
        argv += ["--format", fmt]
    argv += ["--dpi", str(dpi), "--theme", theme]
    if title:
        argv += ["--title", title]

    rc = mod.main(argv)
    if rc != 0:
        raise SystemExit(f"qqa.bench.plot: render failed with rc={rc}")
    return out


def bench_discs_main(argv: list[str] | None = None) -> int:
    """Run the packaged benchmark CLI implementation."""
    return _load_bench_discs().main(argv)


def plot_benchmarks_main(argv: list[str] | None = None) -> int:
    """Run the packaged benchmark plotting CLI implementation."""
    return _load_plot_benchmarks().main(argv)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _normalise_output(output: Path | str | None) -> Path | None:
    """Prepend :data:`DEFAULT_RESULTS_DIR` only for bare filenames.

    - ``"results.json"`` → ``bench_results/results.json`` (convenience
      shortcut for the "I just want a report next to the runs" case).
    - ``"data/fig/bench.png"`` → ``data/fig/bench.png`` (honor any
      explicit directory the caller passed; prepending ``bench_results/``
      would be surprising and creates confusing nested trees).
    - ``"./x.png"`` / ``"~/x.png"`` / ``"/abs/x.png"`` → kept as-is.
    """
    if output is None:
        return None
    p = Path(output)
    has_dir = p.parent != Path("")
    if not p.is_absolute() and not str(p).startswith((".", "~")) and not has_dir:
        p = DEFAULT_RESULTS_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
