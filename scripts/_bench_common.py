"""Shared CLI / runner helpers for the benchmark scripts.

Both ``scripts/bench_discs.py`` and ``scripts/bench_factorization.py``
expose the same QQA hyperparameter knobs, do the same JSON-output
plumbing, and need the same auto-CPU/CUDA selection.  Centralising those
bits keeps the per-benchmark scripts focused on what is actually unique
about each suite (data loading, per-instance scoring) and means a single
edit propagates to every benchmark.

Public surface
--------------
``add_qqa_hp_args(parser)``
    Append the standard QQA / paper-relevant hyperparameter flags
    (``--learning-rate``, ``--temp``, ``--curve-rate``,
    ``--gamma-min/--gamma-max``, ``--div-param``) to an
    ``argparse.ArgumentParser``.  Defaults match the PQQA paper
    (Ichikawa, NeurIPS 2024) so a bare ``--learning-rate 0.1`` already
    reproduces the SATLIB MIS recipe.

``qqa_hp_kwargs(args)``
    Extract those flags from a parsed ``argparse.Namespace`` into a
    dict suitable for passing as ``**kwargs`` to
    :func:`run_qqa_anneal` (or to a custom runner with the same
    contract).

``run_qqa_anneal(problem, *, device, sol_size, num_epochs, **hp)``
    Single, opinionated wrapper around ``qqa.anneal`` so every bench
    runs through the same code path.  It always disables history
    recording (the JSON output is the source of truth) and sets a
    sensible ``check_interval``.

``setup_device(args)`` / ``setup_logging(verbose)`` / ``dump_results_json``
    Wrappers around the obvious one-liners with the small things that
    are easy to forget (e.g. creating the output parent directory).

These are intentionally *not* re-exported from the top-level ``qqa``
package: they are CLI-script helpers, not part of the library API.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

import qqa

# Paper-relevant hyperparameter defaults (PQQA, Ichikawa NeurIPS 2024).
# These are conservative "smoke" defaults; the per-script callers can
# override them via the Makefile presets (`bench-discs-paper`) or by
# passing explicit CLI flags.
_QQA_HP_DEFAULTS = {
    "learning_rate": 1.0,   # paper sweeps {1, 0.1, 0.01}
    "temp": 1e-3,           # paper: 0.001 fixed
    "curve_rate": 4,        # paper: 4
    "gamma_min": -2.0,      # paper: -2 (some MaxCut: -5/-20)
    "gamma_max": 0.1,       # paper: 0.1
    "div_param": 0.2,       # paper Ablation Fig 6
}


_HP_HELP = {
    "learning_rate": "AdamW base learning rate (paper: {1, 0.1, 0.01}).",
    "temp": "Langevin temperature T (paper: 0.001 fixed).",
    "curve_rate": "alpha-entropy exponent (paper: 4).",
    "gamma_min": "initial bias schedule (paper: -2; -5/-20 for some MaxCut).",
    "gamma_max": "final bias schedule (paper: 0.1).",
    "div_param": "diversity coupling alpha (paper Ablation Fig 6).",
}


def add_qqa_hp_args(parser: argparse.ArgumentParser) -> None:
    """Append the standard QQA / paper-relevant flags to ``parser``.

    Idempotent: calling this twice on the same parser is a no-op for
    flags already present (argparse will raise on duplicates, so the
    expectation is exactly one call per parser).
    """
    parser.add_argument(
        "--learning-rate", type=float,
        default=_QQA_HP_DEFAULTS["learning_rate"], help=_HP_HELP["learning_rate"],
    )
    parser.add_argument(
        "--temp", type=float,
        default=_QQA_HP_DEFAULTS["temp"], help=_HP_HELP["temp"],
    )
    parser.add_argument(
        "--curve-rate", type=int,
        default=_QQA_HP_DEFAULTS["curve_rate"], help=_HP_HELP["curve_rate"],
    )
    parser.add_argument(
        "--gamma-min", type=float,
        default=_QQA_HP_DEFAULTS["gamma_min"], help=_HP_HELP["gamma_min"],
    )
    parser.add_argument(
        "--gamma-max", type=float,
        default=_QQA_HP_DEFAULTS["gamma_max"], help=_HP_HELP["gamma_max"],
    )
    parser.add_argument(
        "--div-param", type=float,
        default=_QQA_HP_DEFAULTS["div_param"], help=_HP_HELP["div_param"],
    )


def qqa_hp_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Extract the QQA HP kwargs from a parsed namespace.

    The returned dict has the *exact* keyword names that
    :func:`run_qqa_anneal` accepts, so the typical caller writes::

        result = run_qqa_anneal(prob, device=..., sol_size=..., num_epochs=...,
                                **qqa_hp_kwargs(args))

    and never has to remember the spelling of each flag.
    """
    return {
        "learning_rate": float(args.learning_rate),
        "temp": float(args.temp),
        "curve_rate": int(args.curve_rate),
        "gamma_min": float(args.gamma_min),
        "gamma_max": float(args.gamma_max),
        "div_param": float(args.div_param),
    }


def run_qqa_anneal(
    problem,
    *,
    device: str | torch.device,
    sol_size: int,
    num_epochs: int,
    learning_rate: float,
    temp: float,
    curve_rate: int,
    gamma_min: float,
    gamma_max: float,
    div_param: float,
):
    """Single opinionated entry-point used by every bench script.

    Wraps :func:`qqa.anneal` with:

    * a paper-aligned ``LinearBGSchedule`` built from
      ``(gamma_min, gamma_max)``,
    * ``check_interval = num_epochs // 4`` (so JSON output sees four
      progress checkpoints),
    * ``record_history=False`` and ``verbose=False`` (the JSON written
      by the bench script is the canonical artefact).

    The full set of accepted kwargs is *exactly* what
    :func:`qqa_hp_kwargs` returns, plus ``device``, ``sol_size`` and
    ``num_epochs`` — keep them in sync if you change either signature.
    """
    return qqa.anneal(
        problem,
        sol_size=sol_size,
        learning_rate=learning_rate,
        temp=temp,
        schedule=qqa.LinearBGSchedule(min_bg=gamma_min, max_bg=gamma_max),
        curve_rate=curve_rate,
        div_param=div_param,
        num_epochs=num_epochs,
        check_interval=max(1, num_epochs // 4),
        device=device,
        verbose=False,
        record_history=False,
    )


def setup_device(args: argparse.Namespace) -> str:
    """Resolve ``args.device == 'auto'`` to ``cuda`` or ``cpu``.

    Mutates ``args.device`` in place (so subsequent code can read the
    resolved value) and also returns it for convenience.
    """
    if getattr(args, "device", None) == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args.device


def setup_logging(verbose: bool, *, name: str | None = None) -> logging.Logger:
    """Configure root logger + return a named child for the caller."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # safe to call from notebook / repeated entry points
    )
    return logging.getLogger(name) if name else logging.getLogger()


def dump_results_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as pretty-JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


__all__ = [
    "add_qqa_hp_args",
    "qqa_hp_kwargs",
    "run_qqa_anneal",
    "setup_device",
    "setup_logging",
    "dump_results_json",
]
