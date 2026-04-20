"""Run the DISCS combinatorial-optimization benchmark suite.

Examples
--------
Smoke (3 instances of the SATLIB MIS subset, default ``qqa.anneal`` backend)::

    python scripts/bench_discs.py --suite mis-satlib --instances 3

All available subsets, save JSON results::

    python scripts/bench_discs.py --suite all --output results.json

Single subset with the SA baseline::

    python scripts/bench_discs.py --suite maxcut-ba-200 --backend sa

The script prints a per-instance objective + approximation ratio and an
aggregate summary (mean ratio, completion rate, runtime).

Suite identifiers
-----------------
``--suite`` accepts:

* ``all``                            — every subset under ``data/discs/``
* ``<problem>``                      — every subset of one problem (mis, maxcut, ...)
* ``<problem>-<graph_type>``         — every subset of one (problem, type)
* ``<problem>-<graph_type>-<subset>`` — exactly one subset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import qqa
from qqa import datasets

# `scripts/` is on sys.path when this file is invoked as `python scripts/...`,
# so a bare-name import works without packaging.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_common as bench  # noqa: E402

LOG = bench.setup_logging(verbose=False, name="bench_discs")


# --------------------------------------------------------------------------- #
# Suite resolution                                                            #
# --------------------------------------------------------------------------- #


_PROBLEM_LOADER = {
    "maxcut": datasets.discs_maxcut,
    "mis": datasets.discs_mis,
    "maxclique": datasets.discs_maxclique,
    "normcut": datasets.discs_normcut,
}


def _resolve_suite(suite: str) -> list[tuple[str, str, str | None]]:
    """Return a list of ``(problem, graph_type, subset)`` triples."""
    catalog = datasets.list_discs_subsets()
    if not catalog:
        raise SystemExit("No DISCS data found. Run `./scripts/setup_discs_data.sh` first.")

    out: list[tuple[str, str, str | None]] = []
    if suite == "all":
        for prob, types in catalog.items():
            for gtype, subsets in types.items():
                for sub in subsets:
                    out.append((prob, gtype, sub))
        return out

    parts = suite.split("-")
    prob = parts[0]
    if prob not in catalog:
        raise SystemExit(f"Unknown suite '{suite}'. Available problems: {sorted(catalog)}")
    if len(parts) == 1:
        for gtype, subsets in catalog[prob].items():
            for sub in subsets:
                out.append((prob, gtype, sub))
        return out

    # parts[1:] is graph_type and possibly subset.
    # The subset name may itself contain dashes (e.g. ``ba-1024-1100``), so we
    # match by trying the longest valid prefix first.
    rest = "-".join(parts[1:])
    types = catalog[prob]
    for gtype in types:
        if rest == gtype:
            for sub in types[gtype]:
                out.append((prob, gtype, sub))
            return out
        prefix = gtype + "-"
        if rest.startswith(prefix):
            sub = rest[len(prefix) :]
            if sub in types[gtype]:
                out.append((prob, gtype, sub))
                return out

    raise SystemExit(
        f"Could not resolve suite '{suite}'. Available under {prob}: "
        + ", ".join(f"{prob}-{g}-{s}" for g, ss in types.items() for s in ss)
    )


# --------------------------------------------------------------------------- #
# Solvers                                                                     #
# --------------------------------------------------------------------------- #


def _run_qqa_anneal(problem, **kwargs):
    """Thin alias around :func:`bench.run_qqa_anneal` for backward compat.

    Kept so that any external test that monkey-patches
    ``bench_discs._run_qqa_anneal`` continues to work; new code should
    call :func:`bench.run_qqa_anneal` directly.
    """
    return bench.run_qqa_anneal(problem, **kwargs)


def _run_sa(problem, *, device: str, sol_size: int, num_epochs: int, **_unused):
    return qqa.simulated_annealing(
        problem,
        sol_size=sol_size,
        num_sweeps=num_epochs,
        device=device,
        verbose=False,
    )


def _run_pa(problem, *, device: str, sol_size: int, num_epochs: int, **_unused):
    return qqa.population_annealing(
        problem,
        population_size=sol_size,
        num_temps=max(10, num_epochs // 50),
        sweeps_per_temp=10,
        device=device,
        verbose=False,
    )


_BACKENDS = {
    "qqa": _run_qqa_anneal,
    "sa": _run_sa,
    "pa": _run_pa,
}


# --------------------------------------------------------------------------- #
# Per-problem objective extraction                                            #
# --------------------------------------------------------------------------- #


def _objective_and_feasibility(problem, result, problem_kind: str) -> tuple[float, bool]:
    """Return the *human-readable* CO objective and a feasibility flag.

    qqa loss conventions (with penalised QUBO formulations):

    * **MIS** ``loss = -|S| + penalty * (#violated edges)``. We MUST use
      :meth:`score_summary` (or recompute |S| only over feasible vertices)
      because ``-loss`` overestimates ``|S|`` whenever an edge constraint
      is violated.
    * **MaxClique** ``loss = -|S| + penalty * (#missing pairs)``. Same
      caveat as MIS — use :meth:`score_summary`.
    * **MaxCut** ``loss = -cut_weight``. Unconstrained, ``cut = -loss`` is
      always exact; we still call :meth:`score_summary` to keep the JSON
      output consistent and pick the best replica directly.
    * **NormCut** is already in minimisation form on the discrete projection;
      we run :meth:`discrete_ncut` on the projected best replica.
    """
    if problem_kind == "normcut":
        with torch.no_grad():
            x_disc = problem.relaxation.project(result.best_sol.unsqueeze(0))
            ncut = float(problem.discrete_ncut(x_disc).item())
        return ncut, True

    # MIS / MaxCut / MaxClique: ask the problem for the feasibility-aware metric.
    summary = problem.score_summary(result.best_sol)
    return float(summary.get("value", -result.best_obj)), bool(summary.get("feasible", True))


def _objective_from_result(problem, result, problem_kind: str) -> float:
    """Backwards-compatible wrapper used by tests."""
    obj, _ = _objective_and_feasibility(problem, result, problem_kind)
    return obj


def _per_instance_objectives_and_feasibility(
    problem, result, problem_kind: str
) -> tuple[list[float], list[bool]]:
    """Vectorised counterpart of :func:`_objective_and_feasibility`.

    Returns ``(objectives, feasibles)`` with one entry per instance for the
    batched ``*Instance`` problem classes (``MaximumIndependentSetInstance``,
    ``MaxCutInstance``, ``MaxCliqueInstance``). NormCut has no batched class
    yet, so this path is only used for the QUBO families.
    """
    if problem_kind not in {"mis", "maxcut", "maxclique"}:
        raise ValueError(f"--parallel does not support problem '{problem_kind}'.")
    summary = problem.score_summary(result.best_sol)
    values = list(summary["value"].tolist())
    feasibles = [bool(f) for f in summary["feasible"].tolist()]
    return [float(v) for v in values], feasibles


def _approx_ratio(objective: float, best_known: float, kind: str) -> float | None:
    """Approximation ratio relative to the published best (NaN-safe)."""
    if best_known is None or np.isnan(best_known) or best_known == 0:
        return None
    if kind in {"mis", "maxcut", "maxclique"}:
        return objective / best_known  # higher is better
    if kind == "normcut":
        return best_known / objective  # lower is better
    return None


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suite", default="all", help="suite identifier (see docstring)")
    p.add_argument("--backend", default="qqa", choices=list(_BACKENDS))
    p.add_argument("--instances", type=int, default=None, help="cap per subset")
    p.add_argument("--sol-size", type=int, default=20, help="parallel population")
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument("--device", default="auto", help="cpu / cuda / auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None, help="JSON output path")
    p.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Pack each subset's instances into a single batched *Instance problem "
            "and solve them in one qqa.anneal call. Padding to the largest n is "
            "handled automatically. Requires --backend qqa and is only supported "
            "for mis / maxcut / maxclique."
        ),
    )
    # PQQA hyperparameters — defaults tuned to be a reasonable smoke. To
    # reproduce the NeurIPS-2024 paper, use the make target or pass:
    #   --learning-rate 1.0 --temp 1e-3 --curve-rate 4
    #   --gamma-min -2 --gamma-max 0.1 --div-param 0.2
    bench.add_qqa_hp_args(p)
    p.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="Constraint penalty (MIS/MaxClique loaders). Paper uses 2.0 for MIS.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.parallel and args.backend != "qqa":
        raise SystemExit("--parallel currently requires --backend qqa.")

    bench.setup_logging(args.verbose, name="bench_discs")
    qqa.fix_seed(args.seed)
    bench.setup_device(args)
    LOG.info("device=%s backend=%s", args.device, args.backend)

    triples = _resolve_suite(args.suite)
    LOG.info("resolved suite '%s' -> %d (problem, type, subset)(s)", args.suite, len(triples))

    runner = _BACKENDS[args.backend]
    runner_kwargs = {
        "device": args.device,
        "sol_size": args.sol_size,
        "num_epochs": args.num_epochs,
        **bench.qqa_hp_kwargs(args),
    }
    LOG.info(
        "qqa hp: lr=%g temp=%g curve_rate=%d gamma=[%g,%g] div=%g",
        runner_kwargs["learning_rate"], runner_kwargs["temp"], runner_kwargs["curve_rate"],
        runner_kwargs["gamma_min"], runner_kwargs["gamma_max"], runner_kwargs["div_param"],
    )
    all_results: list[dict[str, Any]] = []

    for prob, gtype, sub in triples:
        LOG.info("== %s / %s / %s ==", prob, gtype, sub)
        loader = _PROBLEM_LOADER[prob]
        load_kwargs: dict[str, Any] = {
            "graph_type": gtype,
            "subset": sub,
            "device": args.device,
            "limit": args.instances,
        }
        if args.penalty is not None and prob in {"mis", "maxclique"}:
            load_kwargs["penalty"] = args.penalty
        use_parallel = args.parallel and prob in {"mis", "maxcut", "maxclique"}
        if args.parallel and not use_parallel:
            LOG.warning("  --parallel ignored for problem '%s' (no batched class).", prob)
        if use_parallel:
            load_kwargs["parallel"] = True
        # Local name `dataset` to avoid shadowing the module-level `bench`
        # (which would otherwise turn `bench.add_qqa_hp_args` into an
        # UnboundLocalError, since Python resolves names statically per
        # function scope).
        dataset = loader(**load_kwargs)
        LOG.info(
            "  loaded %d instances%s",
            len(dataset.manifest),
            "  [parallel batched-instance solve]" if use_parallel else "",
        )

        per_instance: list[dict[str, Any]] = []
        if use_parallel:
            problem = dataset.problems[0]
            t0 = time.time()
            result = runner(problem, **runner_kwargs)
            wall_total = time.time() - t0
            objs, feasibles = _per_instance_objectives_and_feasibility(problem, result, prob)
            wall_per = wall_total / max(1, len(objs))
            for i, (obj, feasible) in enumerate(zip(objs, feasibles, strict=True)):
                best = float(dataset.best_known[i]) if not np.isnan(dataset.best_known[i]) else None
                ratio = _approx_ratio(obj, best, prob)
                per_instance.append(
                    {
                        "instance": i,
                        "id": dataset.manifest[i]["id"],
                        "num_nodes": dataset.manifest[i]["num_nodes"],
                        "objective": obj,
                        "feasible": feasible,
                        "best_known": best,
                        "ratio": ratio,
                        "wall_s": wall_per,
                    }
                )
            LOG.info(
                "  [parallel] %d instances solved in %.2fs (%.3fs/inst), feas=%d/%d",
                len(objs),
                wall_total,
                wall_per,
                sum(feasibles),
                len(objs),
            )
        else:
            for i, problem in enumerate(dataset.problems):
                t0 = time.time()
                result = runner(problem, **runner_kwargs)
                wall = time.time() - t0
                obj, feasible = _objective_and_feasibility(problem, result, prob)
                best = float(dataset.best_known[i]) if not np.isnan(dataset.best_known[i]) else None
                ratio = _approx_ratio(obj, best, prob)
                per_instance.append(
                    {
                        "instance": i,
                        "id": dataset.manifest[i]["id"],
                        "num_nodes": dataset.manifest[i]["num_nodes"],
                        "objective": obj,
                        "feasible": feasible,
                        "best_known": best,
                        "ratio": ratio,
                        "wall_s": wall,
                    }
                )
                LOG.info(
                    "  [%2d] obj=%.4f  feas=%s  best=%s  ratio=%s  t=%.2fs",
                    i,
                    obj,
                    "Y" if feasible else "N",
                    f"{best:.4f}" if best is not None else "NA",
                    f"{ratio:.4f}" if ratio is not None else "NA",
                    wall,
                )

        ratios = [r["ratio"] for r in per_instance if r["ratio"] is not None]
        n_feas = sum(1 for r in per_instance if r["feasible"])
        agg = {
            "problem": prob,
            "graph_type": gtype,
            "subset": sub,
            "n": len(per_instance),
            "n_feasible": n_feas,
            "mean_objective": float(np.mean([r["objective"] for r in per_instance])),
            "mean_ratio": float(np.mean(ratios)) if ratios else None,
            "min_ratio": float(np.min(ratios)) if ratios else None,
            "max_ratio": float(np.max(ratios)) if ratios else None,
            "total_wall_s": float(sum(r["wall_s"] for r in per_instance)),
            "instances": per_instance,
        }
        all_results.append(agg)
        LOG.info(
            "  -> mean_obj=%.4f  mean_ratio=%s  total=%.2fs",
            agg["mean_objective"],
            f"{agg['mean_ratio']:.4f}" if agg["mean_ratio"] is not None else "NA",
            agg["total_wall_s"],
        )

    LOG.info("== SUMMARY ==")
    for agg in all_results:
        LOG.info(
            "  %-9s %-10s %-18s n=%d feas=%d/%d mean_obj=%.3f mean_ratio=%s",
            agg["problem"],
            agg["graph_type"],
            agg["subset"],
            agg["n"],
            agg["n_feasible"],
            agg["n"],
            agg["mean_objective"],
            f"{agg['mean_ratio']:.4f}" if agg["mean_ratio"] is not None else "NA",
        )

    if args.output:
        payload = {
            "backend": args.backend,
            "suite": args.suite,
            "device": args.device,
            "sol_size": args.sol_size,
            "num_epochs": args.num_epochs,
            "seed": args.seed,
            "parallel": bool(args.parallel),
            "qqa_hp": {**bench.qqa_hp_kwargs(args), "penalty": args.penalty},
            "results": all_results,
        }
        bench.dump_results_json(args.output, payload)
        LOG.info("wrote %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
