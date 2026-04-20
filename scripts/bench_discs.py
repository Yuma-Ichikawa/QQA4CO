"""Run combinatorial-optimisation benchmarks against QQA / SA / PA.

This script drives every benchmark family shipped with QQA4CO:

* **DISCS** (Goshvadi et al. NeurIPS 2023, repackaged):
  ``mis-*``, ``maxcut-*``, ``maxclique-*``, ``normcut-*``.
* **PQQA paper** (Ichikawa & Arai NeurIPS 2024) extras not in DISCS:
  ``mis-rrg-*`` (MIS on d-regular random graphs),
  ``coloring-*`` (Graph coloring, Trick 2002 subset),
  ``balanced-partition-*`` (balanced k-way cut on the DISCS nets graphs).
* **Physics stress test** requested on top of the paper:
  ``ea3d-*`` (3D Edwards-Anderson spin glass, Gaussian and ±J).

Examples
--------
Smoke (3 instances of the SATLIB MIS subset, default ``qqa.anneal`` backend)::

    python scripts/bench_discs.py --suite mis-satlib --instances 3

Smoke for the 3D EA spin glass (L=4 Gaussian, 5 instances)::

    python scripts/bench_discs.py --suite ea3d-gaussian-L4 --instances 5

All available subsets, save JSON results::

    python scripts/bench_discs.py --suite all --output results.json

Single subset with the SA baseline::

    python scripts/bench_discs.py --suite maxcut-ba-200 --backend sa

The script prints a per-instance objective + approximation ratio and an
aggregate summary (mean ratio, completion rate, runtime).

Suite identifiers
-----------------
``--suite`` accepts:

* ``all``                            — every subset under ``data/`` (DISCS + extras)
* ``<family>``                       — every subset of one family
  (``mis``, ``maxcut``, ``mis-rrg``, ``ea3d``, ``coloring``,
  ``balanced-partition``, ...)
* ``<family>-<graph_type>``          — every subset of one (family, type)
* ``<family>-<graph_type>-<subset>`` — exactly one subset

Family names may themselves contain a hyphen (``mis-rrg``,
``balanced-partition``); resolution is longest-prefix-first.
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


# Loaders are wrapped so every one has the same
#   fn(graph_type, subset, *, device, limit, **extras) -> DiscsBenchmark
# signature. ``extras`` can carry family-specific knobs (``penalty``,
# ``num_category``, ...). The wrappers translate ``graph_type`` into the
# loader's native first positional argument.
def _load_mis_rrg(graph_type, subset, **kw):
    kw.pop("parallel", None)  # mis-rrg has no batched class
    return datasets.mis_rrg(subset=subset, **kw)


def _load_coloring(graph_type, subset, **kw):
    kw.pop("parallel", None)
    kw.pop("penalty", None)
    return datasets.coloring(graph_type=graph_type, subset=subset, **kw)


def _load_ea3d(graph_type, subset, **kw):
    kw.pop("parallel", None)
    kw.pop("penalty", None)
    return datasets.ea3d(dist=graph_type, subset=subset, **kw)


def _load_balanced(graph_type, subset, **kw):
    kw.pop("parallel", None)
    kw.pop("penalty", None)
    return datasets.balanced_partition(graph_type=graph_type, subset=subset, **kw)


def _load_gset(graph_type, subset, **kw):
    kw.pop("parallel", None)
    kw.pop("penalty", None)
    # gset uses a flat layout (data/gset/<subset>/manifest.jsonl); graph_type
    # carries the ``standard`` tag for parity with the rest of the catalog.
    return datasets.gset(subset=graph_type or subset or "standard", **kw)


_PROBLEM_LOADER = {
    "maxcut": datasets.discs_maxcut,
    "mis": datasets.discs_mis,
    "maxclique": datasets.discs_maxclique,
    "normcut": datasets.discs_normcut,
    "mis-rrg": _load_mis_rrg,
    "coloring": _load_coloring,
    "ea3d": _load_ea3d,
    "balanced-partition": _load_balanced,
    "gset": _load_gset,
}


def _build_catalog() -> dict[str, dict[str, list[str]]]:
    """Unified ``{family: {graph_type: [subset, ...]}}`` catalog.

    Collapses DISCS layout + the new generate_* layouts under ``data/``.
    A single-level family (where the leaf with ``manifest.jsonl`` is
    directly under ``data/<family>/``) is recorded as
    ``{"": ["<subset>"]}`` so downstream dispatch keeps a consistent
    3-slot triple even when the family has no ``graph_type`` layer.
    """
    catalog: dict[str, dict[str, list[str]]] = {}
    discs = datasets.list_discs_subsets()
    for prob, types in discs.items():
        catalog[prob] = {gt: list(subs) for gt, subs in types.items()}
    fams = datasets.list_benchmark_families()
    for fam_name, type_dict in fams.items():
        if fam_name == "discs":
            continue
        if fam_name == "ea3d":
            # two-level ({dist: [L-label, ...]}) — keep as-is.
            catalog.setdefault(fam_name, {}).update(type_dict)
        elif fam_name == "coloring":
            # coloring/<graph_type>/manifest.jsonl (flat subset). Expose
            # each graph_type as its own 1-element "subset list" so the
            # suite name reads ``coloring-myciel`` uniformly.
            for gt in type_dict.get("", []):
                catalog.setdefault("coloring", {})[gt] = [""]
        elif fam_name == "mis-rrg":
            subsets = type_dict.get("", [])
            if subsets:
                catalog.setdefault("mis-rrg", {})["rrg"] = subsets
        elif fam_name == "gset":
            # data/gset/<subset>/manifest.jsonl (flat). Expose each
            # subset directly so suite identifiers read ``gset-standard``
            # and roll up as ``gset``.
            for gt in type_dict.get("", []):
                catalog.setdefault("gset", {})[gt] = [""]
    # balanced-partition is derived from the DISCS normcut layout — expose
    # it as a separate family so users can pick the `BalancedGraphPartition`
    # objective alongside the `NormalizedCut` objective on the same graphs.
    if "normcut" in discs:
        catalog.setdefault("balanced-partition", {}).update(
            {gt: list(subs) for gt, subs in discs["normcut"].items()}
        )
    return catalog


def _resolve_suite(suite: str) -> list[tuple[str, str, str | None]]:
    """Return a list of ``(family, graph_type, subset)`` triples.

    Uses longest-prefix-first family matching so compound names
    (``mis-rrg``, ``balanced-partition``) are resolved before single-token
    ones (``mis``).
    """
    catalog = _build_catalog()
    if not catalog:
        raise SystemExit(
            "No benchmark data found under ``data/``. Run "
            "`./scripts/setup_discs_data.sh` and/or the appropriate "
            "`scripts/generate_*_instances.py` first."
        )

    out: list[tuple[str, str, str | None]] = []
    if suite == "all":
        for fam, types in catalog.items():
            for gtype, subsets in types.items():
                for sub in subsets:
                    out.append((fam, gtype, sub))
        return out

    # Longest-prefix family match.
    fam: str | None = None
    rest: str = ""
    for cand in sorted(catalog.keys(), key=len, reverse=True):
        if suite == cand:
            fam, rest = cand, ""
            break
        if suite.startswith(cand + "-"):
            fam, rest = cand, suite[len(cand) + 1 :]
            break
    if fam is None:
        raise SystemExit(f"Unknown suite '{suite}'. Available families: {sorted(catalog)}")

    types = catalog[fam]
    if not rest:
        for gtype, subsets in types.items():
            for sub in subsets:
                out.append((fam, gtype, sub))
        return out

    for gtype in types:
        if rest == gtype:
            for sub in types[gtype]:
                out.append((fam, gtype, sub))
            return out
        prefix = gtype + "-"
        if rest.startswith(prefix):
            sub = rest[len(prefix) :]
            if sub in types[gtype]:
                out.append((fam, gtype, sub))
                return out

    # Allow ``mis-rrg-d20_n10000`` (flat subset, no graph_type).
    if "" in types and rest in types[""]:
        out.append((fam, "", rest))
        return out

    raise SystemExit(
        f"Could not resolve suite '{suite}'. Available under {fam}: "
        + ", ".join(f"{fam}-{g}-{s}" if g else f"{fam}-{s}" for g, ss in types.items() for s in ss)
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

    qqa loss conventions:

    * **mis / maxcut / maxclique**: penalised QUBO; use ``score_summary``.
    * **normcut**: minimisation via ``discrete_ncut(x_disc)``.
    * **coloring**: ``score_summary`` returns ``conflicts`` (≥0, 0 ≡ feasible).
    * **mis-rrg**: same contract as mis (MaximumIndependentSet).
    * **balanced-partition**: ``score_summary`` returns ``edge cut``.
    * **ea3d**: unconstrained Ising energy — return ``best_obj`` directly.
    """
    if problem_kind == "normcut":
        with torch.no_grad():
            x_disc = problem.relaxation.project(result.best_sol.unsqueeze(0))
            ncut = float(problem.discrete_ncut(x_disc).item())
        return ncut, True

    if problem_kind == "ea3d":
        # Unconstrained Ising energy (``best_obj`` is already the relaxed
        # loss evaluated on the projected best replica).
        return float(result.best_obj), True

    # mis / maxcut / maxclique / coloring / mis-rrg / balanced-partition:
    # each problem's ``score_summary`` returns ``value`` + ``feasible``.
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
    if problem_kind not in {"mis", "maxcut", "maxclique", "mis-rrg"}:
        raise ValueError(f"--parallel does not support problem '{problem_kind}'.")
    summary = problem.score_summary(result.best_sol)
    values = list(summary["value"].tolist())
    feasibles = [bool(f) for f in summary["feasible"].tolist()]
    return [float(v) for v in values], feasibles


def _approx_ratio(objective: float, best_known: float, kind: str) -> float | None:
    """Approximation ratio relative to the published best (NaN-safe).

    Sign convention: the returned number is "higher is better", clipped so
    ``1.0 = optimal``. For minimisation problems (normcut, coloring,
    balanced-partition, ea3d with known ground state) we return
    ``best_known / objective`` so that pairs with ``objective > 0`` stay
    comparable. For ea3d the ground-state energy is negative; the ratio is
    then ``objective / best_known`` (both negative, ratio ∈ (0, 1]).
    """
    if best_known is None or (isinstance(best_known, float) and np.isnan(best_known)):
        return None
    if kind in {"mis", "maxcut", "maxclique", "mis-rrg", "gset"}:
        return objective / best_known if best_known != 0 else None
    if kind == "normcut":
        return best_known / objective if objective != 0 else None
    if kind == "coloring":
        # best_known is 0 for procedural families when we set K = chromatic
        # number; "distance from feasibility" is ``objective`` itself.
        # Return ``None`` so the mean aggregator ignores this family but
        # ``feasible`` still flags 0-conflict solutions.
        return None
    if kind == "balanced-partition":
        # best_known is NaN upstream (no exact reference); the caller
        # already short-circuited.
        return None
    if kind == "ea3d":
        # Both ``objective`` and ``best_known`` are negative energies; the
        # closer ``objective`` is to ``best_known`` the better. Return
        # ``objective / best_known ∈ (0, 1]`` so "higher is better" still
        # holds (1.0 when the solver reaches the ground state).
        if best_known == 0:
            return None
        return objective / best_known
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
        runner_kwargs["learning_rate"],
        runner_kwargs["temp"],
        runner_kwargs["curve_rate"],
        runner_kwargs["gamma_min"],
        runner_kwargs["gamma_max"],
        runner_kwargs["div_param"],
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
        if args.penalty is not None and prob in {"mis", "maxclique", "mis-rrg"}:
            load_kwargs["penalty"] = args.penalty
        # ``sub == ""`` is the "no real subset" sentinel for flat layouts
        # (coloring/<graph_type>/manifest.jsonl). Pass ``None`` to the
        # loader so it picks up the whole family.
        if load_kwargs["subset"] == "":
            load_kwargs["subset"] = None
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
                        "num_nodes": dataset.manifest[i].get(
                            "num_nodes", dataset.manifest[i].get("num_spins", 0)
                        ),
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
                        "num_nodes": dataset.manifest[i].get(
                            "num_nodes", dataset.manifest[i].get("num_spins", 0)
                        ),
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
