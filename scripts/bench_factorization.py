"""Benchmark runner for the planted-solution factorization Ising/QUBO suite.

Usage examples
--------------
$ uv run python scripts/bench_factorization.py --bits 4 --instances 5
$ uv run python scripts/bench_factorization.py --bits 5 --instances 3 \
      --device cuda --sol-size 500 --num-epochs 5000

For each instance the script reports:

* the planted ``(p, q)`` and ``N``;
* the number of free Ising spins (problem size after pin folding);
* QQA's residual energy ``H(x̂) − E_0`` (zero if the planted is found);
* the decoded ``(p̂, q̂)``, ``N̂`` and the bit-Hamming distance to ``x*``;
* wall-clock time.

The benchmark is intentionally **CPU-friendly** for ``bits ≤ 5`` so that
contributors can iterate without a GPU; for ``bits ≥ 6`` the recommended
defaults are ``--device cuda --sol-size 1000 --num-epochs 5000``.

Citation
--------
Hen, *Planted-solution SAT and Ising benchmarks from integer
factorization*, arXiv:2604.09837 (2026).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import qqa

# `scripts/` is on sys.path when this file is invoked as
# ``python scripts/bench_factorization.py``; importing as a sibling module
# works without packaging.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_common as bench  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bits",
        type=int,
        default=4,
        help="bit-length d of each prime factor (≥ 2). Problem size "
        "scales as O(d^4) without preprocessing.",
    )
    p.add_argument(
        "--instances", type=int, default=5, help="number of random semiprimes to benchmark."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="cpu / cuda / auto")
    p.add_argument("--sol-size", type=int, default=200)
    p.add_argument("--num-epochs", type=int, default=2000)
    bench.add_qqa_hp_args(p)
    p.add_argument(
        "--output", type=Path, default=None, help="optional JSON file to dump per-instance results."
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    log = bench.setup_logging(args.verbose, name="bench_factorization")
    bench.setup_device(args)
    log.info("bits=%d instances=%d device=%s", args.bits, args.instances, args.device)
    hp = bench.qqa_hp_kwargs(args)
    log.info(
        "qqa hp: lr=%g temp=%g curve_rate=%d gamma=[%g,%g] div=%g sol_size=%d num_epochs=%d",
        hp["learning_rate"],
        hp["temp"],
        hp["curve_rate"],
        hp["gamma_min"],
        hp["gamma_max"],
        hp["div_param"],
        args.sol_size,
        args.num_epochs,
    )

    suite = qqa.random_factorization_problems(
        bit_length=args.bits,
        num_instances=args.instances,
        seed=args.seed,
        device=args.device,
    )

    rows: list[dict[str, Any]] = []
    n_ground_state = 0  # E_min = 0 AND decoded N̂ = N
    n_decoded = 0  # decoded N̂ = N (input bits right, internal bits maybe not)
    for k, prob in enumerate(suite):
        t0 = time.time()
        result = bench.run_qqa_anneal(
            prob,
            device=args.device,
            sol_size=args.sol_size,
            num_epochs=args.num_epochs,
            **hp,
        )
        wall = time.time() - t0
        s = result.score
        ex = s["extra"]
        # Two distinct success criteria:
        # - `decoded`: the optimiser placed the right bits on the input wires
        #   p_0..p_{n_p-1} and q_0..q_{n_q-1} (cheap to satisfy locally).
        # - `gs`: the entire spin string sits at the planted ground state
        #   (energy 0), which additionally requires every internal pp/sum/
        #   carry spin to be consistent with the multiplication circuit.
        decoded_ok = bool(ex["matches_planted"])
        gs_ok = bool(s["feasible"])
        n_decoded += int(decoded_ok)
        n_ground_state += int(gs_ok)
        log.info(
            "  [%d/%d] N=%d=%d*%d  free=%d  E_min=%.3f  decoded=%d*%d=%d  "
            "decoded_N=%s  gs=%s  t=%.2fs",
            k + 1,
            args.instances,
            ex["N"],
            ex["p"],
            ex["q"],
            ex["num_free_spins"],
            float(s["value"]),
            ex["p_hat"],
            ex["q_hat"],
            ex["N_hat"],
            "Y" if decoded_ok else "N",
            "Y" if gs_ok else "N",
            wall,
        )
        rows.append(
            {
                "k": k,
                "N": ex["N"],
                "p": ex["p"],
                "q": ex["q"],
                "num_free_spins": ex["num_free_spins"],
                "energy_above_planted": float(s["value"]),
                "p_hat": ex["p_hat"],
                "q_hat": ex["q_hat"],
                "N_hat": ex["N_hat"],
                "decoded_N_correct": decoded_ok,
                "is_ground_state": gs_ok,
                "hamming_to_planted": ex["hamming_to_planted"],
                "wall_s": wall,
            }
        )

    log.info(
        "== SUMMARY == ground-state %d/%d (%.1f%%)  decoded-N %d/%d (%.1f%%)",
        n_ground_state,
        len(suite),
        100.0 * n_ground_state / max(1, len(suite)),
        n_decoded,
        len(suite),
        100.0 * n_decoded / max(1, len(suite)),
    )
    if args.output:
        payload = {
            "bits": args.bits,
            "instances": args.instances,
            "device": args.device,
            "qqa_hp": {**hp, "sol_size": args.sol_size, "num_epochs": args.num_epochs},
            "ground_state_count": n_ground_state,
            "decoded_n_count": n_decoded,
            "total": len(suite),
            "results": rows,
        }
        bench.dump_results_json(args.output, payload)
        log.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
