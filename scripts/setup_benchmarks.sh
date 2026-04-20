#!/usr/bin/env bash
# One-shot setup for *every* combinatorial-optimisation benchmark family
# consumed by scripts/bench_discs.py.
#
# Covered families:
#   * DISCS (maxcut / mis / maxclique / normcut) — delegates to
#     ``setup_discs_data.sh`` (HF Hub snapshot).
#   * Graph Coloring (COLOR: myciel, queen) — procedural generator.
#   * MIS on d-regular Random Graphs (PQQA §5.1)   — procedural generator.
#   * 3D Edwards-Anderson spin glass               — procedural generator.
#   * Balanced k-way partition (reuses DISCS normcut/nets graphs).
#
# Usage:
#   ./scripts/setup_benchmarks.sh                       # all families
#   ./scripts/setup_benchmarks.sh --skip-discs          # only procedural ones
#   ./scripts/setup_benchmarks.sh --skip-procedural     # only DISCS
#   ./scripts/setup_benchmarks.sh --discs-args "--limit 5"
#
# Environment variables follow ``setup_discs_data.sh``. Procedural families
# are deterministic (seeded); re-running is a no-op once the .gpickle/.npz
# files exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SKIP_DISCS=0
SKIP_PROCEDURAL=0
DISCS_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-discs)       SKIP_DISCS=1;       shift ;;
        --skip-procedural)  SKIP_PROCEDURAL=1;  shift ;;
        --discs-args)       DISCS_ARGS="$2";    shift 2 ;;
        -h|--help)
            sed -n '1,30p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

cd "${REPO_ROOT}"

if [[ "${SKIP_DISCS}" -eq 0 ]]; then
    echo "[setup_benchmarks] >>> DISCS families (HF snapshot)"
    # shellcheck disable=SC2086
    ./scripts/setup_discs_data.sh ${DISCS_ARGS}
else
    echo "[setup_benchmarks] >>> skipping DISCS"
fi

if [[ "${SKIP_PROCEDURAL}" -eq 0 ]]; then
    echo "[setup_benchmarks] >>> Coloring (myciel, queen)"
    python scripts/generate_coloring_instances.py

    echo "[setup_benchmarks] >>> MIS on RRG (d=20 and d=100, n=10000)"
    python scripts/generate_rrg_instances.py

    echo "[setup_benchmarks] >>> 3D Edwards-Anderson (Gaussian, bimodal; L=4, 6, 8)"
    python scripts/generate_ea3d_instances.py
else
    echo "[setup_benchmarks] >>> skipping procedural families"
fi

echo "[setup_benchmarks] done. Try:"
echo "    python scripts/bench_discs.py --suite coloring-myciel --instances 3"
echo "    python scripts/bench_discs.py --suite mis-rrg-rrg-d20_n10000 --instances 1"
echo "    python scripts/bench_discs.py --suite ea3d-gaussian-L4 --instances 3"
echo "    python scripts/bench_discs.py --suite balanced-partition-nets-MNIST --instances 1"
