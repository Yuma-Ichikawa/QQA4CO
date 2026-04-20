#!/usr/bin/env bash
# One-shot setup for *every* combinatorial-optimisation benchmark family
# consumed by scripts/bench_discs.py.
#
# Covered families:
#   * DISCS (maxcut / mis / maxclique / normcut)        -- ~3.9 GB
#   * Graph Coloring (COLOR: myciel, queen)             -- ~340 KB
#   * MIS on d-regular Random Graphs (PQQA §5.1)        -- ~82  MB
#   * 3D Edwards-Anderson spin glass (Gaussian/bimodal) -- ~280 KB
#   * Balanced k-way partition (reuses DISCS nets)      -- (no extra payload)
#
# Two sources are supported. The default ``hf`` path pulls every family
# straight from the Hugging Face dataset
#   https://huggingface.co/datasets/Yuma-Ichikawsa/discs-co-bench
# (no login required, no conversion step). The ``local`` path re-generates
# the non-DISCS families from scratch on this machine (useful when you
# cannot reach the Hub, or when tweaking the generators themselves).
#
# Usage:
#   ./scripts/setup_benchmarks.sh                           # everything, HF
#   ./scripts/setup_benchmarks.sh --source local            # procedural regen
#   ./scripts/setup_benchmarks.sh --only coloring,ea3d      # cherry-pick
#   ./scripts/setup_benchmarks.sh --skip discs              # skip DISCS only
#   ./scripts/setup_benchmarks.sh --hf-repo OTHER/discs-co-bench
#   ./scripts/setup_benchmarks.sh --discs-args "--limit 5"  # DISCS smoke only
#
# Environment variables:
#   QQA_DATA_DIR       overrides ``data/`` root if set (rare)
#   DISCS_HF_REPO_ID   overrides the HF dataset repo id

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE="hf"                       # hf | local
ONLY=""                           # comma-separated family names
SKIP=""                           # comma-separated family names
DISCS_ARGS=""
HF_REPO_ID="${DISCS_HF_REPO_ID:-Yuma-Ichikawsa/discs-co-bench}"

usage() { sed -n '1,30p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)           SOURCE="$2";       shift 2 ;;
        --only)             ONLY="$2";         shift 2 ;;
        --skip)             SKIP="$2";         shift 2 ;;
        --discs-args)       DISCS_ARGS="$2";   shift 2 ;;
        --hf-repo)          HF_REPO_ID="$2";   shift 2 ;;
        --skip-discs)       SKIP="discs${SKIP:+,$SKIP}";      shift ;;
        --skip-procedural)  SKIP="coloring,mis-rrg,ea3d${SKIP:+,$SKIP}"; shift ;;
        -h|--help)          usage ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

cd "${REPO_ROOT}"

ALL="discs coloring mis-rrg ea3d"
should_run() {
    local fam="$1"
    if [[ -n "${ONLY}" ]]; then
        [[ ",${ONLY}," == *",${fam},"* ]] || return 1
    fi
    [[ ",${SKIP}," != *",${fam},"* ]]
}

case "${SOURCE}" in
    hf|local) ;;
    *) echo "unknown --source '${SOURCE}' (expected hf|local)" >&2; exit 2 ;;
esac

DATA_ROOT="${QQA_DATA_DIR:-${REPO_ROOT}/data}"
mkdir -p "${DATA_ROOT}"

fetch_hf_family() {
    local fam="$1"
    echo "[setup_benchmarks] >>> fetching ${fam}/ from ${HF_REPO_ID}"
    SB_FAM="${fam}" SB_ROOT="${DATA_ROOT}" SB_REPO="${HF_REPO_ID}" python - <<'PY'
import os, sys
from huggingface_hub import snapshot_download
fam = os.environ["SB_FAM"]
root = os.environ["SB_ROOT"]
repo = os.environ["SB_REPO"]
try:
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=[f"{fam}/**"],
    )
except Exception as exc:  # noqa: BLE001
    sys.stderr.write(f"[setup_benchmarks] HF fetch for {fam!r} failed: {exc}\n")
    sys.exit(7)
PY
}

# --------------------------------------------------------------------------- #
# DISCS                                                                       #
# --------------------------------------------------------------------------- #
if should_run discs; then
    if [[ "${SOURCE}" == "hf" ]]; then
        # shellcheck disable=SC2086
        DISCS_HF_REPO_ID="${HF_REPO_ID}" ./scripts/setup_discs_data.sh ${DISCS_ARGS}
    else
        echo "[setup_benchmarks] --source local does not regenerate DISCS (too big)."
        echo "[setup_benchmarks] forcing HF for DISCS; pass --skip discs to suppress."
        # shellcheck disable=SC2086
        DISCS_HF_REPO_ID="${HF_REPO_ID}" ./scripts/setup_discs_data.sh ${DISCS_ARGS}
    fi
else
    echo "[setup_benchmarks] >>> skip discs"
fi

# --------------------------------------------------------------------------- #
# Coloring / MIS-RRG / EA3D                                                   #
# --------------------------------------------------------------------------- #
for fam in coloring mis-rrg ea3d; do
    if ! should_run "${fam}"; then
        echo "[setup_benchmarks] >>> skip ${fam}"
        continue
    fi
    if [[ "${SOURCE}" == "hf" ]]; then
        fetch_hf_family "${fam}"
    else
        case "${fam}" in
            coloring) python scripts/generate_coloring_instances.py ;;
            mis-rrg)  python scripts/generate_rrg_instances.py ;;
            ea3d)     python scripts/generate_ea3d_instances.py ;;
        esac
    fi
done

echo ""
echo "[setup_benchmarks] done. Try:"
echo "    python scripts/bench_discs.py --suite coloring-myciel --instances 3"
echo "    python scripts/bench_discs.py --suite mis-rrg-rrg-d20_n10000 --instances 1"
echo "    python scripts/bench_discs.py --suite ea3d-gaussian-L4 --instances 3"
echo "    python scripts/bench_discs.py --suite balanced-partition-nets-MNIST --instances 1"
echo "    python scripts/bench_discs.py --suite all --instances 3 --output results.json"
echo "    python scripts/plot_benchmarks.py results.json --output bench_report.png"
