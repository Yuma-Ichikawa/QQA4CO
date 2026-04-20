#!/usr/bin/env bash
# Set up the DISCS combinatorial-optimization benchmark suite.
#
# This script is a *thin wrapper*. It picks a download source, fetches the
# raw archive, and hands off to ``convert_discs_to_qqa.py`` to produce the
# unified ``data/discs/<problem>/<graph_type>/<subset>/`` layout.
#
# Usage:
#   ./scripts/setup_discs_data.sh                       # HF Hub, all problems
#   ./scripts/setup_discs_data.sh --source gdrive       # fallback to Drive
#   ./scripts/setup_discs_data.sh --problem mis --subsets satlib
#   ./scripts/setup_discs_data.sh --limit 5             # smoke (5 inst/subset)
#
# Environment variables:
#   QQA_DATA_DIR        if set, overrides ``data/`` root (rare)
#   DISCS_HF_REPO_ID    HF dataset repo id (default: Yuma-Ichikawsa/qqa4co-bench)
#   DISCS_GDRIVE_FOLDER Google Drive folder URL (default: paper-published one)

set -euo pipefail

# --------------------------------------------------------------------------- #
# locate repo root                                                            #
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${QQA_DATA_DIR:-${REPO_ROOT}/data}"
DST="${DATA_ROOT}/discs"
RAW="${DST}/_raw"

# --------------------------------------------------------------------------- #
# defaults                                                                    #
# --------------------------------------------------------------------------- #
SOURCE="hf"
PROBLEM="all"
SUBSETS=""
LIMIT=""
HF_REPO_ID="${DISCS_HF_REPO_ID:-Yuma-Ichikawsa/qqa4co-bench}"
# The published DISCS Drive folder (1nEppx...) contains exactly two files:
#   * 2dtsp.zip                    (TSP data, NOT used by us)
#   * DISCS-DATA.tar.gz  ID=1lbpdEqs_rDqaLmS3YkFrbn7iK8z1K1it (~6.7 GB)
# We download the tarball directly to skip the TSP archive and the folder
# walk. The CO subset we need lives at DISCS-DATA/sco/ inside the tarball.
DISCS_GDRIVE_FILE_ID="${DISCS_GDRIVE_FILE_ID:-1lbpdEqs_rDqaLmS3YkFrbn7iK8z1K1it}"

# --------------------------------------------------------------------------- #
# arg parsing                                                                 #
# --------------------------------------------------------------------------- #
usage() {
    sed -n '2,17p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)   SOURCE="$2";   shift 2 ;;
        --problem)  PROBLEM="$2";  shift 2 ;;
        --subsets)  SUBSETS="$2";  shift 2 ;;
        --limit)    LIMIT="$2";    shift 2 ;;
        -h|--help)  usage ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${RAW}"
echo "[setup_discs_data] dst=${DST}"
echo "[setup_discs_data] source=${SOURCE}  problem=${PROBLEM}  subsets=${SUBSETS:-<all>}  limit=${LIMIT:-<none>}"

# --------------------------------------------------------------------------- #
# fetch                                                                       #
# --------------------------------------------------------------------------- #
fetch_hf() {
    # The HF dataset hosts the *converted* `.gpickle` + `manifest.jsonl`
    # tree directly (~3.9 GB). We snapshot it straight into ${DST} and
    # skip the `convert_discs_to_qqa.py` step entirely.
    echo "[setup_discs_data] fetching pre-converted dataset from Hugging Face Hub: ${HF_REPO_ID}"
    python - <<PY
from huggingface_hub import snapshot_download
import os
allow = ["maxcut/**", "mis/**", "maxclique/**", "normcut/**", "**/manifest.jsonl", "README.md"]
local = snapshot_download(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    local_dir=os.environ["DST"],
    allow_patterns=allow,
)
print("[setup_discs_data] HF snapshot at:", local)
PY
}

fetch_gdrive() {
    echo "[setup_discs_data] fetching DISCS-DATA.tar.gz (~6.7 GB) from Google Drive"
    echo "[setup_discs_data] file id=${DISCS_GDRIVE_FILE_ID}"
    python - <<PY
import os, sys, tarfile
import gdown

raw = os.environ["RAW"]
os.makedirs(raw, exist_ok=True)
tarball = os.path.join(raw, "DISCS-DATA.tar.gz")
if not os.path.exists(tarball) or os.path.getsize(tarball) < 1_000_000_000:
    url = f"https://drive.google.com/uc?id={os.environ['DISCS_GDRIVE_FILE_ID']}"
    gdown.download(url, output=tarball, quiet=False, resume=True)
else:
    print(f"[setup_discs_data] reusing cached {tarball} ({os.path.getsize(tarball)/1e9:.1f} GB)")

print("[setup_discs_data] extracting DISCS-DATA/sco/ ...")
with tarfile.open(tarball, "r:gz") as tf:
    members = [m for m in tf.getmembers() if m.name.startswith("DISCS-DATA/sco/")]
    if not members:
        print("[setup_discs_data] tarball does not contain DISCS-DATA/sco/", file=sys.stderr)
        sys.exit(5)
    tf.extractall(raw, members=members)
print(f"[setup_discs_data] extracted {len(members)} entries under {raw}/DISCS-DATA/sco/")
PY
}

case "${SOURCE}" in
    hf)
        if ! python -c "import huggingface_hub" >/dev/null 2>&1; then
            echo "[setup_discs_data] huggingface_hub not installed; run: pip install huggingface_hub" >&2
            exit 3
        fi
        export HF_REPO_ID DST
        if fetch_hf; then
            echo "[setup_discs_data] done. HF snapshot is already in the unified gpickle+manifest format."
            echo "[setup_discs_data] Try:"
            echo "    python scripts/bench_discs.py --suite mis-satlib --backend qqa --instances 3"
            exit 0
        else
            echo "[setup_discs_data] HF download failed, falling back to gdrive..." >&2
            SOURCE="gdrive"
        fi
        ;;
esac

if [[ "${SOURCE}" == "gdrive" ]]; then
    if ! python -c "import gdown" >/dev/null 2>&1; then
        echo "[setup_discs_data] gdown not installed; run: pip install gdown" >&2
        exit 3
    fi
    export DISCS_GDRIVE_FILE_ID RAW
    fetch_gdrive
fi

# --------------------------------------------------------------------------- #
# locate raw `sco` root                                                       #
# --------------------------------------------------------------------------- #
# After download the structure may be:
#   ${RAW}/sco/...                       (HF preformatted upload)
#   ${RAW}/DISCS-DATA/sco/...            (gdrive tarball extraction)
#   ${RAW}/...                           (HF unpacked at top)
SCO_ROOT=""
for cand in "${RAW}/sco" "${RAW}/DISCS-DATA/sco" "${RAW}"; do
    if [[ -d "${cand}/RB_test" || -d "${cand}/satlib_test" || -d "${cand}/maxcut-ba" ]]; then
        SCO_ROOT="${cand}"
        break
    fi
done

if [[ -z "${SCO_ROOT}" ]]; then
    echo "[setup_discs_data] could not locate the DISCS sco/ root under ${RAW}" >&2
    echo "[setup_discs_data] inspect the directory and run convert_discs_to_qqa.py manually" >&2
    exit 4
fi
echo "[setup_discs_data] sco root: ${SCO_ROOT}"

# --------------------------------------------------------------------------- #
# convert                                                                     #
# --------------------------------------------------------------------------- #
CONVERT_ARGS=(
    --src "${SCO_ROOT}"
    --dst "${DST}"
    --problem "${PROBLEM}"
)
[[ -n "${SUBSETS}" ]] && CONVERT_ARGS+=(--subsets "${SUBSETS}")
[[ -n "${LIMIT}" ]]   && CONVERT_ARGS+=(--limit   "${LIMIT}")

python "${SCRIPT_DIR}/convert_discs_to_qqa.py" "${CONVERT_ARGS[@]}"

echo "[setup_discs_data] done. Try:"
echo "    python scripts/bench_discs.py --suite mis-satlib --backend qqa --instances 3"
