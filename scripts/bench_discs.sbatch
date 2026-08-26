#!/bin/bash
#SBATCH --job-name=qqa-discs-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=benchmark-output/logs/%x-%j.out
#SBATCH --error=benchmark-output/logs/%x-%j.err
#
# DISCS combinatorial-optimization benchmark on a single GPU.
#
# Defaults to the full MIS / SATLIB sweep (500 instances) and the full
# NormCut / nets sweep (9 graphs). Override with environment variables:
#
#   SUITES="mis-satlib-uf normcut-nets"   sbatch scripts/bench_discs.sbatch
#   SOL_SIZE=64  NUM_EPOCHS=1500          sbatch scripts/bench_discs.sbatch
#   QQA_DATA_DIR=/path/to/data            sbatch scripts/bench_discs.sbatch
#   PARALLEL=1                            sbatch scripts/bench_discs.sbatch
#       (folds every instance of MIS / MaxCut / MaxClique into a single
#        batched-instance ``qqa.anneal`` call; NormCut still runs sequentially
#        because no batched class exists yet.)
#
# Output JSON files live under benchmark-output/discs/<jobid>/.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

QQA4CO_ROOT="$(pwd)"
LOG_DIR="${QQA4CO_ROOT}/benchmark-output/logs"
RESULT_DIR="${QQA4CO_ROOT}/benchmark-output/discs/${SLURM_JOB_ID:-manual}"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

source "${QQA4CO_ROOT}/.venv/bin/activate"
date; nvidia-smi -L || true
python -c "import torch, qqa; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('qqa', qqa.__version__)"

# Default to "the full DISCS sweep we actually have data for". The data is
# expected to live under ${QQA_DATA_DIR}/discs/ — convert with
#   ./scripts/setup_discs_data.sh
# beforehand.
DEFAULT_SUITES=(
    "mis-satlib-uf"
    "normcut-nets"
)
SUITES="${SUITES:-${DEFAULT_SUITES[*]}}"

SOL_SIZE="${SOL_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1.0}"
TEMP="${TEMP:-1e-3}"
CURVE_RATE="${CURVE_RATE:-4}"
GAMMA_MIN="${GAMMA_MIN:--2}"
GAMMA_MAX="${GAMMA_MAX:-0.1}"
DIV_PARAM="${DIV_PARAM:-0.2}"
PARALLEL_FLAG=""
if [[ "${PARALLEL:-0}" == "1" ]]; then
    PARALLEL_FLAG="--parallel"
fi
PENALTY_FLAG=""
if [[ -n "${PENALTY:-}" ]]; then
    PENALTY_FLAG="--penalty ${PENALTY}"
fi

for suite in ${SUITES}; do
    echo "================================================================"
    echo "=== suite: ${suite}${PARALLEL_FLAG:+  ($PARALLEL_FLAG)}"
    echo "================================================================"
    out="${RESULT_DIR}/${suite//\//_}.json"
    date
    python "${QQA4CO_ROOT}/scripts/bench_discs.py" \
        --suite "${suite}" \
        --backend qqa \
        --sol-size "${SOL_SIZE}" \
        --num-epochs "${NUM_EPOCHS}" \
        --learning-rate "${LEARNING_RATE}" \
        --temp "${TEMP}" \
        --curve-rate "${CURVE_RATE}" \
        --gamma-min "${GAMMA_MIN}" \
        --gamma-max "${GAMMA_MAX}" \
        --div-param "${DIV_PARAM}" \
        --device cuda \
        ${PARALLEL_FLAG} \
        ${PENALTY_FLAG} \
        --output "${out}"
    date
done

echo "=== DISCS bench done ==="
echo "results in: ${RESULT_DIR}"
