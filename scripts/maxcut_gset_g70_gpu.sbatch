#!/bin/bash
# Reproduce G-set G70 MaxCut >= 9514 with PQQA / CRA-PI-GNN / CPRA on a
# single GPU. Submits all three methods sequentially in the same job so
# one log file shows the head-to-head comparison.
#
# Submit with:    sbatch scripts/maxcut_gset_g70_gpu.sbatch
# Watch with:     squeue -u $USER
# Logs land in:   tasks/logs/<job-name>-<jobid>.{out,err}
#
# Supply site-specific partition/account/exclusion flags to ``sbatch`` rather
# than committing them in this portable template. Runtime varies by hardware.

#SBATCH --job-name=qqa-g70
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=tasks/logs/%x-%j.out
#SBATCH --error=tasks/logs/%x-%j.err
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p tasks/logs data/gset

source .venv/bin/activate
unset VIRTUAL_ENV_PROMPT

echo "=== GPU RUNTIME ==="
date
nvidia-smi -L || true
python -c "import torch; print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# A single ``--method all`` invocation iterates over PQQA / CRA / CPRA inside
# the same Python process, so the BFS-2color baseline and the G70 graph are
# only built once. ``|| true`` keeps the job running even if one method
# fails — the per-method PASS/FAIL line is the source of truth.
python scripts/maxcut_gset_g70.py --method all --device cuda || true

echo
echo "=== SUMMARY ==="
grep -E "method=.* cut=" tasks/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out || true

echo
echo "=== DONE ==="
