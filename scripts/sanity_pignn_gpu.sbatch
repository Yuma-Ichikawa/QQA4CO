#!/bin/bash
# GPU sanity check for the optional CRA-PI-GNN (PyG) backend.
#
# Submit with:    sbatch scripts/sanity_pignn_gpu.sbatch
# Watch with:     squeue -u $USER
# Logs land under: tasks/logs/<job-name>-<jobid>.{out,err}
#
# Supply site-specific partition/account/exclusion flags to ``sbatch`` rather
# than committing them in this portable template.

#SBATCH --job-name=qqa-pignn-sanity
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=tasks/logs/%x-%j.out
#SBATCH --error=tasks/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p tasks/logs

source .venv/bin/activate
unset VIRTUAL_ENV_PROMPT

echo "=== GPU RUNTIME ==="
date
nvidia-smi -L || true
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv 2>&1 || true

echo
echo "=== PyTorch CUDA capability ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda); \
print('arch_list:', torch.cuda.get_arch_list()); \
print('available:', torch.cuda.is_available()); \
print('device 0 :', torch.cuda.get_device_name(0))"

echo
echo "=== PyG import sanity ==="
python -c "import torch_geometric; print('torch_geometric:', torch_geometric.__version__)"

echo
echo "=== demo_pignn_mis.py (CRA-PI-GNN) ==="
python scripts/demo_pignn_mis.py

echo
echo "=== bench_qqa_vs_pignn.py (head-to-head on GPU) ==="
python scripts/bench_qqa_vs_pignn.py --device cuda

echo "=== DONE ==="
