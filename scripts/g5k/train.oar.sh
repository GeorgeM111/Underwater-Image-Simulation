#!/usr/bin/env bash
# OAR job script — submit from the frontend with:   oarsub -S ./scripts/g5k/train.oar.sh
# Override the trainer:   TRAIN=Technique_1/Make3D/base/train.py oarsub -S ./scripts/g5k/train.oar.sh
#
# Reserve 1 node with 1 GPU. Some GPU clusters need `-t exotic` — check your site's
# hardware page, or probe with:  oarsub -l "host=1/gpu=1,walltime=0:10:0" -I
#OAR -l host=1/gpu=1,walltime=6:0:0
#OAR -n underwater-train
#OAR -O logs/train.%jobid%.out
#OAR -E logs/train.%jobid%.err

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs

# shellcheck disable=SC1091
source scripts/g5k/env.sh          # PROJECT_DATA / PROJECT_OUT / TORCH_HOME / venv

echo "host      : $(hostname)"
echo "job       : ${OAR_JOB_ID:-<none>}"
echo "PROJECT_DATA=$PROJECT_DATA"
echo "PROJECT_OUT =$PROJECT_OUT"
nvidia-smi || echo "WARNING: no GPU visible"

TRAIN="${TRAIN:-Technique_3/NYU/var2/train.py}"

# --logdir puts TensorBoard runs on NFS ($PROJECT_OUT) so the frontend can serve them.
# (The GAN/EncDec baselines have no --logdir; they read cfg.runs_dir, which is
#  ${PROJECT_OUT}/runs anyway — so they land in the same place.)
if grep -q "add_argument('--logdir'" "$TRAIN"; then
    python "$TRAIN" --config config.yaml --logdir "$PROJECT_OUT/runs"
else
    python "$TRAIN" --config config.yaml
fi
