#!/usr/bin/env bash
# ONE-TIME setup, run on the Grid'5000 FRONTEND (not a compute node).
#
#   ssh <login>@access.grid5000.fr
#   ssh <site>                       # rennes / nantes / lyon / ...
#   cd <repo> && bash scripts/g5k/prepare.sh
#
# Frontends have internet; compute nodes generally do NOT. Anything that needs to be
# downloaded must be fetched HERE, into $HOME (NFS), so the nodes can read it.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

: "${PROJECT_OUT:=$HOME/underwater/out}"
: "${TORCH_HOME:=$HOME/.cache/torch}"
export TORCH_HOME
mkdir -p "$HOME/underwater" "$PROJECT_OUT/runs" "$PROJECT_OUT/checkpoints" "$TORCH_HOME"

# ---- 1. virtualenv -----------------------------------------------------------
if [ ! -d "$HOME/underwater/venv" ]; then
    python3 -m venv "$HOME/underwater/venv"
fi
# shellcheck disable=SC1091
source "$HOME/underwater/venv/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
# torch with CUDA must match the node's driver; check `nvidia-smi` on a GPU node first.
python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"

# ---- 2. pre-cache torchvision pretrained weights ------------------------------
# models/encoder.py uses densenet169(pretrained=True); utils/loss.py (Technique_4)
# uses vgg16(pretrained=True). Both hit the network -> must be cached on the frontend.
python - <<'PY'
import os, torch, torchvision.models as m
print("TORCH_HOME =", os.environ.get("TORCH_HOME"))
m.densenet169(pretrained=True)
m.vgg16(pretrained=True)
print("cached into:", torch.hub.get_dir())
PY

echo
echo "Done. Now:"
echo "  1) put the datasets under \$PROJECT_DATA (or point PROJECT_DATA at group storage)"
echo "  2) submit a job:   oarsub -S ./scripts/g5k/train.oar.sh"
echo "  3) tensorboard:    bash scripts/g5k/tensorboard.sh   (then tunnel from your laptop)"
