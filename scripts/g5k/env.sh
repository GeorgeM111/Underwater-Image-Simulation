# Source me on the Grid'5000 frontend AND inside jobs:  source scripts/g5k/env.sh
#
# The repo's config.yaml uses ${PROJECT_DATA} / ${PROJECT_OUT}, so the SAME config
# works locally and on G5K — you only set these two variables.

# Where the repo lives (this file is scripts/g5k/env.sh)
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Datasets + ground truth + parameter matrices.
# NOTE: $HOME on G5K is per-site NFS and QUOTA-LIMITED. If you bring KITTI (~198 GB)
# put PROJECT_DATA on group storage instead, e.g. /srv/storage/<group>@<site>/underwater
export PROJECT_DATA="${PROJECT_DATA:-$HOME/underwater/data}"

# Checkpoints + TensorBoard runs. Keep this on $HOME (NFS) so the FRONTEND can read
# the runs that the COMPUTE NODE writes — that's what makes remote TensorBoard work.
export PROJECT_OUT="${PROJECT_OUT:-$HOME/underwater/out}"

# torchvision pretrained weights (densenet169, vgg16) are downloaded from the internet.
# Compute nodes usually can't reach it, so we pre-cache on the frontend into NFS $HOME
# and point every process at that cache.
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# venv created by scripts/g5k/prepare.sh
if [ -f "$HOME/underwater/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$HOME/underwater/venv/bin/activate"
fi

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Unbuffered stdout. Python BLOCK-buffers stdout when it is redirected to a file, so the
# per-epoch "epoch N/M train=... val=..." lines would sit in a 4-8 KB buffer and only appear
# hours later (or never, if the job is killed). With this, `tail -f logs/T1_NYU_base.log`
# shows progress live. PYTHONFAULTHANDLER dumps a traceback on a hard crash (segfault/OOM),
# which is otherwise silent.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

mkdir -p "$PROJECT_OUT/runs" "$PROJECT_OUT/checkpoints" "$TORCH_HOME"
