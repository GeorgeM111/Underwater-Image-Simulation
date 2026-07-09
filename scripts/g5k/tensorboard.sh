#!/usr/bin/env bash
# Serve TensorBoard on the Grid'5000 FRONTEND, reading the runs your compute node wrote.
#
# This works because $HOME is per-site NFS, shared between the frontend and the nodes:
# the job writes ${PROJECT_OUT}/runs, and the frontend can read it live.
#
#   frontend:   bash scripts/g5k/tensorboard.sh
#   laptop:     ssh -N -L 6006:localhost:6006 -J <login>@access.grid5000.fr <login>@<site>
#   browser:    http://localhost:6006
#
# Bind to 127.0.0.1 only — never expose it on the frontend's public interface.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
# shellcheck disable=SC1091
source scripts/g5k/env.sh

PORT="${PORT:-6006}"
echo "serving ${PROJECT_OUT}/runs on 127.0.0.1:${PORT}"
echo "tunnel from your laptop:"
echo "  ssh -N -L ${PORT}:localhost:${PORT} -J \$USER@access.grid5000.fr \$USER@<site>"
exec tensorboard --logdir "$PROJECT_OUT/runs" --port "$PORT" --host 127.0.0.1
