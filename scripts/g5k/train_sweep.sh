#!/usr/bin/env bash
# Concurrent multi-GPU sweep launcher — Layer 2 acceleration.
#
# Runs many independent train.py jobs across the node's GPUs, one job pinned per
# GPU via CUDA_VISIBLE_DEVICES, keeping every GPU busy. This is the highest-ROI
# way to use the 4x H200s for the technique/variant/dataset sweep: near-perfect
# 4x throughput on the *campaign*, with ZERO changes to the training loops.
#
# Usage (from repo root, inside an OAR job that reserved the GPUs):
#     scripts/g5k/train_sweep.sh                 # runs the default manifest below
#     NGPU=4 scripts/g5k/train_sweep.sh          # override GPU count
#     scripts/g5k/train_sweep.sh a/train.py b/train.py ...   # explicit targets
#
# Reserve the whole node's GPUs first, e.g.:
#     #OAR -l host=1/gpu=4,walltime=12:0:0
#
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs

# shellcheck disable=SC1091
source scripts/g5k/env.sh          # PROJECT_DATA / PROJECT_OUT / TORCH_HOME / venv

NGPU="${NGPU:-4}"
LOGDIR_TB="${PROJECT_OUT:-.}/runs"

# Default sweep manifest: every technique x variant x dataset. Edit freely, or
# pass explicit train.py paths as arguments to override.
if [ "$#" -gt 0 ]; then
    JOBS=("$@")
else
    JOBS=()
    for T in Technique_1 Technique_2 Technique_3 Technique_4; do
        for DS in NYU Make3D KITTI; do
            for V in base var1 var2; do
                f="$T/$DS/$V/train.py"
                [ -f "$f" ] && JOBS+=("$f")
            done
        done
    done
fi

echo "host   : $(hostname)"
echo "GPUs   : $NGPU"
echo "jobs   : ${#JOBS[@]}"
nvidia-smi -L || echo "WARNING: no GPU visible"

# Simple GPU-slot scheduler: keep NGPU jobs in flight, pinning each to one GPU.
declare -a SLOT_PID          # PID currently occupying each GPU slot (0 = free)
for ((g=0; g<NGPU; g++)); do SLOT_PID[$g]=0; done

# helper: is any GPU slot still running?
_any_running() {
    for ((g=0; g<NGPU; g++)); do
        [ "${SLOT_PID[$g]}" != "0" ] && kill -0 "${SLOT_PID[$g]}" 2>/dev/null && return 0
    done
    return 1
}

launch() {  # launch <gpu> <train.py>
    local gpu="$1" train="$2"
    local tag; tag="$(echo "$train" | tr '/' '_')"
    local out="logs/${tag}.gpu${gpu}.out"
    echo "[launch] gpu=$gpu  $train  -> $out"
    local logflag=()
    grep -q "add_argument('--logdir'" "$train" && logflag=(--logdir "$LOGDIR_TB")
    CUDA_VISIBLE_DEVICES="$gpu" python "$train" --config config.yaml "${logflag[@]}" \
        > "$out" 2>&1 &
    SLOT_PID[$gpu]=$!
}

i=0
FAIL=0
while [ "$i" -lt "${#JOBS[@]}" ] || _any_running; do
    # fill free slots
    for ((g=0; g<NGPU; g++)); do
        pid="${SLOT_PID[$g]}"
        if [ "$pid" != "0" ] && ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; rc=$?
            [ "$rc" -ne 0 ] && { echo "[done] gpu=$g pid=$pid FAILED rc=$rc"; FAIL=1; } \
                            || echo "[done] gpu=$g pid=$pid ok"
            SLOT_PID[$g]=0
        fi
        if [ "${SLOT_PID[$g]}" = "0" ] && [ "$i" -lt "${#JOBS[@]}" ]; then
            launch "$g" "${JOBS[$i]}"; i=$((i+1))
        fi
    done
    sleep 5
done
echo "sweep complete (FAIL=$FAIL)"
exit "$FAIL"
