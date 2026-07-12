#!/usr/bin/env bash
# NYU-ONLY concurrent multi-GPU sweep — 12 jobs (4 techniques x 3 variants) over N GPUs.
#
# One train.py == one GPU (pinned via CUDA_VISIBLE_DEVICES). The scheduler keeps NGPU jobs
# in flight and starts the next queued job the moment a slot frees, so 12 jobs on 4 GPUs is
# 3 waves with no idle time. There is ZERO change to the training loops — this is pure
# campaign-level parallelism.
#
# Usage (from repo root, inside an OAR job that reserved the GPUs):
#     scripts/g5k/train_sweep_nyu.sh                    # all 12 NYU jobs
#     NGPU=4 scripts/g5k/train_sweep_nyu.sh             # override GPU count
#     scripts/g5k/train_sweep_nyu.sh Technique_2/NYU/base/train.py ...   # explicit subset
#     RESUME=1 scripts/g5k/train_sweep_nyu.sh           # resume each job from its _last.ckpt
#
# Reserve the GPUs first — see scripts/g5k/sweep_nyu.oar.sh.

set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# shellcheck disable=SC1091
source scripts/g5k/env.sh          # PROJECT_DATA / PROJECT_OUT / TORCH_HOME / venv / PYTHONUNBUFFERED

NGPU="${NGPU:-4}"
CONFIG="${CONFIG:-config.yaml}"
RESUME="${RESUME:-0}"
LOGDIR_TB="${PROJECT_OUT:-.}/runs"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOGDIR="${PROJECT_OUT:-.}/logs/${STAMP}"
CKPT_DIR="${PROJECT_OUT:-.}/checkpoints"
mkdir -p "$LOGDIR"

# --- Hard-fail if the GPUs we were promised are not actually here -------------------
# This script runs `set -uo pipefail` WITHOUT -e, so without this check a job that landed
# on a GPU-less node would silently fall back to CPU and "train" 4 models for the entire
# reservation, producing nothing. Fail in 2 seconds instead of wasting 24 hours.
REAL_GPU="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
if [ "${REAL_GPU:-0}" -lt "$NGPU" ]; then
    echo "FATAL: NGPU=$NGPU requested but only ${REAL_GPU:-0} GPU(s) visible to this process."
    echo "       (Did the OAR reservation actually include gpu=$NGPU?)"
    nvidia-smi -L 2>/dev/null || echo "       nvidia-smi found no GPUs at all."
    exit 2
fi

# --- Job manifest: NYU only ---------------------------------------------------------
if [ "$#" -gt 0 ]; then
    JOBS=("$@")
else
    JOBS=()
    for T in Technique_1 Technique_2 Technique_3 Technique_4; do
        for V in base var1 var2; do
            f="$T/NYU/$V/train.py"
            if [ ! -f "$f" ]; then
                echo "FATAL: manifest expects $f but it does not exist."
                exit 2
            fi
            JOBS+=("$f")
        done
    done
fi

echo "host    : $(hostname)"
echo "GPUs    : $NGPU (of ${REAL_GPU} visible)"
echo "jobs    : ${#JOBS[@]}  (NYU only)"
echo "config  : $CONFIG"
echo "logs    : $LOGDIR"
echo "resume  : $RESUME"
nvidia-smi -L

# job_tag Technique_2/NYU/var1/train.py -> T2_NYU_var1   (matches the checkpoint/TB naming)
job_tag() {
    local p="$1"
    local t v
    t="$(echo "$p" | sed -E 's#^Technique_([0-9]+)/.*#\1#')"
    v="$(echo "$p" | sed -E 's#^Technique_[0-9]+/NYU/([^/]+)/train\.py#\1#')"
    echo "T${t}_NYU_${v}"
}

declare -a SLOT_PID SLOT_TAG
for ((g=0; g<NGPU; g++)); do SLOT_PID[$g]=0; SLOT_TAG[$g]=""; done

# Per-job outcome, so a summary can name WHICH runs failed. The old script collapsed this
# to a single FAIL bit: 1 failure and 35 failures looked identical.
declare -a RESULT_TAG RESULT_RC

_any_running() {
    for ((g=0; g<NGPU; g++)); do
        [ "${SLOT_PID[$g]}" != "0" ] && kill -0 "${SLOT_PID[$g]}" 2>/dev/null && return 0
    done
    return 1
}

launch() {  # launch <gpu> <train.py>
    local gpu="$1" train="$2"
    local tag; tag="$(job_tag "$train")"
    local out="${LOGDIR}/${tag}.log"

    local extra=()
    grep -q -- "--logdir" "$train" && extra+=(--logdir "$LOGDIR_TB")

    # Resume from the per-epoch checkpoint written by every trainer.
    if [ "$RESUME" = "1" ]; then
        local last="${CKPT_DIR}/${tag}_last.ckpt"
        [ -f "$last" ] && { extra+=(--resume "$last"); echo "[resume] $tag <- $last"; }
    fi

    echo "[launch] gpu=$gpu  $tag  -> $out"
    CUDA_VISIBLE_DEVICES="$gpu" python "$train" --config "$CONFIG" "${extra[@]}" \
        > "$out" 2>&1 &
    SLOT_PID[$gpu]=$!
    SLOT_TAG[$gpu]="$tag"
}

i=0
while [ "$i" -lt "${#JOBS[@]}" ] || _any_running; do
    for ((g=0; g<NGPU; g++)); do
        pid="${SLOT_PID[$g]}"
        if [ "$pid" != "0" ] && ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; rc=$?
            tag="${SLOT_TAG[$g]}"
            RESULT_TAG+=("$tag"); RESULT_RC+=("$rc")
            if [ "$rc" -ne 0 ]; then
                echo "[done] gpu=$g  $tag  FAILED rc=$rc   (tail: ${LOGDIR}/${tag}.log)"
                tail -n 5 "${LOGDIR}/${tag}.log" 2>/dev/null | sed 's/^/         | /'
            else
                echo "[done] gpu=$g  $tag  ok"
            fi
            SLOT_PID[$g]=0; SLOT_TAG[$g]=""
        fi
        if [ "${SLOT_PID[$g]}" = "0" ] && [ "$i" -lt "${#JOBS[@]}" ]; then
            launch "$g" "${JOBS[$i]}"; i=$((i+1))
        fi
    done
    sleep 5
done

# --- Summary ------------------------------------------------------------------------
echo ""
echo "==================== NYU SWEEP SUMMARY ===================="
FAIL=0
for ((j=0; j<${#RESULT_TAG[@]}; j++)); do
    if [ "${RESULT_RC[$j]}" -eq 0 ]; then
        printf "  %-16s OK\n" "${RESULT_TAG[$j]}"
    else
        printf "  %-16s FAILED (rc=%s)\n" "${RESULT_TAG[$j]}" "${RESULT_RC[$j]}"
        FAIL=$((FAIL+1))
    fi
done
echo "-----------------------------------------------------------"
echo "  ${#RESULT_TAG[@]} jobs, $FAIL failed"
echo "  logs: $LOGDIR"
echo "==========================================================="
exit $(( FAIL > 0 ? 1 : 0 ))
