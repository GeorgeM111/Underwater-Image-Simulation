#!/usr/bin/env bash
#OAR -l host=1/gpu=4,walltime=24:0:0
#OAR -O logs/sweep_nyu.%jobid%.out
#OAR -E logs/sweep_nyu.%jobid%.err
#
# Submittable OAR script for the 12-job NYU sweep.
#
#     oarsub -S ./scripts/g5k/sweep_nyu.oar.sh
#
# The #OAR directives above MUST start at column 0 of the first lines of the file — OAR
# scans them literally. (The old train_sweep.sh had its `#OAR -l ...` line nested inside a
# usage comment block, where OAR never sees it, so it reserved nothing.)
#
# `logs/` must already exist: OAR resolves -O/-E at SUBMIT time, before the job's own
# mkdir ever runs. scripts/g5k/prepare.sh creates it.

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

exec ./scripts/g5k/train_sweep_nyu.sh
