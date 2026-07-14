#!/usr/bin/env bash
# Regenerate ALL NYU ground truth, in the ONE correct order.
#
# The complex target is FROZEN ON DISK. After any change to the physics (gamma_angular,
# alpha_metric, u, turbu_*, jerlov_water_types, airlight, the depth clip, complex_model,
# ...) the GT is STALE and every downstream number is meaningless. There is no way to
# detect this by looking at the images.
#
# ORDER MATTERS, and each step consumes the previous one's output:
#
#   1. params  — sample a Jerlov water type + airlight PER IMAGE (Beta_Mat / A_Mat .npy).
#                The dataloader indexes these positionally, so they must exist before GT.
#   2. filter  — rank images by informativeness -> {nyu_subset_size}_filtered_nyu.npy.
#                Restricted to [0, split_idx) so the test tail can never leak into training.
#   3. GT      — render haze + complex GT for the training subset AND THE TEST TAIL.
#
# Step 3's --with-test-tail is NOT optional when nyu_test_mode: "tail". The subset generator
# only writes GT for TRAINING indices, but the test loader reads GT for the TAIL indices. On
# a pre-existing GT directory the tail files SURVIVE FROM THE OLD PHYSICS, so you would train
# on new-physics GT and be scored against old-physics GT — silently, with no missing file and
# no error. (A physics_manifest.json is now written into the GT dir and asserted at load time,
# so this specific failure is caught, but generating the tail is still the fix.)
#
# Usage:
#     scripts/regenerate_nyu_gt.sh              # full pipeline
#     SKIP_PARAMS=1 scripts/regenerate_nyu_gt.sh   # keep existing water types/airlight
#     SKIP_FILTER=1 scripts/regenerate_nyu_gt.sh   # keep the existing informative subset
#
# NOTE: regenerating the params RESAMPLES the water type and airlight of every image. That is
# fine (it is seeded by config.random_seed, so it is reproducible), but it means the GT MUST
# be fully regenerated afterwards — never regenerate params without regenerating GT.

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CONFIG="${CONFIG:-config.yaml}"
SKIP_PARAMS="${SKIP_PARAMS:-0}"
SKIP_FILTER="${SKIP_FILTER:-0}"

if [ -f scripts/g5k/env.sh ]; then
    # shellcheck disable=SC1091
    source scripts/g5k/env.sh
fi

echo "=============================================================="
echo " NYU GT REGENERATION"
echo " config : $CONFIG"
python - <<'PY'
from config import CONFIG as C
from data_generation.data_2 import physics_fingerprint
_, h = physics_fingerprint(C)
print(" physics_hash : %s" % h)
print(" gamma_angular: %s" % (C.gamma_angular,))
print(" alpha_metric : %s" % (C.alpha_metric,))
print(" u / turbu_pr : %s / %s" % (C.u, C.turbu_pr))
print(" GT dir       : %s" % C.nyu_gt_train_dir)
PY
echo "=============================================================="

if [ "$SKIP_PARAMS" != "1" ]; then
    echo ""
    echo "[1/3] parameter matrices (per-image water type + airlight)"
    python data_generation/beta_atmosphere_nyu_train.py --config "$CONFIG"
    python data_generation/beta_atmosphere_nyu_test.py  --config "$CONFIG"
else
    echo ""
    echo "[1/3] params  — SKIPPED (SKIP_PARAMS=1)"
fi

TRAIN_MODE="$(python -c 'from config import CONFIG; print(CONFIG.nyu_train_mode)')"

if [ "$TRAIN_MODE" = "all" ]; then
    echo ""
    echo "[2/3] informative-subset selection — SKIPPED (nyu_train_mode='all')"
    echo ""
    echo "[3/3] ground truth: the FULL dataset (all 50,688 indices)"
    echo "      This covers the 96% training pool AND the 4% test tail by construction, so the"
    echo "      stale-tail failure mode cannot occur. ~23 GB on disk (both GTs are uint8)."
    python data_generation/generate_gt_nyu_train.py
else
    if [ "$SKIP_FILTER" != "1" ]; then
        echo ""
        echo "[2/3] informative-subset selection"
        python data_generation/filter_nyu_subset.py --config "$CONFIG"
    else
        echo ""
        echo "[2/3] filter  — SKIPPED (SKIP_FILTER=1)"
    fi

    echo ""
    echo "[3/3] ground truth: training subset + HELD-OUT TEST TAIL"
    echo "      --with-test-tail is NOT optional: the subset generator writes only TRAINING"
    echo "      indices, but the test loader reads the TAIL."
    python data_generation/generate_gt_nyu_subset.py --config "$CONFIG" --with-test-tail
fi

echo ""
echo "=============================================================="
echo " DONE. Sanity-check before training:"
echo "   python data_generation/count_files.py"
echo "   python data_generation/preview_complex_params.py --indices 0 1 2 3 4 5 6 7"
echo ""
echo " Then the sweep:"
echo "   scripts/g5k/train_sweep_nyu.sh"
echo "=============================================================="
