# --- repo-root path bootstrap ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Re-stamp a GT directory's physics manifest — but ONLY after PROVING the GT is unchanged.

WHY THIS EXISTS. The manifest hashes config VALUES, not effective physics. So a pure
refactor — renaming a knob, or splitting one multiplier into two whose product is identical —
changes the hash while every GT pixel stays exactly the same. The loader then refuses to run
against ground truth that is, in fact, perfectly valid, and you would have to burn hours
regenerating it for no reason.

The dangerous "fix" is to blindly rewrite the manifest. That is precisely the lie the manifest
exists to prevent, so this script does NOT do that. Instead it:

  1. picks N random indices that the manifest says it covers,
  2. RE-GENERATES their GT in memory with the CURRENT config,
  3. compares byte-for-byte against the .npy files on disk,
  4. re-stamps ONLY if every sampled pixel is identical, preserving covered_indices.

If a single byte differs, it refuses and tells you to regenerate. So the guarantee is
"verified equal", not "assumed equal".

Run:
    python scripts/restamp_manifest.py --dataset nyu --dry-run   # check only
    python scripts/restamp_manifest.py --dataset nyu             # verify + restamp
"""

import argparse
import json
import random

import numpy as np

from config import CONFIG
from utils.provenance import physics_fingerprint, write_physics_manifest, MANIFEST_NAME


def _regen_nyu(idx, cfg):
    """Regenerate index `idx`'s (haze, complex) GT in memory, exactly as the generator does."""
    from io import BytesIO
    import cv2
    from PIL import Image
    from numpy import load as npload
    from data_generation.data_2 import (ToTensorCustom, create_reorganize_dimension_custom,
                                        _PARAMS_DIR)
    from utils.physics import compute_complex_noise
    from utils.depth_range import DEPTH_CLIP_FRAC
    from data.nyu import loadZipToMem
    import torch

    data, rows = loadZipToMem(cfg.nyu_zip_path)
    a_mat_arr = npload(_os.path.join(_PARAMS_DIR, 'A_Mat_NYU_train.npy'))
    beta_mat_arr = npload(_os.path.join(_PARAMS_DIR, 'Beta_Mat_NYU_train.npy'))

    sample = rows[idx]
    image = Image.open(BytesIO(data[sample[0]]))
    depth = Image.open(BytesIO(data[sample[1]]))
    s = ToTensorCustom({'image': image, 'depth': depth}, False)
    image_half, depth01 = s['image_half_norm'], s['depth_half_norm_0_1']

    depth_m = torch.clamp(depth01 * 10.0, DEPTH_CLIP_FRAC * cfg.nyu_max_depth_m, cfg.nyu_max_depth_m)
    m, n = depth_m.shape[1], depth_m.shape[2]
    d_np = np.swapaxes(np.swapaxes(np.array(depth_m), 0, 2), 0, 1)
    d3 = cv2.cvtColor(d_np, cv2.COLOR_GRAY2RGB)

    beta_mat, a_mat = beta_mat_arr[idx], a_mat_arr[idx]
    beta_mod = create_reorganize_dimension_custom(beta_mat, m, n)
    a_mod = create_reorganize_dimension_custom(a_mat, m, n)
    unit = create_reorganize_dimension_custom([1.0, 1.0, 1.0], m, n)

    img_np = np.swapaxes(np.swapaxes(np.array(image_half), 0, 2), 0, 1)
    tx1 = np.exp(-np.multiply(beta_mod * cfg.nyu_beta_scale, d3))
    haze = np.add(np.multiply(img_np, tx1), np.multiply(a_mod, np.subtract(unit, tx1)))
    cx = compute_complex_noise(img_np, d3[:, :, 0] / 10.0, beta_mat, a_mat,
                               max_depth_m=cfg.nyu_max_depth_m, focal_px=cfg.nyu_focal_px,
                               clarity=cfg.nyu_water_clarity, beta_scale=cfg.nyu_beta_scale,
                               cfg=cfg, seed=int(getattr(cfg, 'random_seed', 42)) + idx) / 255.0

    haze_u8 = np.rint(np.clip(haze, 0.0, 1.0) * 255.0).astype(np.uint8)
    cx_u8 = np.rint(np.clip(cx, 0.0, 1.0) * 255.0).astype(np.uint8)
    return haze_u8, cx_u8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='nyu', choices=['nyu'])
    ap.add_argument('--samples', type=int, default=8, help='how many indices to verify')
    ap.add_argument('--dry-run', action='store_true', help='verify only; never write')
    args = ap.parse_args()

    gt_dir = CONFIG.nyu_gt_train_dir
    path = _os.path.join(gt_dir, MANIFEST_NAME)
    if not _os.path.exists(path):
        raise SystemExit('No %s in %s — nothing to restamp.' % (MANIFEST_NAME, gt_dir))

    with open(path) as fh:
        blob = json.load(fh)
    disk_hash = blob.get('physics_hash')
    covered = [int(i) for i in (blob.get('covered_indices') or [])]
    _, live_hash = physics_fingerprint(CONFIG)

    print('GT dir      : %s' % gt_dir)
    print('manifest    : physics_hash=%s  covered=%d indices' % (disk_hash, len(covered)))
    print('live config : physics_hash=%s' % live_hash)
    if disk_hash == live_hash:
        print('\nHashes already match — nothing to do.')
        return
    if not covered:
        raise SystemExit('Manifest records no covered_indices; cannot verify. Regenerate instead.')

    rng = random.Random(0)
    probe = rng.sample(covered, min(args.samples, len(covered)))
    print('\nVerifying %d sampled indices by REGENERATING them and comparing to disk...' % len(probe))

    mismatched = []
    for i in probe:
        hp = _os.path.join(gt_dir, '%dhaze_image.npy' % i)
        cp = _os.path.join(gt_dir, '%dcomplex_haze_image.npy' % i)
        if not (_os.path.exists(hp) and _os.path.exists(cp)):
            print('  idx %-6d MISSING on disk' % i); mismatched.append(i); continue
        d_h, d_c = np.load(hp), np.load(cp)
        r_h, r_c = _regen_nyu(i, CONFIG)
        # Compare on the stored dtype. Float dirs predate the uint8 switch -> not comparable.
        if d_h.dtype != np.uint8 or d_c.dtype != np.uint8:
            print('  idx %-6d on-disk dtype is %s/%s, not uint8 -> predates the current writer;'
                  ' cannot verify.' % (i, d_h.dtype, d_c.dtype))
            mismatched.append(i); continue
        nh = int((d_h != r_h).sum()); nc = int((d_c != r_c).sum())
        ok = (nh == 0 and nc == 0)
        print('  idx %-6d haze diff=%-6d complex diff=%-6d  %s' % (i, nh, nc, 'OK' if ok else 'MISMATCH'))
        if not ok:
            mismatched.append(i)

    if mismatched:
        raise SystemExit(
            '\nREFUSING TO RESTAMP: %d/%d sampled indices do not match the current physics.\n'
            'The GT on disk is genuinely STALE. Regenerate it:\n'
            '    scripts/regenerate_nyu_gt.sh' % (len(mismatched), len(probe)))

    print('\nAll %d sampled indices are BYTE-IDENTICAL to what the current config produces.' % len(probe))
    print('The hash changed only because the config was re-spelled, not because the physics moved.')
    if args.dry_run:
        print('\n--dry-run: not writing. Re-run without it to restamp.')
        return
    write_physics_manifest(gt_dir, CONFIG, covered_indices=covered)
    print('Restamped: %s -> %s (covered_indices preserved)' % (disk_hash, live_hash))


if __name__ == '__main__':
    main()
