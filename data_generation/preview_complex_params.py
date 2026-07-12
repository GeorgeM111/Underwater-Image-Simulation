# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Preview the COMPLEX (ricardo/v2) ground-truth for a handful of NYU images.

Generates, for each requested index, a side-by-side strip:

    [ original | depth | classical haze | complex GT ]

so you can judge the scattering blur / colour WITHOUT regenerating the whole
dataset. The scattering knobs can be overridden on the command line, so you can
sweep them in seconds and only commit to a full regen once you like the look:

    # current config values
    python data_generation/preview_complex_params.py --config config.yaml

    # halve the blur width and soften the direct-light falloff, on the fly
    python data_generation/preview_complex_params.py --gamma-scale 0.5 \
        --alpha-metric 0.35 0.35 0.15

    # explicit gamma_angular triple, specific images, custom output dir
    python data_generation/preview_complex_params.py \
        --gamma-angular 0.0049 0.0049 0.0021 --indices 0 5 12 30 --out /tmp/preview

Notes
-----
* Mirrors data_2.generate_and_save_haze_image exactly (same ToTensorCustom, same
  metre/[0,1] depth axes, same beta*clarity), so what you see is what the real GT
  generator would produce under the chosen parameters.
* Uses the PRE-COMPUTED NYU-train parameter matrices (Beta_Mat/A_Mat) when present,
  so the water type/airlight match the real GT; falls back to sampling a fresh
  Jerlov type per image if they are missing.
* Read-only w.r.t. config on disk: overrides are applied to the in-memory CONFIG
  only, so nothing here changes config.yaml.
"""

import argparse

import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from numpy import load

from config import CONFIG
from data.nyu import loadZipToMem
from data_2 import (ToTensorCustom, create_reorganize_dimension_custom,
                    _water_airlight, _seed_params)
from utils.physics import compute_complex_noise
import random


def _to_u8_hwc(chw_float01):
    """(3,H,W) float [0,1] -> (H,W,3) uint8 RGB."""
    arr = np.asarray(chw_float01)
    arr = np.swapaxes(arr, 0, 2)
    arr = np.swapaxes(arr, 0, 1)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _depth_vis(depth_hw_metres, max_depth_m):
    """(H,W) metres -> (H,W,3) uint8 grayscale for display."""
    d = np.clip(depth_hw_metres / max(max_depth_m, 1e-6), 0, 1) * 255.0
    d = d.astype(np.uint8)
    return cv2.cvtColor(d, cv2.COLOR_GRAY2RGB)


def main():
    ap = argparse.ArgumentParser(description="Preview complex GT under chosen scattering params.")
    ap.add_argument('--config', default=None, help='(informational) config path; CONFIG is already loaded')
    ap.add_argument('--indices', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7],
                    help='dataset indices to preview (post-shuffle order, matches training)')
    ap.add_argument('--gamma-scale', type=float, default=None,
                    help='multiply the current gamma_angular by this factor')
    ap.add_argument('--gamma-angular', type=float, nargs=3, default=None,
                    help='explicit gamma_angular triple (overrides --gamma-scale)')
    ap.add_argument('--alpha-metric', type=float, nargs=3, default=None,
                    help='explicit alpha_metric triple')
    ap.add_argument('--clarity', type=float, default=None,
                    help='override nyu_water_clarity (TURBIDITY: higher = murkier). Scales beta, '
                         'alpha and gamma together.')
    ap.add_argument('--airlight-min', type=float, default=None,
                    help='override nyu_airlight_brightness_min AND RESAMPLE the airlight for the '
                         'previewed images. The veil is (1-t)*A, so a dark A adds nothing and the '
                         'result is a colour grade, not an underwater scene. Use this to eyeball a '
                         'new floor WITHOUT regenerating the A_Mat parameter matrices.')
    ap.add_argument('--out', default=None, help='output dir (default: <repo>/preview_complex)')
    args = ap.parse_args()

    # ---- apply in-memory overrides (config.yaml on disk is untouched) ----
    if args.gamma_angular is not None:
        CONFIG.gamma_angular = list(args.gamma_angular)
    elif args.gamma_scale is not None:
        CONFIG.gamma_angular = [g * args.gamma_scale for g in CONFIG.gamma_angular]
    if args.alpha_metric is not None:
        CONFIG.alpha_metric = list(args.alpha_metric)
    if args.clarity is not None:
        CONFIG.nyu_water_clarity = float(args.clarity)
    if args.airlight_min is not None:
        CONFIG.nyu_airlight_brightness_min = float(args.airlight_min)

    out_dir = args.out or _os.path.join(_REPO_ROOT, 'preview_complex')
    _os.makedirs(out_dir, exist_ok=True)

    f_px = CONFIG.nyu_focal_px
    clarity = CONFIG.nyu_water_clarity
    print("gamma_angular =", CONFIG.gamma_angular)
    print("alpha_metric  =", CONFIG.alpha_metric)
    print("focal_px = %.2f  clarity = %.3f  -> sigma@10m = %.1f px (R/G), %.1f px (B)"
          % (f_px, clarity,
             f_px * CONFIG.gamma_angular[0] * clarity * 10.0,
             f_px * CONFIG.gamma_angular[2] * clarity * 10.0))

    data, nyu_dataset = loadZipToMem(CONFIG.nyu_zip_path)

    # Parameter matrices (water type + airlight). Fall back to fresh sampling.
    params_dir = _os.path.dirname(CONFIG.beta_mat_nyu_train)
    try:
        a_mat_arr = load(_os.path.join(params_dir, 'A_Mat_NYU_train.npy'))
        beta_mat_arr = load(_os.path.join(params_dir, 'Beta_Mat_NYU_train.npy'))
        have_params = True
        print("Loaded precomputed NYU-train parameter matrices (%d rows)." % len(beta_mat_arr))
    except Exception as exc:
        have_params = False
        _seed_params(0)
        print("No parameter matrices (%s) -> sampling a fresh water type per image." % exc)

    for idx in args.indices:
        sample = nyu_dataset[idx]
        image = Image.open(BytesIO(data[sample[0]]))
        depth = Image.open(BytesIO(data[sample[1]]))
        s = ToTensorCustom({'image': image, 'depth': depth}, False)

        image_half = s['image_half_norm']                 # (3,H,W) 0-1
        depth01 = s['depth_half_norm_0_1']                 # (1,H,W) 0-1
        depth_m = (depth01 * 10.0)                         # metres, (1,H,W)
        m, n = depth_m.shape[1], depth_m.shape[2]

        depth_m_np = np.array(depth_m)
        depth_m_np = np.swapaxes(depth_m_np, 0, 2)
        depth_m_np = np.swapaxes(depth_m_np, 0, 1)         # (H,W,1)
        depth_m_3d = cv2.cvtColor(depth_m_np, cv2.COLOR_GRAY2RGB)  # (H,W,3) metres

        image_half_np = np.array(image_half)
        image_half_np = np.swapaxes(image_half_np, 0, 2)
        image_half_np = np.swapaxes(image_half_np, 0, 1)   # (H,W,3) 0-1

        if have_params and args.airlight_min is None:
            beta_mat = list(beta_mat_arr[idx])
            a_mat = list(a_mat_arr[idx])
        elif have_params:
            # Keep this image's real water type, but RESAMPLE the airlight under the new floor.
            # The stored A_Mat was drawn with the OLD floor, so without this the preview would
            # still show the dark, veil-less airlights that make the GT look like a colour grade.
            beta_mat = list(beta_mat_arr[idx])
            random.seed(int(getattr(CONFIG, 'random_seed', 42)) + idx)
            a_mat = _water_airlight(
                random.uniform(CONFIG.nyu_airlight_brightness_min, 1.0), beta_mat)
        else:
            beta_mat = list(random.choice(CONFIG.jerlov_water_types))
            a_mat = _water_airlight(random.uniform(CONFIG.nyu_airlight_brightness_min, 1.0), beta_mat)

        beta_mod = create_reorganize_dimension_custom(
            np.asarray(beta_mat, dtype=np.float64) * clarity, m, n)
        a_mod = create_reorganize_dimension_custom(a_mat, m, n)
        unit = create_reorganize_dimension_custom([1.0, 1.0, 1.0], m, n)

        # Classical haze (matches generator + train-time recomputation).
        tx1 = np.exp(-np.multiply(beta_mod, depth_m_3d))
        haze = np.multiply(image_half_np, tx1) + np.multiply(a_mod, (unit - tx1))

        # Complex GT (uses the overridden gamma_angular / alpha_metric via CONFIG).
        complex_img = compute_complex_noise(
            image_half_np, depth_m_3d[:, :, 0] / 10.0, beta_mat, a_mat,
            max_depth_m=CONFIG.nyu_max_depth_m, focal_px=f_px, clarity=clarity) / 255.0

        strip = np.concatenate([
            _to_u8_hwc(np.swapaxes(np.swapaxes(image_half_np, 0, 1), 0, 2)),  # original
            _depth_vis(depth_m_3d[:, :, 0], CONFIG.nyu_max_depth_m),          # depth
            _to_u8_hwc(np.swapaxes(np.swapaxes(haze, 0, 1), 0, 2)),           # classical haze
            _to_u8_hwc(np.swapaxes(np.swapaxes(complex_img, 0, 1), 0, 2)),    # complex GT
        ], axis=1)

        out_path = _os.path.join(out_dir, "preview_%d.png" % idx)
        cv2.imwrite(out_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        print("  wrote", out_path)

    print("\nDone. Strips: [ original | depth | classical haze | complex GT ] in", out_dir)


if __name__ == '__main__':
    main()
