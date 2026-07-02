# --- repo-root path bootstrap (find the dir containing config.py) ---
import os as _os, sys as _sys
_p = _os.path.abspath(__file__)
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Select the most-informative subset of the NYU Depth V2 training set.

Ranks every image by a COMPOSITE information score and writes the indices of the
top-``subset_size`` images, to later drive underwater image-formation / GT
generation on a smaller but high-information subset.

Scoring rationale
------------------
Each image gets three raw terms:
  1. entropy   - skimage.measure.shannon_entropy of the grayscale image
                 (information-theoretic richness of the intensity distribution).
  2. gradient  - mean Sobel gradient magnitude, mean(sqrt(gx^2 + gy^2))
                 (spatial detail / sharpness / texture).
  3. depth     - depth range (max - min) over VALID (non-zero) depth pixels
                 (structural informativeness for the physical water model — a
                 flat-depth scene carries little geometric signal).

Because the three terms live on different scales, each is **min-max normalised
to [0, 1] across the WHOLE dataset** (a first pass collects all raw values, a
second pass normalises), then combined:

    score = w_ent*ent_norm + w_grad*grad_norm + w_depth*depth_norm

Everything is configurable (config.yaml + CLI overrides):
    subset_size -> config.nyu_subset_size        / --subset-size
    w_ent       -> config.nyu_subset_w_entropy    / --w-entropy
    w_grad      -> config.nyu_subset_w_gradient   / --w-gradient
    w_depth     -> config.nyu_subset_w_depth      / --w-depth
    output dir  -> config parameters dir          / --output-dir

Data / loading
--------------
This project stores NYU as a zip (config.nyu_zip_path) listed by
``data/nyu2_train.csv``; each row is ``<rgb>.jpg,<depth>.png`` (both 640x480,
depth is an 8-bit map with 0 = invalid). Images are read straight from the zip
via the shared ``data.nyu.loadZipToMem`` so the emitted indices align with the
exact ordering ``data.nyu`` uses for training (fixed random_state=0 shuffle).

Outputs (in the config parameters dir):
  * ``{subset_size}_filtered_nyu.npy``      - top-N indices, sorted by score desc
  * ``nyu_information_scores.csv``           - ALL indices with raw + normalised
                                               sub-scores and the final score

Run:
    python data_generation/filter_nyu_subset.py --config config.yaml
    python data_generation/filter_nyu_subset.py --subset-size 5000 --w-depth 0.5 --w-entropy 0.25 --w-gradient 0.25
"""

import os
import csv
import argparse
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from skimage.measure import shannon_entropy

from config import load_config
from data.nyu import loadZipToMem


# ---------------------------------------------------------------------------
# Raw per-image terms (defensive about grayscale-vs-RGB and invalid depth).
# ---------------------------------------------------------------------------

def _to_gray_u8(pil_img, resize):
    """PIL image -> single-channel uint8 numpy array (handles RGB or L)."""
    arr = np.asarray(pil_img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = arr.astype(np.uint8)
    if resize:
        arr = cv2.resize(arr, (resize, resize), interpolation=cv2.INTER_AREA)
    return arr


def raw_entropy(gray_u8):
    """Shannon entropy (bits) of the grayscale intensity histogram."""
    return float(shannon_entropy(gray_u8))


def raw_gradient(gray_u8):
    """Mean Sobel gradient magnitude — spatial detail / sharpness."""
    g = gray_u8.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def raw_depth_range(pil_depth):
    """(max - min) over valid (non-zero) depth pixels; 0 if none valid."""
    d = np.asarray(pil_depth)
    if d.ndim == 3:
        d = d[..., 0]
    valid = d[d > 0]
    if valid.size == 0:
        return 0.0
    return float(valid.max() - valid.min())


def _minmax(a):
    """Min-max normalise a 1-D array to [0, 1]; constant arrays -> zeros."""
    a = np.asarray(a, dtype=np.float64)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def main():
    parser = argparse.ArgumentParser(description='Select the most-informative NYU subset (composite score).')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--subset-size', type=int, default=None,
                        help='number of images to keep (default: config.nyu_subset_size)')
    parser.add_argument('--w-entropy', type=float, default=None, help='weight for the entropy term')
    parser.add_argument('--w-gradient', type=float, default=None, help='weight for the Sobel-gradient term')
    parser.add_argument('--w-depth', type=float, default=None, help='weight for the depth-range term')
    parser.add_argument('--pool', choices=['train', 'all'], default='train',
                        help="candidate pool: 'train' = only the training split (before "
                             "train_split_ratio, avoids test leakage); 'all' = whole set")
    parser.add_argument('--resize', type=int, default=0,
                        help='resize the (square) grayscale image before entropy/gradient; 0 = full res')
    parser.add_argument('--output-dir', default=None,
                        help='where to write outputs (default: the config parameters dir)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    subset_size = args.subset_size if args.subset_size is not None else cfg.nyu_subset_size
    w_ent = args.w_entropy if args.w_entropy is not None else cfg.nyu_subset_w_entropy
    w_grad = args.w_gradient if args.w_gradient is not None else cfg.nyu_subset_w_gradient
    w_depth = args.w_depth if args.w_depth is not None else cfg.nyu_subset_w_depth
    out_dir = args.output_dir if args.output_dir is not None else os.path.dirname(cfg.beta_mat_nyu_train)

    print("Composite NYU subset  |  size=%d  weights: ent=%.3f grad=%.3f depth=%.3f  pool=%s  resize=%s"
          % (subset_size, w_ent, w_grad, w_depth, args.pool, args.resize))

    data, rows = loadZipToMem(cfg.nyu_zip_path)
    n_total = len(rows)
    pool_end = int(cfg.train_split_ratio * n_total) if args.pool == 'train' else n_total
    candidates = list(range(0, pool_end))
    print("Total images: %d  |  candidate pool: %d" % (n_total, len(candidates)))

    # ---- Pass 1: collect RAW terms for every candidate ----
    ent_raw = np.zeros(len(candidates), dtype=np.float64)
    grad_raw = np.zeros(len(candidates), dtype=np.float64)
    depth_raw = np.zeros(len(candidates), dtype=np.float64)

    for k, idx in enumerate(tqdm(candidates, desc='scoring', unit='img')):
        row = rows[idx]
        try:
            gray = _to_gray_u8(Image.open(BytesIO(data[row[0]])), args.resize)
            ent_raw[k] = raw_entropy(gray)
            grad_raw[k] = raw_gradient(gray)
            depth_raw[k] = raw_depth_range(Image.open(BytesIO(data[row[1]]))) if len(row) > 1 else 0.0
        except Exception as exc:
            print("  [warn] idx %d failed: %s" % (idx, exc))
            ent_raw[k] = grad_raw[k] = depth_raw[k] = np.nan

    # ---- Pass 2: normalise across the whole dataset, then combine ----
    ent_n = _minmax(ent_raw)
    grad_n = _minmax(grad_raw)
    depth_n = _minmax(depth_raw)
    score = w_ent * ent_n + w_grad * grad_n + w_depth * depth_n

    idx_arr = np.asarray(candidates, dtype=np.int64)
    order = np.argsort(score)[::-1]                 # descending by score
    top_n = min(subset_size, len(candidates))
    top_indices = idx_arr[order[:top_n]]            # sorted by score DESC (as requested)

    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, "%d_filtered_nyu.npy" % subset_size)
    np.save(npy_path, top_indices)

    # Full ranking (all candidates) for inspection.
    csv_path = os.path.join(out_dir, "nyu_information_scores.csv")
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['index', 'entropy_raw', 'gradient_raw', 'depth_raw',
                    'entropy_norm', 'gradient_norm', 'depth_norm', 'score', 'selected'])
        selected = set(top_indices.tolist())
        for o in order:                              # rows written in score-desc order
            gi = int(idx_arr[o])
            w.writerow([gi, ent_raw[o], grad_raw[o], depth_raw[o],
                        ent_n[o], grad_n[o], depth_n[o], score[o], int(gi in selected)])

    # ---- Summary ----
    def stats(name, a):
        a = a[np.isfinite(a)]
        print("  %-9s min=%.4f  max=%.4f  mean=%.4f" % (name, a.min(), a.max(), a.mean()))

    print("\nRaw term stats over %d images:" % len(candidates))
    stats('entropy', ent_raw); stats('gradient', grad_raw); stats('depth', depth_raw)
    print("Chosen subset score: min=%.4f  max=%.4f  mean=%.4f"
          % (score[order[:top_n]].min(), score[order[:top_n]].max(), score[order[:top_n]].mean()))
    print("\nSaved:")
    print("  top-%d indices (score desc) -> %s" % (top_n, npy_path))
    print("  full ranking (all + sub-scores) -> %s" % csv_path)
    print("  first 8 selected indices: %s" % top_indices[:8].tolist())


if __name__ == '__main__':
    main()
