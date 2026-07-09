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

Diversity (SSIM)
----------------
NYU is video sequences, so the highest-scoring images include many near-duplicate
consecutive frames. After ranking by score, a greedy SSIM pass keeps an image
only if its structural similarity (SSIM) to EVERY already-accepted image is below
``nyu_subset_ssim_threshold``; rejected (too-similar) images are skipped and
lower-ranked-but-distinct images backfill, until ``subset_size`` diverse images
are collected. If the pool runs dry first, the threshold is relaxed and the
rejects are re-scanned. Disable with ``--no-diversity`` for the plain top-N.

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


def _thumbnail(gray_u8, size):
    """Small [0,1] grayscale thumbnail (flattened) for fast SSIM comparison."""
    t = cv2.resize(gray_u8, (size, size), interpolation=cv2.INTER_AREA)
    return (t.astype(np.float32) / 255.0).reshape(-1)


def select_diverse(order, thumbs, target, threshold, relax=0.05):
    """Greedy SSIM-diversity selection.

    Walk candidates in score-DESC order (``order`` = candidate positions). Accept
    one only if its SSIM to EVERY already-accepted image is below ``threshold``
    (near-duplicates, e.g. consecutive frames, are skipped). Because rejects are
    skipped, lower-ranked-but-distinct images naturally BACKFILL. If the whole
    pool is exhausted before reaching ``target``, the threshold is relaxed by
    ``relax`` and the rejected images are re-scanned; finally any residual deficit
    is filled by score order. Returns accepted candidate positions (accept order).

    SSIM here is the global structural-similarity index between two thumbnails
    (single window = whole thumbnail), vectorised over the accepted set so the
    max-similarity check is one BLAS matvec per candidate.
    """
    N, P = thumbs.shape
    C1, C2 = 0.01 ** 2, 0.03 ** 2                      # data range = 1.0 ([0,1] thumbnails)
    A = np.zeros((target, P), dtype=np.float32)        # accepted thumbnails
    mu = np.zeros(target, dtype=np.float64)
    var = np.zeros(target, dtype=np.float64)
    cnt = 0
    accepted = []
    remaining = list(order)
    th = threshold

    pbar = tqdm(total=target, desc='diversity(SSIM)', unit='img')
    while cnt < target and remaining and th <= 1.0:
        still = []
        for k in remaining:
            if cnt >= target:
                still.append(k)
                continue
            x = thumbs[k]
            if cnt > 0:
                mu_x = float(x.mean())
                var_x = float(x.var())
                cov = A[:cnt].dot(x) / P - mu[:cnt] * mu_x
                num = (2 * mu_x * mu[:cnt] + C1) * (2 * cov + C2)
                den = (mu_x * mu_x + mu[:cnt] * mu[:cnt] + C1) * (var_x + var[:cnt] + C2)
                if float((num / den).max()) >= th:
                    still.append(k)
                    continue
            A[cnt] = x
            mu[cnt] = x.mean()
            var[cnt] = x.var()
            accepted.append(k)
            cnt += 1
            pbar.update(1)
        remaining = still
        if cnt < target and remaining:
            th += relax                                # relax and retry the rejected
    # threshold hit 1.0 but still short -> backfill by score
    for k in remaining:
        if cnt >= target:
            break
        A[cnt] = thumbs[k]
        accepted.append(k)
        cnt += 1
        pbar.update(1)
    pbar.close()
    return accepted, th


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
    parser.add_argument('--ssim-threshold', type=float, default=None,
                        help='drop an image if SSIM to any accepted image >= this (default config)')
    parser.add_argument('--ssim-thumb', type=int, default=None,
                        help='thumbnail size for the SSIM comparison (default config)')
    parser.add_argument('--ssim-relax', type=float, default=0.05,
                        help='if too few diverse images, relax the threshold by this and retry')
    parser.add_argument('--no-diversity', action='store_true',
                        help='disable the SSIM diversity pass (plain top-N by score)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    subset_size = args.subset_size if args.subset_size is not None else cfg.nyu_subset_size
    w_ent = args.w_entropy if args.w_entropy is not None else cfg.nyu_subset_w_entropy
    w_grad = args.w_gradient if args.w_gradient is not None else cfg.nyu_subset_w_gradient
    w_depth = args.w_depth if args.w_depth is not None else cfg.nyu_subset_w_depth
    out_dir = args.output_dir if args.output_dir is not None else os.path.dirname(cfg.beta_mat_nyu_train)
    ssim_th = args.ssim_threshold if args.ssim_threshold is not None else cfg.nyu_subset_ssim_threshold
    ssim_thumb = args.ssim_thumb if args.ssim_thumb is not None else cfg.nyu_subset_ssim_thumb

    print("Composite NYU subset  |  size=%d  weights: ent=%.3f grad=%.3f depth=%.3f  pool=%s  resize=%s"
          % (subset_size, w_ent, w_grad, w_depth, args.pool, args.resize))
    print("Diversity: %s  (SSIM threshold=%.2f, thumb=%dx%d)"
          % ("OFF" if args.no_diversity else "ON", ssim_th, ssim_thumb, ssim_thumb))

    data, rows = loadZipToMem(cfg.nyu_zip_path)
    n_total = len(rows)
    pool_end = int(cfg.train_split_ratio * n_total) if args.pool == 'train' else n_total
    candidates = list(range(0, pool_end))
    print("Total images: %d  |  candidate pool: %d" % (n_total, len(candidates)))

    # ---- Pass 1: collect RAW terms (+ SSIM thumbnail) for every candidate ----
    ent_raw = np.zeros(len(candidates), dtype=np.float64)
    grad_raw = np.zeros(len(candidates), dtype=np.float64)
    depth_raw = np.zeros(len(candidates), dtype=np.float64)
    thumbs = None if args.no_diversity else np.zeros((len(candidates), ssim_thumb * ssim_thumb), dtype=np.float32)

    for k, idx in enumerate(tqdm(candidates, desc='scoring', unit='img')):
        row = rows[idx]
        try:
            gray = _to_gray_u8(Image.open(BytesIO(data[row[0]])), args.resize)
            ent_raw[k] = raw_entropy(gray)
            grad_raw[k] = raw_gradient(gray)
            depth_raw[k] = raw_depth_range(Image.open(BytesIO(data[row[1]]))) if len(row) > 1 else 0.0
            if thumbs is not None:
                thumbs[k] = _thumbnail(gray, ssim_thumb)
        except Exception as exc:
            print("  [warn] idx %d failed: %s" % (idx, exc))
            ent_raw[k] = grad_raw[k] = depth_raw[k] = np.nan

    # ---- Pass 2: normalise across the whole dataset, then combine ----
    ent_n = _minmax(ent_raw)
    grad_n = _minmax(grad_raw)
    depth_n = _minmax(depth_raw)
    score = w_ent * ent_n + w_grad * grad_n + w_depth * depth_n
    score = np.where(np.isfinite(score), score, -np.inf)   # failed images sort last

    idx_arr = np.asarray(candidates, dtype=np.int64)
    order = np.argsort(score)[::-1]                 # candidate positions, score DESC
    top_n = min(subset_size, len(candidates))

    # ---- Selection ----
    if args.no_diversity:
        chosen_pos = list(order[:top_n])            # plain top-N by score
        final_th = None
    else:
        chosen_pos, final_th = select_diverse(order, thumbs, top_n, ssim_th, args.ssim_relax)
        print("Diversity pass kept %d images (final SSIM threshold %.2f)" % (len(chosen_pos), final_th))
    top_indices = idx_arr[np.asarray(chosen_pos, dtype=np.int64)]   # in acceptance (score-desc) order

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

    chosen_scores = score[np.asarray(chosen_pos, dtype=np.int64)]
    print("\nRaw term stats over %d images:" % len(candidates))
    stats('entropy', ent_raw); stats('gradient', grad_raw); stats('depth', depth_raw)
    print("Chosen subset score: min=%.4f  max=%.4f  mean=%.4f"
          % (chosen_scores.min(), chosen_scores.max(), chosen_scores.mean()))
    print("\nSaved:")
    print("  top-%d indices (score desc) -> %s" % (top_n, npy_path))
    print("  full ranking (all + sub-scores) -> %s" % csv_path)
    print("  first 8 selected indices: %s" % top_indices[:8].tolist())


if __name__ == '__main__':
    main()
