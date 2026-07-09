# --- repo-root path bootstrap (find the dir containing config.py) ---
import os as _os, sys as _sys
_p = _os.path.abspath(__file__)
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Select the most-informative subset of KITTI raw frames.

Same composite information score as ``filter_nyu_subset.py``, adapted to KITTI:

  1. entropy   - skimage.measure.shannon_entropy of the grayscale image_02.
  2. gradient  - mean Sobel gradient magnitude (detail / sharpness).
  3. depth     - depth range (max - min) over VALID LiDAR pixels. KITTI has no
                 depth images, so depth is PROJECTED from the Velodyne point
                 cloud into image_02 (data.kitti.project_velodyne_to_depth) and
                 clipped to kitti_max_depth_m.

Each term is min-max normalised across the dataset (two passes), then combined:
    score = w_ent*ent_n + w_grad*grad_n + w_depth*depth_n

Indices are positions into ``data.kitti.list_frames`` (ordered by date/drive/frame),
so the SAME index selects the same frame for later GT generation and the loader.

Output (in the config parameters dir):
  * ``{subset_size}_filtered_kitti.npy``  - top-N indices, sorted by score desc
  * ``kitti_information_scores.csv``       - all frames + raw/normalised sub-scores

Run:
    python data_generation/filter_kitti_subset.py --config config.yaml
    python data_generation/filter_kitti_subset.py --subset-size 5000 --w-depth 0.5
"""

import os
import csv
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from tqdm import tqdm
from skimage.measure import shannon_entropy

from config import load_config
from data.kitti import list_completed_frames, read_completed_depth


def _minmax(a):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def main():
    parser = argparse.ArgumentParser(description='Select the most-informative KITTI subset (composite score).')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--subset-size', type=int, default=None,
                        help='number of frames to keep (default: config.kitti_subset_size)')
    parser.add_argument('--w-entropy', type=float, default=None)
    parser.add_argument('--w-gradient', type=float, default=None)
    parser.add_argument('--w-depth', type=float, default=None)
    parser.add_argument('--resize', type=int, default=0,
                        help='resize the (square) grayscale image before entropy/gradient; 0 = full res')
    parser.add_argument('--output-dir', default=None, help='default: the config parameters dir')
    parser.add_argument('--jobs', type=int, default=None, help='worker threads (default config.n_parallel_jobs)')
    parser.add_argument('--limit', type=int, default=0, help='only score the first N frames (for testing)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    subset_size = args.subset_size if args.subset_size is not None else cfg.kitti_subset_size
    w_ent = args.w_entropy if args.w_entropy is not None else cfg.kitti_subset_w_entropy
    w_grad = args.w_gradient if args.w_gradient is not None else cfg.kitti_subset_w_gradient
    w_depth = args.w_depth if args.w_depth is not None else cfg.kitti_subset_w_depth
    jobs = args.jobs if args.jobs is not None else cfg.n_parallel_jobs
    out_dir = args.output_dir if args.output_dir is not None else os.path.dirname(cfg.beta_mat_kitti_train)
    max_depth = cfg.kitti_max_depth_m

    print("KITTI subset  |  size=%d  weights: ent=%.3f grad=%.3f depth=%.3f  resize=%s  jobs=%d"
          % (subset_size, w_ent, w_grad, w_depth, args.resize, jobs))

    frames = list_completed_frames('train')
    if args.limit:
        frames = frames[:args.limit]
    n = len(frames)
    print("Total KITTI train frames (image_02 with completed depth): %d" % n)

    ent_raw = np.zeros(n, dtype=np.float64)
    grad_raw = np.zeros(n, dtype=np.float64)
    depth_raw = np.zeros(n, dtype=np.float64)

    def _score(k):
        f = frames[k]
        try:
            img = cv2.imread(f['image'])                      # BGR uint8
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape
            g = cv2.resize(gray, (args.resize, args.resize), interpolation=cv2.INTER_AREA) if args.resize else gray
            ent = float(shannon_entropy(g))
            gf = g.astype(np.float32)
            gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
            grad = float(np.sqrt(gx * gx + gy * gy).mean())
            # depth range over the VALID (measured) completed-depth pixels — do NOT
            # densify here (scoring should reflect the real measured spread).
            depth = read_completed_depth(f['depth'], max_depth, densify=False)
            valid = depth[depth > 0]
            drange = float(valid.max() - valid.min()) if valid.size else 0.0
            return k, ent, grad, drange
        except Exception as exc:
            print("  [warn] frame %d (%s) failed: %s" % (k, f['image'], exc))
            return k, np.nan, np.nan, np.nan

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for k, ent, grad, drange in tqdm(ex.map(_score, range(n)), total=n, desc='scoring', unit='frame'):
            ent_raw[k], grad_raw[k], depth_raw[k] = ent, grad, drange

    ent_n = _minmax(ent_raw)
    grad_n = _minmax(grad_raw)
    depth_n = _minmax(depth_raw)
    score = w_ent * ent_n + w_grad * grad_n + w_depth * depth_n

    order = np.argsort(score)[::-1]
    top_n = min(subset_size, n)
    top_indices = np.asarray(order[:top_n], dtype=np.int64)   # sorted by score desc

    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, "%d_filtered_kitti.npy" % subset_size)
    np.save(npy_path, top_indices)

    csv_path = os.path.join(out_dir, "kitti_information_scores.csv")
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['index', 'date', 'drive', 'frame', 'entropy_raw', 'gradient_raw', 'depth_raw',
                    'entropy_norm', 'gradient_norm', 'depth_norm', 'score', 'selected'])
        selected = set(top_indices.tolist())
        for o in order:
            f = frames[o]
            w.writerow([o, f['date'], f['drive'], f['frame'], ent_raw[o], grad_raw[o], depth_raw[o],
                        ent_n[o], grad_n[o], depth_n[o], score[o], int(o in selected)])

    def stats(name, a):
        a = a[np.isfinite(a)]
        print("  %-9s min=%.4f  max=%.4f  mean=%.4f" % (name, a.min(), a.max(), a.mean()))

    print("\nRaw term stats over %d frames:" % n)
    stats('entropy', ent_raw); stats('gradient', grad_raw); stats('depth', depth_raw)
    print("Saved:")
    print("  top-%d indices (score desc) -> %s" % (top_n, npy_path))
    print("  full ranking -> %s" % csv_path)
    print("  first 8 selected indices: %s" % top_indices[:8].tolist())


if __name__ == '__main__':
    main()
