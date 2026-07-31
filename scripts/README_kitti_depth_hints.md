# More stable KITTI depth GT via the official Depth-Hints (SGM stereo)

We stabilise the KITTI dense-depth ground truth by replacing the IP-Basic morphological
densification of the LiDAR `completed_depth` (which *invents* the sky and large holes) with a
fusion of the real LiDAR measurements and dense **stereo depth hints**. The hints are produced by
the **official** nianticlabs/depth-hints `precompute_depth_hints.py` — a *geometric* Semi-Global
Matching of the left+right cameras, **not** a network prediction. The depth-estimation network of
our method is **not touched**; only the KITTI ground truth changes.

What "official" means here: `precompute_depth_hints.py` runs OpenCV `StereoSGBM` **12 times**
(3 block sizes × 4 disparity ranges) and **fuses** them per pixel by picking, for each pixel, the
disparity whose reprojection (SSIM+L1) error is lowest. That fusion is exactly why it is more
robust than a single SGBM pass, and it is the procedure to use.

Our only additions are two thin glue points (nothing about the hint generation is reimplemented):
`kitti_export_frame_list.py` (tells the official script *which* frames to do) and the fusion in
`data.kitti.load_frame_image_depth` (keeps the LiDAR where valid, fills holes/sky with the hint,
scale-aligned per frame, LiDAR restored exactly).

---

## The arguments (this is the part to understand)

`precompute_depth_hints.py` (run from inside the cloned depth-hints repo):

| argument | meaning | what to pass |
|---|---|---|
| `--data_path` | root of the **KITTI raw** data, in monodepth2 layout: `<data_path>/<date>/<drive>_sync/image_0X/data/<frame:010d>.<ext>` | `$PROJECT_DATA/kitti` (= `kitti_raw_dir`) |
| `--save_path` | where the hint `.npy` files are written, as `<save_path>/<date>/<drive>_sync/image_02/<frame:010d>.npy` | `$PROJECT_DATA/kitti/depth_hints` (= `kitti_depth_hints_dir`) |
| `--filenames` | a text file listing which frames to process, one `"<date>/<drive>_sync <frame_index> <side>"` per line. **Default = `splits/eigen_full/all_files.txt`** | our list (see below) |

### `--filenames` and the **Eigen split** (the `--eigen` thing)

The **Eigen split** (Eigen et al., 2014) is the *standard* KITTI train/test partition for monocular
depth: ~23k training frames and 697 test frames, listed in the repo as
`splits/eigen_full/{train,val,test,all}_files.txt`. By default `precompute_depth_hints.py` computes
hints for **that** split (`all_files.txt`). Related monodepth2/depth-hints scripts take a
`--split eigen` / `--eval_split eigen` flag that selects the same partition for training/evaluation.

**Why we override it:** our underwater pipeline does not use the Eigen split — it uses the KITTI
**depth-completion** split (Uhrig et al.: the `completed_depth/{train,val}` folders). Those are a
*different* set of frames. So we generate our own `--filenames` list covering exactly the frames
`data.kitti` loads, and point the official script at it. The *procedure* is unchanged; only the
*list of frames* is ours.

Other flags you may need: `--png` if your raw images are `.png` (KITTI raw ships `.png`; monodepth2
otherwise assumes the `.jpg` it converts to). The 12 SGBM settings are fixed inside the script.

---

## Run it

### 0. One-time: clone the repo + a legacy venv (no conda on Grid'5000)
The script imports the old monodepth2 code (`Image.ANTIALIAS`, old `torchvision.transforms`), so it
needs an older stack — use the `~/dh_venv` from before, not the project env.
```bash
git clone https://github.com/nianticlabs/depth-hints ~/depth-hints
python3 -m venv ~/dh_venv && source ~/dh_venv/bin/activate && pip install --upgrade pip
pip install torch==1.9.0 torchvision==0.10.0 "numpy<2" "pillow<7" opencv-python scikit-image
```

### 1. PROJECT env: list our frames in monodepth2 format
```bash
cd <repo> && source scripts/g5k/env.sh
python scripts/kitti_export_frame_list.py --config config.yaml
#   -> $PROJECT_DATA/parameters/kitti_depth_hints_files.txt   (lines: "<date>/<drive>_sync <frame> l")
```

### 2. LEGACY env: run the OFFICIAL precompute on our frames
```bash
deactivate; source ~/dh_venv/bin/activate; cd ~/depth-hints
python precompute_depth_hints.py \
    --data_path  $PROJECT_DATA/kitti \
    --save_path  $PROJECT_DATA/kitti/depth_hints \
    --filenames  $PROJECT_DATA/parameters/kitti_depth_hints_files.txt \
    --png        # only if your raw frames are .png
#   -> $PROJECT_DATA/kitti/depth_hints/<date>/<drive>_sync/image_02/<frame:010d>.npy
```

### 3. PROJECT env: enable the fused depth and regenerate the KITTI GT
```bash
# in config.yaml:  kitti_depth_source: "depth_hints"     # was "completed"
cd <repo> && source scripts/g5k/env.sh
python data_generation/generate_gt_kitti_train.py    # kitti_train_mode="all"  (or _subset.py for "subset")
python data_generation/generate_gt_kitti_test.py     # official val split
```

`data.kitti.load_frame_image_depth` now reads the official hint at
`<date>/<drive>_sync/image_02/<frame>.npy` (as `np.load(...)[0]`), fuses it with the LiDAR, and
returns the stabilised depth for both the underwater GT generator and the depth-branch target.
Set `kitti_depth_source` back to `"completed"` to restore the IP-Basic behaviour.

---

## ⚠️ Storage

The official hints are saved **float32 at full resolution** (~1.9 MB/frame), and they must persist
through training (depth is re-read every batch). For the full train split (~90k frames) that is
**~170 GB**; the 10k `subset` is ~19 GB. If disk is tight, keep `kitti_train_mode: "subset"`, or
run the export for the subset only. Keep `data_depth_annotated` — the fusion reads the LiDAR from it.
