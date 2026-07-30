# More stable KITTI depth ground truth via SGM stereo "depth hints"

The KITTI dense-depth ground truth can be stabilised by replacing the IP-Basic morphological
densification of the LiDAR `completed_depth` (which *invents* the sky and large holes) with a
**fusion of the real LiDAR measurements and dense SGM stereo depth** computed from the left and
right cameras. The "depth hints" here are exactly the classical Semi-Global Matching output that
Depth-Hints (Watson et al.) uses — a **geometric measurement from the two real cameras, not a
network prediction**. The depth-estimation network of the method is **not touched**.

For every frame: keep the LiDAR where it is valid, fill the holes/sky with the stereo depth after
a per-frame median scale alignment on the overlap (so the fill is metric-consistent with the
LiDAR and independent of the exact baseline). The measured pixels are restored exactly.

**No pretrained weights, no legacy environment, no GPU, no downloads** — only OpenCV + numpy,
which the project env already has. Requires the raw stereo pair (`image_02` + `image_03`) and
`calib_cam_to_cam.txt`, which the KITTI raw data provides.

## Steps (all in the PROJECT env, from the repo root)

```bash
cd <repo_dir>
source scripts/g5k/env.sh            # sets PROJECT_DATA and activates the project venv

# 1) list the frames training + eval actually touch (keeps the stereo-depth footprint small)
python scripts/kitti_export_frame_list.py --config config.yaml
#   -> $PROJECT_DATA/parameters/kitti_frames_for_depth.txt

# 2) compute dense SGM stereo depth for those frames (OpenCV; CPU is fine)
python scripts/kitti_depth_hints_sgm.py --config config.yaml
#   -> $PROJECT_DATA/kitti/depth_hints/<drive>/<frame>.npy   (float16, metres, raw resolution)
#   (add --all to process every completed-depth frame instead of just the used ones)

# 3) switch the GT source and regenerate the KITTI underwater ground truth
#    in config.yaml:   kitti_depth_source: "depth_hints"     # was "completed"
python data_generation/beta_atmosphere_kitti_train.py
python data_generation/beta_atmosphere_kitti_test.py
python data_generation/generate_gt_kitti_subset.py --config config.yaml   # or _train / _test
```

After step 3, `data.kitti.load_frame_image_depth` returns the fused (LiDAR + SGM stereo) depth for
every KITTI frame, so both the underwater GT generator and the depth-branch *target* use the more
stable depth automatically. Setting `kitti_depth_source` back to `"completed"` restores the
original IP-Basic behaviour.

## Tuning / notes

* SGM parameters: `--num_disparities` (default 192, multiple of 16 — raise for very near scenes),
  `--block_size` (default 7). Defaults follow the depth-hints SGBM settings.
* The stereo depth is only ever used to fill LiDAR *holes*; the LiDAR measurements are kept
  exactly, so SGM noise in textured regions cannot corrupt the measured part of the GT.
* Storage: `float16`, and only the **used** frames (subset + tail), a few GB. They are an
  intermediate — deletable after the KITTI GT is regenerated
  (`rm -r $PROJECT_DATA/kitti/depth_hints`); regenerate if you change the subset.
* Keep `data_depth_annotated` — the fusion reads the LiDAR measurements from it at generation time.
