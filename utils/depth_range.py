"""The canonical depth axis — SINGLE SOURCE OF TRUTH.

Deliberately dependency-free (no torch, no torchvision, no cv2) so that every layer —
the transforms, the GT generator, the physics, the model head — can import it without
dragging in a heavy dependency or risking a circular import.

THE AXES
--------
    z            depth in METRES.                        NYU: [0.4, 10]
    d            the stored depth tensor, d = z * 100.   NYU: [40, 1000]
                 (an 8-bit PNG in [0,1] scaled by depth_norm_max = 1000)
    y            the network's TARGET, y = DepthNorm(d) = 1000 / d.   NYU: [1, 25]
                 => y = 10 / z, i.e. exactly the paper's reciprocal target y = m / y_orig
                 with m = 10 m (paper Sec. 4.2.1).

The paper's supplementary clips target depth to [0.4, 10] m. That is what fixes y_max = 25.
The repo previously clamped d to [10, 1000] => y in [1, 100], a 4x wider range than the
paper's, which (a) let near-zero / missing-depth pixels reach the physics and (b) meant the
depth head's output range did not match its target range.

FOUR independent copies of this clamp used to exist (utils/transforms, data_2 x2,
data/make3d, data/kitti). They would inevitably drift. Import from here instead.

WHY THE SAME [1, 25] WORKS FOR EVERY DATASET
--------------------------------------------
The near clip is a FRACTION of max_depth (DEPTH_CLIP_FRAC = 0.04), so:
    NYU     max 10 m -> near 0.4 m  -> y in [1, 25]
    Make3D  max 80 m -> near 3.2 m  -> y in [1, 25]
    KITTI   max 80 m -> near 3.2 m  -> y in [1, 25]
so the depth head's bounds are dataset-independent, and 3.2 m is a sane near clip for both
outdoor sets.
"""

# Stored-depth tensor axis (d = z * 100).
DEPTH_CLAMP_MIN = 40.0     # 0.4 m
DEPTH_CLAMP_MAX = 1000.0   # 10 m

# Network target axis (y = 1000 / d). These are the depth head's output bounds.
DEPTH_NORM_MIN = 1.0       # DepthNorm(1000) -> farthest
DEPTH_NORM_MAX = 25.0      # DepthNorm(40)   -> nearest

# Near clip as a fraction of each dataset's max_depth_m.
DEPTH_CLIP_FRAC = 0.04

# SSIM's dynamic range for the depth domain. MUST be a constant: it was previously
# float(depth_n.max()), making SSIM's C1=(0.01L)^2 / C2=(0.03L)^2 a per-BATCH random
# variable (a ~45x swing in C1/C2 between neighbouring batches for identical prediction
# quality). val_range must be the fixed dynamic range of the signal domain.
DEPTH_VAL_RANGE = 25.0
