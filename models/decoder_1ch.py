"""1-channel decoder (depth output) — single canonical copy.

Also defines the shared ``UpSample`` block reused by the 3-channel decoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)],
                             mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.convA(torch.cat([up_x, concat_with], dim=1))))


class Decoder1Ch(nn.Module):
    """Decoder producing a single-channel depth output in the reciprocal (DepthNorm) domain.

    The output is ``y = y_min + softplus(raw)`` — bounded BELOW at ``y_min`` and unbounded
    above. This is the fix for the "flat gray/white screen" collapse.

    Why a floor is needed. The head regresses ``y = DepthNorm(z) = depth_norm_max / d``, and
    the physics recovers metres as ``z = max_depth_m / y``. With a bare Conv2d the output
    is unbounded in (-inf, +inf) while its target lives in [1, 25], so ``z(y)`` is a
    HYPERBOLA in the raw output with three pathological regimes:
      (i)   an exploding band around y ~ 0.05-0.5, where dz/dy is ~300x its healthy value;
      (ii)  a dead zone at y < 0.02 (z > 500 m => t = exp(-beta*z) = 0), where the haze
            image degenerates to PURE AIRLIGHT — a flat, uniform, water-tinted frame;
      (iii) an ABSORBING STATE at y <= 0, where ``depthnorm_to_metres``'s
            ``torch.clamp(y, min=eps)`` has a backward of EXACTLY 0, so once the head
            crosses zero it can never come back. The collapse is permanent.

    Why NOT clamp/ReLU: both have exactly-zero gradient outside their range, so they merely
    RELOCATE the absorbing state instead of removing it.

    Why SOFTPLUS and not a scaled sigmoid. A scaled sigmoid
    ``y = y_min + (y_max - y_min) * sigmoid(raw)`` bounds y on BOTH sides — and that ceiling
    is a liability. A bounded output has a saturating derivative at each end, so a large
    gradient (e.g. the DenseDepth edge term at lambda_grad=1.0, ten times the L1 weight)
    drives ``raw`` far negative within ~10 optimizer steps. The sigmoid saturates, its
    derivative collapses to ~0, AND ``t = exp(-beta*z) -> 0`` at the same time, so NEITHER
    loss can pull it back: out_depth is pinned at the floor forever. This was observed and
    then reproduced/bisected: sigmoid head + lambda_grad=1.0 -> pinned at 1.0 from epoch 1
    (white depth panel, loss/depth frozen); softplus head survives the SAME loss weighting.

    Softplus keeps the bound that actually matters (the FLOOR, y >= y_min, i.e. z <= max_depth
    -> t can never collapse to 0 -> no flat-airlight degeneracy) and drops the one that does
    not (a ceiling only means "very close", z -> 0, t -> 1, which is harmless and which the
    depth loss corrects anyway).

    y >= y_min = 1 corresponds to z <= max_depth_m/1 = 10 m for NYU. ``y_max`` is retained
    only as the reference range for initialisation and for TensorBoard's bound check.
    """

    def __init__(self, num_features=1664, decoder_width=1.0, y_min=1.0, y_max=25.0,
                 y_init=4.0):
        super(Decoder1Ch, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features // 1 + 256, output_features=features // 2)
        self.up2 = UpSample(skip_input=features // 2 + 128, output_features=features // 4)
        self.up3 = UpSample(skip_input=features // 4 + 64, output_features=features // 8)
        self.up4 = UpSample(skip_input=features // 8 + 64, output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

        self.y_min = float(y_min)
        self.y_max = float(y_max)

        # Start the head near a typical NYU depth instead of at an arbitrary point.
        # y_init = 4 => z = max_depth_m / 4 = 2.5 m, the median NYU scene depth. Small weights
        # keep the pre-activation close to the bias at init, so the head begins in the
        # high-gradient region of softplus rather than anywhere near saturation.
        with torch.no_grad():
            nn.init.normal_(self.conv3.weight, mean=0.0, std=1e-3)
            target = max(float(y_init) - self.y_min, 1e-3)
            nn.init.constant_(self.conv3.bias, math.log(math.expm1(target)))  # softplus^-1

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[3], features[4], features[6], features[8], features[12])
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        raw = self.conv3(x_d4)
        # float() so the floor survives bf16 autocast: near y_min, bf16's 8-bit mantissa has a
        # spacing of ~0.008, which would quantise the depth signal away exactly where it matters.
        return self.y_min + F.softplus(raw.float())
