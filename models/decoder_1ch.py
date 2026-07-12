"""1-channel decoder (depth output) — single canonical copy.

Also defines the shared ``UpSample`` block reused by the 3-channel decoder.
"""

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

    The output is squashed into ``[y_min, y_max]`` by a SCALED SIGMOID. This is the fix
    for the "flat gray/white screen" collapse.

    Why it is needed. The head regresses ``y = DepthNorm(z) = depth_norm_max / d``, and
    the physics recovers metres as ``z = max_depth_m / y``. With a bare Conv2d the output
    is unbounded in (-inf, +inf) while its target lives in [1, 25], so ``z(y)`` is a
    HYPERBOLA in the raw output with three pathological regimes:
      (i)   an exploding band around y ~ 0.05-0.5, where dz/dy is ~300x its healthy value;
      (ii)  a dead zone at y < 0.02 (z > 500 m => t = exp(-beta*z) = 0), where the haze
            image degenerates to PURE AIRLIGHT — a flat, uniform, water-tinted frame;
      (iii) an ABSORBING STATE at y <= 0, where ``depthnorm_to_metres``'s
            ``torch.clamp(y, min=eps)`` has a backward of EXACTLY 0, so once the head
            crosses zero it can never come back. The collapse is permanent.

    Why sigmoid and NOT clamp/ReLU: a hard clamp or ReLU also has exactly-zero gradient
    outside its range, so it would merely RELOCATE the absorbing state rather than remove
    it. The scaled sigmoid makes all three regimes unreachable by construction and its
    derivative is never exactly zero, so the head can always recover.

    Bounds: y in [1, 25] corresponds to z in [max_depth_m/25, max_depth_m/1] = [0.4, 10] m
    for NYU — exactly the paper's supplementary clip ("target depth maps are clipped to
    the range [0.4, 10] in meters").
    """

    def __init__(self, num_features=1664, decoder_width=1.0, y_min=1.0, y_max=25.0):
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

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[3], features[4], features[6], features[8], features[12])
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        raw = self.conv3(x_d4)
        return self.y_min + (self.y_max - self.y_min) * torch.sigmoid(raw)
