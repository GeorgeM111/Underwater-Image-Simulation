"""3-channel decoder (residual / direct RGB output) — single canonical copy."""

import torch.nn as nn
import torch.nn.functional as F

from models.decoder_1ch import UpSample


class Decoder3Ch(nn.Module):
    """Decoder producing a 3-channel (RGB) output.

    Identical to ``Decoder1Ch`` except the final conv emits 3 channels.
    """

    def __init__(self, num_features=1664, decoder_width=1.0):
        super(Decoder3Ch, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features // 1 + 256, output_features=features // 2)
        self.up2 = UpSample(skip_input=features // 2 + 128, output_features=features // 4)
        self.up3 = UpSample(skip_input=features // 4 + 64, output_features=features // 8)
        self.up4 = UpSample(skip_input=features // 8 + 64, output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[3], features[4], features[6], features[8], features[12])
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)
