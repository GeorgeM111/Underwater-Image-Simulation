"""
model_3D_weight_tech4.py

3-channel output model (same DenseNet-169 encoder/decoder as model_3D.py) with
configurable softmax weight heads for Technique 4 Variants 1 & 2.

Usage:
    model_2 = PTModel(num_weight_heads=2)   # heads: w_Residue (L_p) + w_Deg (L_t)
    model_3 = PTModel(num_weight_heads=1)   # head:  w_Dir (L_g)

Forward returns:
    num_weight_heads=1 → (image_3ch, w_softmax_3)
    num_weight_heads=2 → (image_3ch, w_softmax_3_head0, w_softmax_3_head1)

Each weight vector has shape [batch, 3] and sums to 1 (softmax), representing
learned weights for (L1, SSIM, Perceptual) components of the corresponding loss.
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA     = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB     = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)],
                             mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.convA(torch.cat([up_x, concat_with], dim=1))))


class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width=1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)
        self.up1   = UpSample(skip_input=features // 1 + 256, output_features=features // 2)
        self.up2   = UpSample(skip_input=features // 2 + 128, output_features=features // 4)
        self.up3   = UpSample(skip_input=features // 4 + 64,  output_features=features // 8)
        self.up4   = UpSample(skip_input=features // 8 + 64,  output_features=features // 16)
        # 3-channel RGB output
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


class ConvBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=1, stride=1, padding=1):
        super(ConvBNRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class FlatAvgPool(nn.Module):
    def __init__(self):
        super(FlatAvgPool, self).__init__()
        self.flat_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.just_flatten  = nn.Flatten()

    def forward(self, x):
        return self.just_flatten(self.flat_avg_pool(x))


class WeightHead3(nn.Module):
    """
    FC branch that produces 3 softmax weights for (L1, SSIM, Perceptual) balance.
    Input:  256-d feature vector
    Output: [batch, 3], values in (0,1) summing to 1.
    """
    def __init__(self):
        super(WeightHead3, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),

            nn.Linear(16, 3)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.classifier(x))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = models.densenet169(pretrained=True)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features


class PTModel(nn.Module):
    """
    Args:
        num_weight_heads (int):
            1  →  model_3 usage: returns (image, w_dir)
            2  →  model_2 usage: returns (image, w_residue, w_deg)
    """

    def __init__(self, num_weight_heads=1):
        super(PTModel, self).__init__()
        assert num_weight_heads in (1, 2), "num_weight_heads must be 1 or 2"

        self.encoder    = Encoder()
        self.decoder    = Decoder()

        # Shared conv trunk for weight-head feature extraction
        self.Conv_1      = ConvBNRelu(1664, 512, 7, 1, 1)
        self.Conv_2      = ConvBNRelu(512,  256, 7, 1, 1)
        self.AvgPoolFlat = FlatAvgPool()

        # One or two independent 3-output softmax heads
        self.weight_heads = nn.ModuleList(
            [WeightHead3() for _ in range(num_weight_heads)]
        )

    def forward(self, x):
        encode_part       = self.encoder(x)
        last_block_feat   = encode_part[12]

        x_conv            = self.Conv_1(last_block_feat.float())
        x_conv            = self.Conv_2(x_conv)
        linear_features   = self.AvgPoolFlat(x_conv)

        weight_outputs    = [head(linear_features) for head in self.weight_heads]
        decode_part       = self.decoder(encode_part)

        # Returns (image, w0) or (image, w0, w1)
        return (decode_part, *weight_outputs)
