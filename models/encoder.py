"""Shared DenseNet-169 encoder (single canonical copy)."""

import torch.nn as nn
from torchvision import models

from config import CONFIG


class Encoder(nn.Module):
    """DenseNet-169 feature extractor.

    The forward pass returns the list of intermediate feature maps (the same
    structure consumed by the decoders), where index 12 is the last dense block.

    Args:
        pretrained: whether to load ImageNet weights. Defaults to
            ``config.pretrained_encoder``.
    """

    def __init__(self, pretrained=None):
        super(Encoder, self).__init__()
        if pretrained is None:
            pretrained = CONFIG.pretrained_encoder
        self.original_model = models.densenet169(pretrained=pretrained)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features
