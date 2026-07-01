"""Learned loss-weighting heads and their shared feature trunk.

- ``WeightTrunk``        : conv trunk turning the encoder's last block (1664-ch)
                           into a 256-d feature vector for the weight heads.
- ``WeightHeadSigmoid``  : n-output sigmoid head (var1 of Techniques 1–3).
- ``WeightHeadSoftmax``  : n-output softmax head (var2 and Technique 4).
"""

import torch.nn as nn


class ConvBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1):
        super(ConvBNRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class WeightTrunk(nn.Module):
    """Maps the 1664-channel last dense block to a 256-d feature vector."""

    def __init__(self):
        super(WeightTrunk, self).__init__()
        self.Conv_1 = ConvBNRelu(1664, 512)
        self.Conv_2 = ConvBNRelu(512, 256)
        self.flat_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, last_block_feature):
        x = self.Conv_1(last_block_feature.float())
        x = self.Conv_2(x)
        x = self.flat_avg_pool(x)
        return self.flatten(x)


def _fc_stack(n_out):
    return nn.Sequential(
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

        nn.Linear(16, n_out),
    )


class WeightHeadSigmoid(nn.Module):
    """n-output sigmoid weighting head (each weight independent in (0, 1))."""

    def __init__(self, n=2):
        super(WeightHeadSigmoid, self).__init__()
        self.n = n
        self.classifier = _fc_stack(n)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.classifier(x))


class WeightHeadSoftmax(nn.Module):
    """n-output softmax weighting head (weights sum to 1)."""

    def __init__(self, n=3):
        super(WeightHeadSoftmax, self).__init__()
        self.n = n
        self.classifier = _fc_stack(n)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        return self.activation(self.classifier(x))
