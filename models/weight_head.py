"""Learned loss-weighting heads and their shared feature trunk.

- ``WeightTrunk``        : conv trunk turning the encoder's last block (1664-ch)
                           into a 256-d feature vector for the weight heads.
- ``WeightHeadSigmoid``  : n-output sigmoid head (var1 of Techniques 1-3).
- ``WeightHeadSoftmax``  : n-output softmax head (var2 and Technique 4).

WHY THE WEIGHTS ARE FLOORED
---------------------------
The total loss is ``total = sum_i w_i * L_i`` where ``w`` is itself a model output
trained by the same gradient descent. That objective is LINEAR in ``w`` over the
box/simplex, so its exact minimiser is a VERTEX: one-hot on ``argmin_i L_i``. And
``d(total)/dz_j = w_j * (L_j - L_bar)`` is strictly positive for every above-average
loss, so descent drives those weights toward 0 — at which point the corresponding
loss stops receiving gradient, stays large, and the sign never flips. There is no
restoring force. This is not a tuning problem; it is a degenerate optimisation and it
WILL collapse. (Observed: every var2 run died; the depth head, whose only gradient
path is L_depth, froze at init and the physics degenerated to flat airlight.)

Flooring the weights away from 0 removes the vertex as an attainable optimum, so no
loss term can ever be switched off entirely. The paper (Eq. 12/13/17/21) specifies no
regulariser — its variants are ill-posed as written. This is a deliberate, documented
divergence; set ``weight_floor: 0.0`` in the config to recover the paper's exact
(degenerate) behaviour.
"""

import torch.nn as nn

from config import CONFIG


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


def _fc_stack(n_out, dropout=None):
    """DropOut-FC-ReLU chain from the paper's supplementary (256->128->64->32->16->n).

    ``dropout`` defaults to ``CONFIG.weight_head_dropout``. The supplementary specifies
    p=0.5 at every stage, but this head maps a 256-d pooled vector to just 2-4 scalars —
    there is essentially nothing to regularise, and heavy dropout has a real cost here:
    the weights are MULTIPLICATIVE on the loss terms, so dropout noise on them noises
    every gradient in the network. Worse, inverted dropout does not preserve the mean
    through 4 ReLUs + a sigmoid, so E_train[w] != w_eval — meaning the TRAIN objective
    and the VAL/checkpoint objective are literally different functions. Default lowered
    to 0.1; set ``weight_head_dropout: 0.5`` to restore the paper's value exactly.
    """
    if dropout is None:
        dropout = float(getattr(CONFIG, 'weight_head_dropout', 0.1))

    def block(i, o):
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers += [nn.Linear(i, o), nn.ReLU(inplace=True)]
        return layers

    return nn.Sequential(
        *block(256, 128),
        *block(128, 64),
        *block(64, 32),
        *block(32, 16),
        nn.Linear(16, n_out),
    )


def _floor() -> float:
    """Minimum mass any single learned loss weight may carry."""
    return float(getattr(CONFIG, 'weight_floor', 0.05))


class WeightHeadSigmoid(nn.Module):
    """n-output sigmoid weighting head; each weight independent in [floor, 1 - floor]."""

    def __init__(self, n=2):
        super(WeightHeadSigmoid, self).__init__()
        self.n = n
        self.classifier = _fc_stack(n)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        w = self.activation(self.classifier(x))
        f = _floor()
        # Squash (0,1) -> [f, 1-f]. Used as w*L1 + (1-w)*SSIM, so BOTH terms keep >= f.
        return f + (1.0 - 2.0 * f) * w


class WeightHeadSoftmax(nn.Module):
    """n-output softmax weighting head; weights sum to 1 and none falls below the floor."""

    def __init__(self, n=3):
        super(WeightHeadSoftmax, self).__init__()
        self.n = n
        self.classifier = _fc_stack(n)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        w = self.activation(self.classifier(x))
        f = _floor()
        # Affine map onto the floored simplex: still sums to 1, no component < f.
        return (1.0 - self.n * f) * w + f
