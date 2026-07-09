"""Assemble the (model_1, model_2, model_3) trio for a technique/variant.

The shared building blocks (``Encoder``, ``Decoder1Ch``, ``Decoder3Ch``, weight
heads) are composed into two generic networks:

- ``DepthModel``  : encoder + 1-channel decoder, with optional sigmoid (per-loss
                    balance) and softmax (global task weighting) heads.
- ``ImageModel``  : encoder + 3-channel decoder, with optional weight heads.

``build_models(technique, variant)`` returns ``(model_1, model_2, model_3)``.
``model_3`` is ``None`` for Techniques 1 and 2 (no direct branch).

Forward contracts
-----------------
DepthModel:
    base                 → depth
    sigmoid only         → (depth, w_sigmoid)
    sigmoid + softmax    → (depth, w_sigmoid, w_global_softmax)
ImageModel:
    base                 → image
    k weight heads       → (image, w_0[, w_1, ...])
"""

import torch.nn as nn

from config import CONFIG
from models.encoder import Encoder
from models.decoder_1ch import Decoder1Ch
from models.decoder_3ch import Decoder3Ch
from models.weight_head import WeightTrunk, WeightHeadSigmoid, WeightHeadSoftmax


class DepthModel(nn.Module):
    def __init__(self, pretrained=None, sigmoid_n=0, softmax_n=0):
        super(DepthModel, self).__init__()
        self.encoder = Encoder(pretrained)
        self.decoder = Decoder1Ch()
        self.sigmoid_n = sigmoid_n
        self.softmax_n = softmax_n
        if sigmoid_n or softmax_n:
            self.trunk = WeightTrunk()
        if sigmoid_n:
            self.w_sigmoid = WeightHeadSigmoid(sigmoid_n)
        if softmax_n:
            self.w_softmax = WeightHeadSoftmax(softmax_n)

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        if not (self.sigmoid_n or self.softmax_n):
            return depth
        feat = self.trunk(features[12])
        outs = [depth]
        if self.sigmoid_n:
            outs.append(self.w_sigmoid(feat))
        if self.softmax_n:
            outs.append(self.w_softmax(feat))
        return tuple(outs)


class ImageModel(nn.Module):
    """3-channel (RGB) output model with optional weight heads.

    Args:
        head_kind: 'sigmoid' or 'softmax' (ignored if num_heads == 0).
        num_heads: number of independent weight heads.
        head_n:    outputs per head.
    """

    def __init__(self, pretrained=None, head_kind=None, num_heads=0, head_n=2):
        super(ImageModel, self).__init__()
        self.encoder = Encoder(pretrained)
        self.decoder = Decoder3Ch()
        self.num_heads = num_heads
        if num_heads:
            self.trunk = WeightTrunk()
            head_cls = WeightHeadSoftmax if head_kind == 'softmax' else WeightHeadSigmoid
            self.heads = nn.ModuleList([head_cls(head_n) for _ in range(num_heads)])

    def forward(self, x):
        features = self.encoder(x)
        image = self.decoder(features)
        if not self.num_heads:
            return image
        feat = self.trunk(features[12])
        weights = [head(feat) for head in self.heads]
        return (image, *weights)


def _global_terms(technique):
    """Number of loss terms combined by the var2 global softmax head.

    Matches the paper's ablation:
      T1: depth + complex           (2, Eq. 13 — Ld vs Lp only; NO initial-degraded Lt)
      T2: depth + complex + haze    (3, Eq. 17 — adds Lt)
      T3/T4: + direct               (4, Eq. 21 — adds Lg)
    """
    if technique == 1:
        return 2
    if technique == 2:
        return 3
    return 4


def build_models(technique, variant):
    """Return (model_1, model_2, model_3) for the given technique and variant.

    Args:
        technique: 1, 2, 3 or 4.
        variant:   'base', 'var1' or 'var2'.
    """
    if technique not in (1, 2, 3, 4):
        raise ValueError("technique must be one of 1, 2, 3, 4")
    if variant not in ('base', 'var1', 'var2'):
        raise ValueError("variant must be one of 'base', 'var1', 'var2'")

    pretrained = CONFIG.pretrained_encoder
    has_direct = technique in (3, 4)
    # Techniques 1–3 use sigmoid per-loss balance heads; Technique 4 uses
    # softmax heads over (L1, SSIM, Perceptual).
    is_tech4 = technique == 4
    head_kind = 'softmax' if is_tech4 else 'sigmoid'
    head_n = 3 if is_tech4 else 2

    if variant == 'base':
        model_1 = DepthModel(pretrained)
        model_2 = ImageModel(pretrained)
        model_3 = ImageModel(pretrained) if has_direct else None
        return model_1, model_2, model_3

    if variant == 'var1':
        model_1 = DepthModel(pretrained, sigmoid_n=2)
        # model_2 holds the weights for the complex (L_p) and haze (L_t) losses.
        # Techniques 1–3: one sigmoid head of size 2 ([0]=L_p, [1]=L_t).
        # Technique 4: two softmax heads of size 3 (w_Residue, w_Deg).
        if is_tech4:
            model_2 = ImageModel(pretrained, head_kind='softmax', num_heads=2, head_n=3)
            model_3 = ImageModel(pretrained, head_kind='softmax', num_heads=1, head_n=3)
        else:
            model_2 = ImageModel(pretrained, head_kind='sigmoid', num_heads=1, head_n=2)
            model_3 = ImageModel(pretrained, head_kind='sigmoid', num_heads=1, head_n=2) if has_direct else None
        return model_1, model_2, model_3

    # variant == 'var2': add a global softmax weighting head on model_1.
    softmax_n = _global_terms(technique)
    model_1 = DepthModel(pretrained, sigmoid_n=2, softmax_n=softmax_n)
    if is_tech4:
        model_2 = ImageModel(pretrained, head_kind='softmax', num_heads=2, head_n=3)
        model_3 = ImageModel(pretrained, head_kind='softmax', num_heads=1, head_n=3)
    else:
        model_2 = ImageModel(pretrained, head_kind='sigmoid', num_heads=1, head_n=2)
        model_3 = ImageModel(pretrained, head_kind='sigmoid', num_heads=1, head_n=2) if has_direct else None
    return model_1, model_2, model_3
