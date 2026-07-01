"""Shared model package.

    from models.model_builder import build_models
    model_1, model_2, model_3 = build_models(technique=4, variant='var2')
"""

from models.encoder import Encoder
from models.decoder_1ch import Decoder1Ch, UpSample
from models.decoder_3ch import Decoder3Ch
from models.weight_head import WeightTrunk, WeightHeadSigmoid, WeightHeadSoftmax
from models.model_builder import build_models, DepthModel, ImageModel

__all__ = [
    "Encoder", "Decoder1Ch", "Decoder3Ch", "UpSample",
    "WeightTrunk", "WeightHeadSigmoid", "WeightHeadSoftmax",
    "build_models", "DepthModel", "ImageModel",
]
