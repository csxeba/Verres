import math
from typing import List

import tensorflow as tf
from tensorflow.keras import layers as tfl

from ..layers import block
from .application import ApplicationBackbone
from . import FeatureSpec


class SideModel(block.VRSLayerStack):

    def __init__(self,
                 working_stride: int,
                 final_width: int,
                 base_width: int = 8,
                 base_depth: int = 1,
                 batch_norm: bool = True):

        super().__init__()

        self.working_stride = working_stride
        self.base_width = base_width

        self.layer_objects = []
        current_stride = 1
        current_width = base_width
        current_depth = base_depth
        for i in range(10):
            self.layer_objects.extend(
                [block.VRSConvolution(current_width, activation="leakyrelu", batch_normalize=batch_norm)
                 for _ in range(current_depth)])
            self.layer_objects.append(tfl.MaxPool2D())
            current_width *= 2
            current_depth *= 2
            current_stride *= 2
            if current_stride >= working_stride:
                if current_stride > working_stride:
                    raise RuntimeError(f"Misspecified working stride: {working_stride}")
                break
        else:
            raise RuntimeError(f"Misspecified or too great working stride: {working_stride}")
        self.layer_objects.append(
            block.VRSConvolution(final_width, batch_normalize=batch_norm, initializer="zeros"))


class SideTunedBackbone(ApplicationBackbone):

    def __init__(self,
                 backbone_name: str,
                 feature_specs: List[FeatureSpec],
                 side_models: List[tf.keras.Model] = "default",
                 input_shape=None,
                 fixed_batch_size=None,
                 weights="imagenet"):

        super().__init__(backbone_name, feature_specs, input_shape, fixed_batch_size, weights)
        self.wrapped_model.trainable = False
        self.side_models = side_models
        self.blending_factors = tf.Variable([0.5]*len(self.side_models), trainable=True)
        if self.side_models == "default":
            self._generate_default_side_models()

    def _generate_default_side_models(self):
        side_models = []
        base_width = 8
        for spec in self.feature_specs:
            stride_level = int(math.log2(spec.working_stride))
            side_models.append(
                SideModel(spec.working_stride,
                          final_width=spec.width,
                          base_width=base_width * stride_level,
                          base_depth=stride_level,
                          batch_norm=True))
        self.side_models = side_models

    # @tf.function
    def call(self, x, training=None, mask=None):
        features = self.wrapped_model(x)
        if len(self.feature_specs) == 1:
            features = [features]

        sided_features = []
        for i in range(len(self.feature_specs)):
            side_feature = self.side_models[i](x)
            sided_features.append(
                features[i] * self.blending_factors[i] + side_feature * (tf.ones(1) - self.blending_factors[i])
            )
        return sided_features
