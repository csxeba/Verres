import tensorflow as tf

import verres as V
from . import FeatureSpec, VRSBackbone
from ..layers import block


class SmallFCNN(VRSBackbone):

    def __init__(self, config: V.Config):

        width_base = config.model.backbone_spec["width_base"]

        super().__init__([FeatureSpec("small_fcnn_output", working_stride=8, width=width_base*8)])
        self.layer_objects = [
            block.VRSConvolution(width=width_base, activation="leakyrelu"),
            block.VRSResidualBottleneck(width=width_base*2, input_width=width_base, stride=2),
            block.VRSResidualBottleneck(width=width_base*4, input_width=width_base*2, stride=2),
            block.VRSResidualBottleneck(width=width_base*4, input_width=width_base*4, stride=1),
            block.VRSResidualBottleneck(width=width_base*8, input_width=width_base*4, stride=2),
            block.VRSResidualBottleneck(width=width_base*8, input_width=width_base*8, stride=1),
            block.VRSResidualBottleneck(width=width_base*8, input_width=width_base*8, stride=1),
            block.VRSResidualBottleneck(width=width_base*8, input_width=width_base*8, stride=1)]

    def preprocess_input(self, inputs):
        return tf.cast(inputs, tf.float32) / 255.

    def call(self, x, training=None, mask=None):
        for layer in self.layer_objects:
            x = layer(x, training=training)
        return [x]
