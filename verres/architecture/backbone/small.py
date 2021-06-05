import tensorflow as tf

import verres as V
from . import FeatureSpec, VRSBackbone
from ..layers import block


class SmallFCNN(VRSBackbone):

    def __init__(self, config: V.Config):

        spec = config.model.backbone_spec.copy()

        width_base = spec["width_base"]
        convolution_kwargs = {"batch_normalize": spec.get("batch_normalize", True),
                              "activation": spec.get("activation", "leakyrelu")}
        bottleneck_kwargs = convolution_kwargs.copy()
        bottleneck_kwargs["channel_expansion"] = spec.get("channel_expansion", 4)

        super().__init__([FeatureSpec("small_fcnn_output", working_stride=8, width=width_base*8)])
        self.layer_objects = [
            block.VRSConvolution(width=width_base, **convolution_kwargs),
            block.VRSResidualBottleneck(width=width_base*2, input_width=width_base, stride=2, **bottleneck_kwargs),
            block.VRSResidualBottleneck(width=width_base*4, input_width=width_base*2, stride=2, **bottleneck_kwargs),
            block.VRSResidualBottleneck(width=width_base*4, input_width=width_base*4, stride=1, **bottleneck_kwargs),
            block.VRSResidualBottleneck(width=width_base*8, input_width=width_base*4, stride=2, **bottleneck_kwargs),
            block.VRSResidualBottleneck(width=width_base*8, input_width=width_base*8, stride=1, **bottleneck_kwargs),
            block.VRSResidualBottleneck(width=width_base*8, input_width=width_base*8, stride=1, **bottleneck_kwargs),
            block.VRSResidualBottleneck(width=width_base*8, input_width=width_base*8, stride=1, **bottleneck_kwargs)]

    def _make_stage(self, depth: int, input_width: int, final_width: int):
        layers = [block.VRSResidualBottleneck(width=final_width, input_width=input_width, stride=2)]
        if depth > 1:
            processing_layers = [
                block.VRSResidualBottleneck(width=final_width, input_width=final_width, stride=1)
                for _ in range(depth-1)]
            layers.extend(processing_layers)
        return layers

    def preprocess_input(self, inputs):
        return tf.cast(inputs, tf.float32) / 255.

    def call(self, x, training=None, mask=None):
        for layer in self.layer_objects:
            x = layer(x, training=training)
        return [x]
