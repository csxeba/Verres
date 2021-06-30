import math

import tensorflow as tf

import verres as V
from . import FeatureSpec, VRSBackbone
from ..layers import block


class SmallFCNN(VRSBackbone):

    # noinspection PyArgumentList
    def __init__(self, config: V.Config):

        self.cfg = config
        self.spec = spec = config.model.backbone_spec.copy()

        width_base = spec["width_base"]
        stage_type = spec.get("stage_type", None)
        if stage_type is None:
            print(" [Verres.SmallFCNN] - Unspecified stage_type, defaulting to 'simple' (can also be 'residual')")
            stage_type = "simple"
        convolution_kwargs = {"batch_normalize": spec.get("batch_normalize", True),
                              "activation": spec.get("activation", "leakyrelu")}

        self.output_feature_strides = spec["strides"]
        self.num_stages = int(round(math.log2(max(self.output_feature_strides))))

        super().__init__([FeatureSpec(f"feature_at_stride_{stride}", working_stride=stride, width=width_base*stride)
                          for stride in self.output_feature_strides])

        self.output_feature_names = {feature.working_stride: feature.layer_name for feature in self.feature_specs}
        self.layer_objects = [
            block.VRSConvolution(width=width_base, kernel_size=3, **convolution_kwargs)]
        if config.context.verbose > 1:
            print(f" [Verres.SmallFCNN] - Starting architecture with Convolution of width {width_base} and ksize 7")

        stage_builder = {
            "residual": block.VRSConvolutionStridedResidualStage,
            "simple": block.VRSConvolutionStrideStage}[stage_type]

        self.layer_objects.append(stage_builder(width_base, width_base*2, 1))
        self.layer_objects.append(stage_builder(width_base*2, width_base*4, 2))
        self.layer_objects.append(stage_builder(width_base*4, width_base*8, 4))

    def preprocess_input(self, inputs):
        inputs = tf.cast(inputs, tf.keras.backend.floatx()) / 255.
        return inputs

    def call(self, x, training=None, mask=None):
        outputs = []
        for i, stage in enumerate(self.layer_objects):
            x = stage(x)
            stride = 2 ** i
            if stride in self.output_feature_strides:
                outputs.append(x)
        return outputs
