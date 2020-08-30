import tensorflow as tf

from .base import FeatureSpec, VRSBackbone
from verres.layers import block


class FeatureFuser(VRSBackbone):

    def __init__(self,
                 backbone: VRSBackbone,
                 final_stride: int,
                 base_width: int = 16,
                 final_width: int = None,
                 batch_normalize: bool = True,
                 activation: str = "leakyrelu"):

        super().__init__([FeatureSpec("fused_features", working_stride=final_stride)])
        self.backbone = backbone
        self.branches = []
        self.final_width = final_width
        for spec in backbone.feature_specs:
            self.branches.append(
                block.VRSDownscaleBlock(
                    starting_stride=spec.working_stride,
                    target_stride=final_stride,
                    width_base=base_width * spec.working_stride,
                    depth_base=spec.working_stride,
                    activation=activation,
                    batch_normalize=batch_normalize))
        if self.final_width is not None:
            self.final_conv = block.VRSConvolution(final_width, activation, batch_normalize, kernel_size=1)

    def call(self, inputs, training=None, mask=None):
        features = self.backbone(inputs)
        result = []
        for x, branch in zip(features, self.branches):
            result.append(branch(x))
        result = tf.concat(result, axis=-1)
        if self.final_width is not None:
            result = self.final_conv(result)
        return [result]
