import tensorflow as tf

import verres as V
from ..backbone import VRSBackbone
from ..layers import block
from .base import VRSNeck


class AutoFusion(VRSNeck):

    def __init__(self,
                 config: V.Config,
                 backbone: VRSBackbone):

        spec = config.model.neck_spec.copy()

        super().__init__(backbone=backbone,
                         input_feature_specs=backbone.feature_specs,
                         output_stride=spec["output_stride"])

        self.branches = []
        self.final_width = spec["output_width"]
        for ftr in backbone.feature_specs:
            self.branches.append(block.VRSRescaler.from_strides(
                feature_stride=ftr.working_stride,
                target_stride=spec["output_stride"],
                base_width=ftr.width,
                kernel_size=3,
                batch_normalize=spec.get("batch_normalize", True),
                activation=spec.get("activation", "leakyrelu")))
        if self.final_width is not None:
            self.final_conv = block.VRSConvolution(self.final_width,
                                                   spec.get("activation", "leakyrelu"),
                                                   spec.get("batch_normalize", True),
                                                   kernel_size=1)

    def call(self, inputs, training=None, mask=None):
        features = self.backbone(inputs)
        result = []
        for x, branch in zip(features, self.branches):
            result.append(branch(x))
        result = tf.concat(result, axis=-1)
        if self.final_width is not None:
            result = self.final_conv(result)
        return [result]
