from typing import List, Dict

import tensorflow as tf
from tensorflow import image as tfimage

import verres as V
from ..backbone import FeatureSpec
from ..head import VRSHead
from ..layers import block


class SemSeg(VRSHead):
    
    def __init__(self, config: V.Config, input_features: List[FeatureSpec]):
        super().__init__(config, input_features)
        num_classes = config.class_mapping.num_classes
        assert len(input_features) == 1
        input_feature = input_features[0]
        assert input_feature.working_stride == 4
        self.working_stride = input_feature.working_stride
        self.sseg = block.VRSHead(pre_width=0, output_width=num_classes + 1)

    @tf.function
    def call(self, inputs: List[tf.Tensor], training=False, mask=None):
        logits_low = self.sseg(inputs[0])
        output_shape = [logits_low.shape[1] * self.working_stride, logits_low.shape[2] * self.working_stride]
        logits_high = tf.image.resize(inputs, output_shape)
        return logits_high


class Panoptic(VRSHead):

    def __init__(self, config: V.Config, input_features: List[FeatureSpec]):
        super().__init__(config, input_features)
        num_classes = config.class_mapping.num_classes
        assert len(input_features) == 1 and input_features[0].working_stride == 4

        self.proc_4 = block.VRSConvolution(width=64, kernel_size=1, activation="leakyrelu")
        self.hmap = block.VRSHead(pre_width=0, output_width=num_classes)
        self.rreg = block.VRSHead(pre_width=0, output_width=num_classes * 2)
        self.sseg = block.VRSHead(pre_width=0, output_width=num_classes + 1)
        self.iseg = block.VRSHead(pre_width=0, output_width=2)

    # @tf.function
    def call(self, inputs, training=None, mask=None):
        ftr4 = inputs[0]
        output_shape = (ftr4.shape[1] * 4, ftr4.shape[2] * 4)

        ftr4 = self.proc_4(ftr4)

        hmap = self.hmap(ftr4, training=training, mask=mask)
        iseg = self.iseg(ftr4, training=training, mask=mask)
        sseg = self.sseg(ftr4, training=training, mask=mask)

        iseg = tfimage.resize(iseg, size=output_shape)
        sseg = tfimage.resize(sseg, size=output_shape)  # These are logits

        return {"heatmap": hmap, "instance_segmentation": iseg, "semantic_segmentation": sseg}
