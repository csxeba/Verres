from typing import List, Dict

import tensorflow as tf

import verres as V
from verres.feature import Feature
from ..layers import block
from .base import VRSHead
from ..backbone import FeatureSpec


class OD(VRSHead):

    def __init__(self, config: V.Config, input_features: List[FeatureSpec]):
        super().__init__(config, input_features)
        pre_width = config.model.head_spec.get("head_convolution_width", 0)
        num_classes = config.class_mapping.num_classes
        self.heads = [
            block.VRSHead(
                output_feature=Feature(
                    name="heatmap",
                    stride=input_features[0].working_stride,
                    sparse=False,
                    dtype="float32",
                    depth=num_classes,
                ),
                pre_width=pre_width
            ),
            block.VRSHead(
                output_feature=Feature(
                    name="refinement",
                    stride=input_features[0].working_stride,
                    sparse=False,
                    dtype="float32",
                    depth=2,
                ),
                pre_width=pre_width
            ),
            block.VRSHead(
                output_feature=Feature(
                    name="box_wh",
                    stride=input_features[0].working_stride,
                    sparse=False,
                    dtype="float32",
                    depth=2,
                ),
                pre_width=pre_width
            )
        ]

    def call(self, inputs, training=None, mask=None) -> Dict[str, tf.Tensor]:
        return {
            head.output_feature.name: head(inputs[0]) for head in self.heads
        }


class CTDet(VRSHead):

    def __init__(self, config: V.Config, input_features: List[FeatureSpec]):
        super().__init__(config, input_features)
        spec = config.model.head_spec.copy()
        num_classes = config.class_mapping.num_classes
        self.hmap_head = block.VRSHead(spec.get("head_convolution_width", 0), output_width=num_classes)
        self.rreg_head = block.VRSHead(spec.get("head_convolution_width", 0), output_width=num_classes*2)
        self.boxx_head = block.VRSHead(spec.get("head_convolution_width", 0), output_width=num_classes*2)
        self.peak_nms = spec.get("peak_nms", 0.1)

    def call(self, inputs, training=None, mask=None):
        centroid_features, box_features = inputs
        hmap = self.hmap_head(centroid_features, training=training)
        rreg = self.rreg_head(centroid_features, training=training)
        boxx = self.boxx_head(centroid_features, training=training)
        return {"heatmap": hmap, "box_wh": boxx, "refinement": rreg}
