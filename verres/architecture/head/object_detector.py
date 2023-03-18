from typing import List

import tensorflow as tf

import verres as V
from ..layers import block
from .base import VRSHead
from ..backbone import FeatureSpec


class OD(VRSHead):

    def __init__(self, config: V.Config, input_features: List[FeatureSpec]):
        super().__init__(config, input_features)
        spec = config.model.head_spec.copy()
        num_classes = config.class_mapping.num_classes
        if len(input_features) == 1:
            input_features = [input_features[0], input_features[0]]
        self.centroid_feature_spec, self.box_feature_spec = input_features
        nearest_po2 = V.utils.numeric_utils.ceil_to_nearest_power_of_2(num_classes)

        self.hmap_head = block.VRSHead(
            spec.get("head_convolution_width", nearest_po2),
            output_width=num_classes)
        if config.context.verbose > 1:
            print(f" [Verres.OD] - Added Heatmap head with widths: {nearest_po2} -> {num_classes}")

        self.rreg_head = block.VRSHead(
            spec.get("head_convolution_width", nearest_po2*2),
            output_width=2)
        if config.context.verbose > 1:
            print(f" [Verres.OD] - Added Refinement head with widths: {nearest_po2*2} -> {2}")

        self.boxx_head = block.VRSHead(
            spec.get("head_convolution_width", nearest_po2*2),
            output_width=2)
        if config.context.verbose > 1:
            print(f" [Verres.OD] - Added Box head with widths: {nearest_po2*2} -> {4}")

        self.peak_nms = spec.get("peak_nms", 0.1)

    def call(self, inputs, training=None, mask=None):
        centroid_features, box_features = inputs
        hmap = self.hmap_head(centroid_features, training=training)
        rreg = self.rreg_head(centroid_features, training=training)
        boxx = self.boxx_head(centroid_features, training=training)
        hmap = tf.cond(training,
                       true_fn=lambda: hmap,
                       false_fn=lambda: tf.nn.sigmoid(hmap))
        return {"heatmap": hmap, "box_wh": boxx, "refinement": rreg}


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
