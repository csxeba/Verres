from typing import List

from ..backbone.abstract import VRSBackbone
from ..backbone.abstract import FeatureSpec


class VRSNeck(VRSBackbone):

    def __init__(self, backbone: VRSBackbone, input_feature_specs: List[FeatureSpec], output_stride: int):
        super().__init__([FeatureSpec(layer_name="fused_features", working_stride=output_stride)])
        self.backbone = backbone
        self.input_feature_specs = input_feature_specs

    def preprocess_input(self, inputs):
        return self.backbone.preprocess_input(inputs)

    def detect(self, inputs):
        outputs = self(inputs, training=False)
        centroids, whs, types, scores = self.postprocess(outputs)
        return centroids, whs, types, scores

    def call(self, inputs, **kwargs):
        raise NotImplementedError
