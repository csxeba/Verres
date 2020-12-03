from ..backbone.base import VRSBackbone
from ..backbone.base import FeatureSpec


class VRSNeck(VRSBackbone):

    def __init__(self, backbone: VRSBackbone, final_stride: int, final_feature_name="fused_features"):
        super().__init__([FeatureSpec(final_feature_name, working_stride=final_stride)])
        self.backbone = backbone

    def preprocess_input(self, inputs):
        return self.backbone.preprocess_input(inputs)
