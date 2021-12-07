import verres as V
from ..backbone import VRSBackbone
from ..layers import block
from .base import VRSNeck


class Rescaler(VRSNeck):

    def __init__(self, config: V.Config, backbone: VRSBackbone):
        input_feature_spec = backbone.feature_specs[-1]
        super().__init__(backbone, [input_feature_spec], config.model.neck_spec["output_stride"])
        self.rescaler = block.VRSRescaler.from_strides(
            feature_stride=input_feature_spec.working_stride,
            target_stride=config.model.neck_spec["output_stride"],
            base_width=config.model.neck_spec.get("base_width", input_feature_spec.width),
            kernel_size=config.model.neck_spec.get("kernel_size", 3),
            batch_normalize=config.model.neck_spec.get("batch_normalize", True),
            activation=config.model.neck_spec.get("activation", "leakyrelu"))

    def call(self, inputs, **kwargs):
        features = self.backbone(inputs, **kwargs)
        features = self.rescaler(features[0], **kwargs)
        return [features]
