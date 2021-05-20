import tensorflow as tf

import verres as V

from . import backbone as _backbone
from . import neck as _neck
from . import head as _head

from .backbone import VRSBackbone
from .head import VRSHead


class VRSArchitecture(tf.keras.Model):

    def __init__(self,
                 config: V.Config,
                 backbone: VRSBackbone,
                 head: VRSHead):

        super().__init__()
        self.cfg = config
        self.backbone = backbone
        if len(backbone.feature_specs) != 2:
            print(" [Verres.architecture] - Single backbone mode is active.")
            self.single_backbone_mode = True
        else:
            self.single_backbone_mode = False
        self.head = head
        if config.model.weights is not None:
            print(" [Verres.architecture] - Loading weights from", config.model.weights)
            self.build((None, None, None, 3))
            self.load_weights(config.model.weights)

    @classmethod
    def factory(cls, config: V.Config):
        backbone = _backbone.factory(config)
        fused = _neck.factory(config, backbone)
        if fused is None:
            fused = backbone
        detector = _head.factory(config)
        architecture = cls(config, fused, detector)
        return architecture

    def preprocess_input(self, inputs):
        return self.backbone.preprocess_input(inputs)

    def call(self, inputs, training=None, mask=None):
        features = self.backbone(inputs)
        if self.single_backbone_mode:
            features = [features[0], features[0]]
        output = self.head(features)
        return output

    def postprocess(self, predictions):
        return self.head.postprocess_network_output(predictions)

    def detect(self, inputs):
        inputs = self.preprocess_input(inputs)
        outputs = self(inputs, training=False)
        detection = self.postprocess(outputs)
        return detection
