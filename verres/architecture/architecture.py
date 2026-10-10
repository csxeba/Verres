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
            # Keras 3 creates the variables lazily, so one traced call must precede load_weights.
            self(tf.zeros((1, *config.model.input_shape_hw, 3), dtype=tf.float32))
            self.load_weights(config.model.weights)

    @classmethod
    def factory(cls, config: V.Config):
        backbone = _backbone.factory(config)
        fused = _neck.factory(config, backbone)
        if fused is None:
            fused = backbone
        detector = _head.factory(config, fused.feature_specs)
        architecture = cls(config, fused, detector)
        return architecture

    def preprocess_input(self, inputs):
        return self.backbone.preprocess_input(inputs)

    def call(self, inputs, training=None, mask=None):
        # Keras 3 layers branch on `training` in Python, so it must never be a tensor here.
        training = False if training is None else training
        features = self.backbone(inputs, training=training)
        if self.single_backbone_mode:
            features = [features[0], features[0]]
        output = self.head(features, training=training)
        return output
