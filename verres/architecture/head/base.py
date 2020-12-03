import tensorflow as tf

from ..backbone.base import VRSBackbone


class VRSHead(tf.keras.Model):

    def __init__(self, backbone: VRSBackbone):
        super().__init__()
        self.backbone = backbone

    def postprocess_network_output(self, predictions):
        raise NotImplementedError

    def call(self, inputs, training=False, mask=None):
        raise NotImplementedError

    def detect(self, inputs):
        preprocessed_inputs = self.backbone.preprocess_input(inputs)
        prediction = self(preprocessed_inputs, training=False)
        detection = self.postprocess_network_output(prediction)
        return detection
