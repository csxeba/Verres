from typing import List

import tensorflow as tf

import verres as V
from ..backbone import FeatureSpec


class VRSHead(tf.keras.layers.Layer):

    def __init__(self, config: V.Config, input_features: List[FeatureSpec]):
        super().__init__()
        self.config = config
        self.input_features = input_features

    def postprocess_network_output(self, predictions):
        raise NotImplementedError

    def call(self, inputs, training=False, mask=None):
        raise NotImplementedError
