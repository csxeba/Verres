from typing import List

import tensorflow as tf


class FeatureSpec:

    def __init__(self, layer_name: str, working_stride: int = None):
        self.layer_name = layer_name
        self.working_stride = working_stride
        self.width = -1


class VRSBackbone(tf.keras.Model):

    def __init__(self, feature_specs: List[FeatureSpec]):
        super().__init__()
        self.feature_specs: List[FeatureSpec] = feature_specs or None
