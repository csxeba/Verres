from typing import List, Tuple

import tensorflow as tf

from verres.utils import keras_utils


class FeatureSpec:

    def __init__(self, layer_name: str, working_strides: Tuple[int, int] = None):
        self.layer_name = layer_name
        self.working_strides = working_strides


class ApplicationBackbone(tf.keras.Model):

    def __init__(self,
                 name: str,
                 feature_specs: List[FeatureSpec],
                 input_shape=None,
                 fixed_batch_size=None):

        super().__init__()
        self.base_model = keras_utils.ApplicationCatalogue().make_model(
            name,
            include_top=False,
            input_shape=input_shape,
            fixed_batch_size=fixed_batch_size,
            build_model=False)
        self.feature_specs = feature_specs

        outputs = [self.base_model.get_layer(spec.layer_name).output for spec in feature_specs]

        self.wrapped_model = tf.keras.Model(inputs=self.base_model.input, outputs=outputs)

    @tf.function
    def call(self, x, training=None, mask=None):
        return self.wrapped_model(x)
