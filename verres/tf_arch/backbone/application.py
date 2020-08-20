from typing import List

import tensorflow as tf

from . import FeatureSpec
from verres.utils import keras_utils


class ApplicationBackbone(tf.keras.Model):

    def __init__(self,
                 name: str,
                 feature_specs: List[FeatureSpec],
                 input_shape=None,
                 fixed_batch_size=None,
                 weights=None):

        super().__init__()
        base_model = keras_utils.ApplicationCatalogue().make_model(
            name,
            include_top=False,
            input_shape=input_shape,
            fixed_batch_size=fixed_batch_size,
            build_model=False,
            weights=weights)

        self.feature_specs: List[FeatureSpec] = feature_specs
        outputs = [base_model.get_layer(spec.layer_name).output for spec in feature_specs]
        self.wrapped_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        for spec, shape in zip(self.feature_specs, self.get_output_shapes()):
            spec.width = shape[-1]

    def get_output_shapes(self):
        shapes = self.wrapped_model.compute_output_shape((None, None, None, 3))
        if len(self.feature_specs) == 1:
            shapes = [shapes]
        return shapes

    def build_model(self):
        shape = tf.keras.backend.int_shape(self.wrapped_model.input)[1:]
        max_stride = max(spec.working_stride for spec in self.feature_specs)
        shape = [max_stride if dim is None else dim for dim in shape]
        self.wrapped_model(tf.zeros([1] + shape))
        for tensor, spec in zip(self.wrapped_model.outputs, self.feature_specs):
            spec.width = tf.keras.backend.int_shape(tensor)[-1]

    @tf.function
    def call(self, x, training=None, mask=None):
        result = self.wrapped_model(x, training, mask)
        if len(self.feature_specs) == 1:
            result = [result]
        return result


