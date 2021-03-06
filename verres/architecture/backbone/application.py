from typing import List, Tuple

import tensorflow as tf

from . import FeatureSpec, VRSBackbone
from verres.utils import keras_utils


available_specs = {
    "resnet50v2": [FeatureSpec("input_1", working_stride=1),
                   FeatureSpec("conv1_conv", working_stride=2),
                   FeatureSpec("conv2_block3_1_relu", working_stride=4),
                   FeatureSpec("conv3_block4_1_relu", working_stride=8),
                   FeatureSpec("conv4_block6_1_relu", working_stride=16),
                   FeatureSpec("post_relu", working_stride=32)]
}


class ApplicationBackbone(VRSBackbone):

    def __init__(self,
                 name: str,
                 feature_specs: List[FeatureSpec],
                 input_shape=None,
                 fixed_batch_size=None,
                 weights=None):

        super().__init__(feature_specs)

        base_model = keras_utils.ApplicationCatalogue().make_model(
            name,
            include_top=False,
            input_shape=input_shape,
            fixed_batch_size=fixed_batch_size,
            build_model=False,
            weights=weights)

        outputs = [base_model.get_layer(spec.layer_name).output for spec in feature_specs]
        self.wrapped_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        for spec, shape in zip(self.feature_specs, self.get_output_shapes()):
            spec.width = shape[-1]

    @classmethod
    def from_feature_strides(cls,
                             name: str,
                             feature_strides: List[int],
                             input_shape: Tuple[int, int, int] = None,
                             fixed_batch_size: int = None,
                             weights: str = None):

        specs_for_model = {spec.working_stride: spec for spec in available_specs[name.lower()]}
        specs = [specs_for_model[stride] for stride in feature_strides]
        return cls(name, specs, input_shape, fixed_batch_size, weights)

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
