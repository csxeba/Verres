import math
from typing import List

import tensorflow as tf
from tensorflow.keras import layers as tfl

from verres.utils import keras_utils
from verres.layers import block


class FeatureSpec:

    def __init__(self, layer_name: str, working_stride: int = None):
        self.layer_name = layer_name
        self.working_stride = working_stride
        self.width = -1


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


class SideModel(tf.keras.Model):

    def __init__(self,
                 working_stride: int,
                 final_width: int,
                 base_width: int = 8,
                 base_depth: int = 1,
                 batch_norm: bool = True):

        super().__init__()

        self.working_stride = working_stride
        self.base_width = base_width
        self.batch_norm = batch_norm

        self.layer_objects = []
        current_stride = 1
        current_width = base_width
        current_depth = base_depth
        for i in range(10):
            self.layer_objects.append(
                block.VRSConvBlock(current_width, current_depth, batch_normalize=batch_norm)
            )
            self.layer_objects.append(tfl.MaxPool2D())
            current_width *= 2
            current_depth *= 2
            current_stride *= 2
            if current_stride >= working_stride:
                if current_stride > working_stride:
                    raise RuntimeError(f"Misspecified working stride: {working_stride}")
                break
        else:
            raise RuntimeError(f"Misspecified or too great working stride: {working_stride}")
        self.layer_objects.append(
            block.VRSConvolution(final_width, batch_normalize=batch_norm, initializer="zeros"))

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        for layer in self.layer_objects:
            x = layer(x)
        return x


class SideTunedBackbone(ApplicationBackbone):

    def __init__(self,
                 backbone_name: str,
                 feature_specs: List[FeatureSpec],
                 side_models: List[tf.keras.Model] = "default",
                 input_shape=None,
                 fixed_batch_size=None,
                 weights="imagenet"):

        super().__init__(backbone_name, feature_specs, input_shape, fixed_batch_size, weights)
        self.wrapped_model.trainable = False
        self.side_models = side_models
        self.blending_factors = tf.Variable([0.5]*len(self.side_models), trainable=True)
        if self.side_models == "default":
            self._generate_default_side_models()

    def _generate_default_side_models(self):
        side_models = []
        base_width = 8
        for spec in self.feature_specs:
            stride_level = int(math.log2(spec.working_stride))
            side_models.append(
                SideModel(spec.working_stride,
                          final_width=spec.width,
                          base_width=base_width * stride_level,
                          base_depth=stride_level,
                          batch_norm=True))
        self.side_models = side_models

    # @tf.function
    def call(self, x, training=None, mask=None):
        features = self.wrapped_model(x)
        if len(self.feature_specs) == 1:
            features = [features]

        sided_features = []
        for i in range(len(self.feature_specs)):
            side_feature = self.side_models[i](x)
            sided_features.append(
                features[i] * self.blending_factors[i] + side_feature * (tf.ones(1) - self.blending_factors[i])
            )
        return sided_features
