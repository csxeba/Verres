from typing import List

import tensorflow as tf
from tensorflow.keras import layers as tfl

from verres.utils import keras_utils
from verres.layers import block


class FeatureSpec:

    def __init__(self, layer_name: str, working_stride: int = None):
        self.layer_name = layer_name
        self.working_stride = working_stride


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
        self.feature_specs = feature_specs

        outputs = [base_model.get_layer(spec.layer_name).output for spec in feature_specs]

        self.wrapped_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    @tf.function
    def call(self, x, training=None, mask=None):
        return self.wrapped_model(x, training, mask)


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
        self.layer_objects.append(block.VRSConvolution(final_width, batch_normalize=batch_norm))

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
                 fixed_batch_size=None):

        super().__init__(backbone_name, feature_specs, input_shape, fixed_batch_size)
        self.wrapped_model.trainable = False
        self.side_models = side_models
        self.blending_factors = tf.Variable([0.5]*len(self.side_models), trainable=True)
        if self.side_models == "default":
            self._generate_default_side_models()

    def _generate_default_side_models(self):
        ...

    @tf.function
    def call(self, x, training=None, mask=None):
        features = self.wrapped_model(x)
        sided_features = []
        for feature, side_model in zip(features, self.side_models):
            side_feature = side_model(x)
            sided_features.append(
                feature * self.blending_factor + side_feature * (tf.ones(1) - self.blending_factor)
            )
        return sided_features
