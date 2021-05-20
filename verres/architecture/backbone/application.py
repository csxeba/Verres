from typing import List

import tensorflow as tf

import verres as V
from . import FeatureSpec, VRSBackbone
from verres.utils import keras_utils


class ApplicationBackbone(VRSBackbone):

    def __init__(self, config: V.Config):

        base_model = keras_utils.ApplicationCatalogue().make_model(
            config.model.backbone_spec["name"],
            input_shape=config.model.input_shape,
            weights=config.model.backbone_spec.get("weights", None))

        feature_layers = config.model.backbone_spec.get("feature_layers", "default")
        if feature_layers == "default":
            feature_specs = [FeatureSpec.from_last_tensor(base_model)]
        else:
            feature_specs = FeatureSpec.from_layer_names(base_model, feature_layers)

        super().__init__(feature_specs)

        self.cfg = config

        self._wrapped_model = self._set_feature_layes(base_model, feature_specs)
        self._preprocess_input_base = base_model.preprocess_input

        # Update feature spec width's
        for spec, shape in zip(self.feature_specs, self.get_output_shapes()):
            spec.width = shape[-1]

    @staticmethod
    def _set_feature_layes(base_model: tf.keras.Model, feature_specs: List[FeatureSpec]):
        outputs = [base_model.get_layer(spec.layer_name).output for spec in feature_specs]
        return tf.keras.Model(inputs=base_model.input, outputs=outputs)

    def get_output_shapes(self):
        shapes = self._wrapped_model.compute_output_shape((None, None, None, 3))
        if len(self.feature_specs) == 1:
            shapes = [shapes]
        return shapes

    def preprocess_input(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        inputs = self._preprocess_input_base(inputs)
        inputs = V.operation.padding.pad_to_stride(inputs, model_stride=self.cfg.model.maximum_stride)
        return inputs

    @tf.function
    def call(self, x, training=None, mask=None):
        result = self._wrapped_model(x, training, mask)
        if len(self.feature_specs) == 1:
            result = [result]
        return result
