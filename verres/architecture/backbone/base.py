import dataclasses
from typing import List, Union, Optional

import tensorflow as tf


@dataclasses.dataclass(repr=True)
class FeatureSpec:
    layer_name: str
    working_stride: Optional[int] = None
    width: int = -1

    @classmethod
    def from_last_tensor(cls, model: Union[tf.keras.Model, tf.keras.layers.Layer]):
        input_tensor = tf.zeros([1, 960, 960, 3], dtype=tf.float32)
        output_tensor = model(input_tensor)
        output_shape = tf.keras.backend.int_shape(output_tensor)
        working_stride = 960 // output_shape[1]

        result = cls(layer_name=model.layers[-1].name,
                     working_stride=working_stride,
                     width=output_shape[-1])

        print(f" [Verres] - automatically deduced feature layer: {result.layer_name}")

        return result

    @classmethod
    def from_layer_names(cls,
                         model: Union[tf.keras.Model, tf.keras.layers.Layer],
                         layer_names: List[str]):

        specs: List[FeatureSpec] = []

        input_tensor = tf.zeros((1, 960, 960, 3), dtype=tf.float32)

        for layer_name in layer_names:
            inputs = model.inputs
            output = model.get_layer(layer_name).output
            fake_model = tf.keras.Model(inputs, output)
            output_tensor = fake_model(input_tensor)
            output_shape = tf.keras.backend.int_shape(output_tensor)
            assert 960 % output_shape[1] == 0
            specs.append(
                cls(layer_name,
                    working_stride=960 // output_shape[1],
                    width=output_shape[-1]))

        return specs


class VRSBackbone(tf.keras.layers.Layer):

    def __init__(self, feature_specs: List[FeatureSpec]):
        super().__init__()
        self.feature_specs: List[FeatureSpec] = feature_specs or None
        self.single_feature_mode = len(feature_specs) == 1

    def preprocess_input(self, inputs):
        raise NotImplementedError
