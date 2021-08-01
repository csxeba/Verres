from enum import IntEnum
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers as tfl


class BackboneMode(IntEnum):

    CONV = 1
    DENSE = 2


def get_input(image_shape: Tuple[int, int, int], fixed_batch_size: int = None, name="input"):

    tensor = tf.keras.Input(batch_size=(fixed_batch_size,) + image_shape, name=name)
    return tensor


def _make_backbone(inputs: tf.keras.Input,
                   units: Tuple[int],
                   batch_normalize: bool,
                   activation: str,
                   backbone_mode: BackboneMode):

    x = inputs
    if backbone_mode == BackboneMode.DENSE:
        layer_factory = lambda c: tfl.Dense(units=c)
    elif backbone_mode == BackboneMode.CONV:
        layer_factory = lambda c: tfl.Conv2D(filters=c, kernel_size=3, padding="same")
    else:
        assert False

    for unit in units:

        x = layer_factory(unit)(x)

        if batch_normalize:
            x = tfl.BatchNormalization()(x)

        if "leaky" in activation.lower():
            activation_layer = tfl.LeakyReLU()
        else:
            activation_layer = tfl.Activation(activation)
        x = activation_layer(x)

    return x


def classifier_head(inputs: tf.Tensor,
                    num_classes: int,
                    output_activation: str = None):

    if output_activation is None:
        output_activation = "linear"

    outputs = tfl.Dense(num_classes, activation=output_activation)(inputs)
    return outputs


def build(input_shape: Tuple[int, int, int],
          num_classes: int,
          backbone_num_units: Tuple[int],
          batch_normalize: bool,
          activation: str = "leakyrelu",
          output_activation: str = "softmax",
          backbone_mode: BackboneMode = BackboneMode.DENSE,
          name="model"):

    inputs = get_input(input_shape, name=name + "_inputs")
    backbone_features = _make_backbone(inputs, backbone_num_units, batch_normalize, activation, backbone_mode)
    outputs = classifier_head(backbone_features, num_classes, output_activation)

    model = tf.keras.Model(inputs, outputs, name=name)

    return model
