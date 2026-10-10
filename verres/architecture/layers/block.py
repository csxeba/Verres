import math
from typing import List

import tensorflow as tf
import tensorflow.keras.layers as tfl

from verres.feature import Feature
from verres.utils import layer_utils


class VRSLayerStack(tf.keras.layers.Layer):

    def __init__(self, layers=(), **kwargs):
        super().__init__(**kwargs)
        self.layer_objects: List[tf.keras.layers.Layer] = list(layers)

    def call(self, x, **kwargs):
        for layer in self.layer_objects:
            x = layer(x)
        return x


class VRSConvolution(VRSLayerStack):

    """The Verres Convolution Layer"""

    def __init__(self,
                 width: int,
                 activation: str = None,
                 batch_normalize: bool = True,
                 kernel_size: int = 3,
                 stride: int = 1,
                 initializer: str = "he_uniform"):

        super().__init__()

        self.layer_objects = [tfl.Conv2D(width, kernel_size, padding="same", kernel_initializer=initializer,
                                         strides=stride)]
        if batch_normalize:
            self.layer_objects.append(tfl.BatchNormalization())
        if activation is not None:
            self.layer_objects.append(layer_utils.get_activation(activation))

        self.output_width = width

    # @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        for layer in self.layer_objects:
            x = layer(x, training=training)
        return x


class VRSHead(VRSLayerStack):

    def __init__(self,
                 pre_width: int,
                 output_feature: Feature,
                 pre_activation: str = "leakyrelu",
                 output_activation: str = "linear",
                 output_initializer: str = "default",
                 batch_normalize: bool = True,
                 **kwargs):

        super().__init__(**kwargs)
        if output_initializer == "default":
            output_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
        if pre_width > 0:
            self.layer_objects = [VRSConvolution(pre_width, pre_activation, batch_normalize, kernel_size=3)]
        else:
            self.layer_objects = []
        self.layer_objects.append(
            VRSConvolution(
                width=output_feature.depth,
                activation=output_activation,
                batch_normalize=False,
                kernel_size=1,
                initializer=output_initializer
            )
        )
        self.output_feature = output_feature


class VRSRescaler(VRSLayerStack):

    MODE_UP = "up"
    MODE_DOWN = "down"

    def __init__(self,
                 stride: int,
                 base_width: int,
                 mode: str = None,
                 kernel_size: int = 3,
                 batch_normalize: bool = True,
                 activation: str = "leakyrelu",
                 **kwargs):

        super().__init__(**kwargs)

        resampler_type = {self.MODE_UP: tf.keras.layers.UpSampling2D,
                          self.MODE_DOWN: tf.keras.layers.MaxPool2D}[mode]

        steps = int(math.log2(stride))
        for i in range(steps):
            if mode == self.MODE_UP:
                width = base_width // (2 ** i)
            else:
                assert base_width % (2 ** i) == 0
                width = base_width * (2 ** i)
            self.layer_objects.append(resampler_type())
            self.layer_objects.append(VRSConvolution(width, activation, batch_normalize, kernel_size))
            self.output_width = width

    @classmethod
    def from_strides(cls,
                     feature_stride: int,
                     target_stride: int,
                     base_width: int,
                     kernel_size: int = 3,
                     batch_normalize: bool = True,
                     activation: str = "leakyrelu",
                     **kwargs):

        if feature_stride > target_stride:
            assert (feature_stride % target_stride) == 0
            stride = feature_stride // target_stride
            return cls(stride, base_width, cls.MODE_UP, kernel_size, batch_normalize, activation, **kwargs)
        elif feature_stride < target_stride:
            assert (target_stride % feature_stride) == 0
            stride = target_stride // feature_stride
            return cls(stride, base_width, cls.MODE_DOWN, kernel_size, batch_normalize, activation, **kwargs)
        else:
            return VRSConvolution(base_width, activation, batch_normalize, **kwargs)
