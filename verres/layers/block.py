import tensorflow as tf
import tensorflow.keras.layers as tfl

from verres.utils import layer_utils


class VRSConvolution(tf.keras.Model):

    """The Verres Convolution Layer"""

    def __init__(self,
                 width: int,
                 activation: str = None,
                 batch_normalize: bool = True):

        super().__init__()

        self.layer_objects = [tfl.Conv2D(width, kernel_size=3, padding="same")]
        if batch_normalize:
            self.layer_objects.append(tfl.BatchNormalization())
        self.layer_objects.append(layer_utils.get_activation(activation, as_layer=True))

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        for layer in self.layer_objects:
            x = layer(x)
        return x


class VRSConvBlock(tf.keras.Model):

    def __init__(self, width, depth, skip_connect=True, batch_normalize=True, activation="leakyrelu"):
        super().__init__()
        self.layer_objects = [VRSConvolution(width, activation, batch_normalize) for _ in range(depth)]
        self.skip_connect = skip_connect

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layer_objects:
            x = layer(x, training=training, mask=mask)
        if self.skip_connect:
            x = tf.concat([inputs, x], axis=3)
        return x


class VRSUpscale(tf.keras.Model):

    def __init__(self, num_stages, width_base, batch_normalize=True, activation="leakyrelu"):
        super().__init__()
        self.layer_objects = []
        if width_base % (2 ** num_stages) != 0:
            raise RuntimeError("width_base for VRSUpscale is not divisible by 2^num_stages!")
        width = width_base
        for _ in range(num_stages):
            self.layer_objects.append(VRSConvolution(width_base, activation, batch_normalize))
            self.layer_objects.append(tfl.UpSampling2D(interpolation="bilinear"))
            width //= 2

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        for layer in self.layers:
            x = layer(x)
        return x
