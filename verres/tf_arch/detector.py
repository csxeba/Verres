import tensorflow as tf
from tensorflow.keras import layers as tfl

from ..layers import block
from ..utils import layer_utils


class StageBody(tf.keras.Model):

    def __init__(self, width, num_blocks=5, skip_connect=True):
        super().__init__()
        self.layer_objects = [block.VRSConvBlock(width, depth=3, skip_connect=True, batch_normalize=True,
                                                 activation="leakyrelu") for _ in range(num_blocks)]
        self.skip_connect = skip_connect

    # @tf.function
    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layer_objects:
            x = layer(x, training=training, mask=mask)
        if self.skip_connect:
            x = tf.concat([inputs, x], axis=3)
        return x


class Head(tf.keras.Model):

    def __init__(self, width: int, num_outputs: int, activation: str = "leakyrelu"):
        super().__init__()
        self.conv = tfl.Conv2D(width, kernel_size=3, padding="same")
        self.act = layer_utils.get_activation(activation, as_layer=True)
        self.out = tfl.Conv2D(num_outputs, kernel_size=3, padding="same")

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.act(x)
        return self.out(x)


class Panoptic(tf.keras.Model):

    def __init__(self, num_classes: int):
        super().__init__()
        self.body8 = StageBody(width=64, num_blocks=5, skip_connect=True)
        self.body4 = StageBody(width=32, num_blocks=1, skip_connect=True)
        self.body2 = StageBody(width=16, num_blocks=1, skip_connect=True)
        self.upsc8_4 = block.VRSUpscale(num_stages=1, width_base=64, batch_normalize=True, activation="leakyrelu")
        self.upsc4_2 = block.VRSUpscale(num_stages=1, width_base=32, batch_normalize=True, activation="leakyrelu")
        self.upsc2_1 = block.VRSUpscale(num_stages=1, width_base=16, batch_normalize=True, activation="leakyrelu")
        self.hmap = Head(width=64, num_outputs=num_classes)
        self.rreg = Head(width=64, num_outputs=num_classes * 2)
        self.sseg = Head(width=8, num_outputs=num_classes + 1)
        self.iseg = Head(width=8, num_outputs=2)

    @tf.function
    def call(self, inputs, training=None, mask=None):

        ftr1, ftr2, ftr4, ftr8 = inputs

        ftr8 = self.body8(ftr8, training=training, mask=mask)

        ftr4 = tf.concat([self.upsc8_4(ftr8, training=training, mask=mask), ftr4], axis=-1)
        ftr4 = self.body4(ftr4, training=training, mask=mask)

        ftr2 = tf.concat([self.upsc4_2(ftr4, training=training, mask=mask), ftr2], axis=-1)
        ftr2 = self.body2(ftr2, training=training, mask=mask)

        ftr1 = tf.concat([self.upsc2_1(ftr2, training=training, mask=mask), ftr1], axis=-1)

        hmap = self.hmap(ftr8, training=training, mask=mask)
        rreg = self.rreg(ftr8, training=training, mask=mask)
        iseg = self.iseg(ftr1, training=training, mask=mask)
        sseg = self.sseg(ftr1, training=training, mask=mask)

        return hmap, rreg, iseg, sseg


class OD(tf.keras.Model):

    def __init__(self, num_classes: int, stride: int):
        super().__init__()
        self.body_centroid = StageBody(width=32, num_blocks=5, skip_connect=True)
        self.body_box = StageBody(width=32, num_blocks=5, skip_connect=True)
        self.hmap_head = Head(32, num_outputs=num_classes, activation="leakyrelu")
        self.rreg_head = Head(32, num_outputs=num_classes*2, activation="leakyrelu")
        self.boxx_head = Head(32, num_outputs=num_classes*2, activation="leakyrelu")
        self.stride = stride

    def call(self, inputs, training=None, mask=None):
        centroid_features, box_features = inputs
        x = self.body_centroid(centroid_features)
        hmap = self.hmap_head(x)
        rreg = self.rreg_head(x)
        x = tf.concat([box_features, hmap, rreg], axis=-1)
        x = self.body_box(x)
        boxx = self.boxx_head(x)
        return hmap, rreg, boxx
