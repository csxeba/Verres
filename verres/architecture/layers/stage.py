import tensorflow as tf

from verres.architecture.layers import block


class StageBody(tf.keras.Model):

    def __init__(self, width, input_width, num_blocks=5, skip_connect=True):
        super().__init__()
        self.layer_objects = [block.VRSResidualBottleneck(width, input_width, stride=1)
                              for _ in range(num_blocks)]
        self.skip_connect = skip_connect

    # @tf.function
    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layer_objects:
            x = layer(x, training=training, mask=mask)
        if self.skip_connect:
            x = tf.concat([inputs, x], axis=3)
        return x
