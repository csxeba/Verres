
import tensorflow as tf


class VRSHead(tf.keras.layers.Layer):

    def postprocess_network_output(self, predictions):
        raise NotImplementedError

    def call(self, inputs, training=False, mask=None):
        raise NotImplementedError
