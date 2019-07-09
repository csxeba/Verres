import tensorflow as tf


class GradientBarrier(tf.keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return tf.keras.backend.stop_gradient(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
