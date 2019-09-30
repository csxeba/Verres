import tensorflow as tf


class Stacker(tf.keras.layers.Layer):

    def __init__(self, axis=-1):
        super().__init__()
        self.stacking_axis = axis

    def call(self, inputs, **kwargs):
        return tf.stack(inputs, axis=self.stacking_axis)
