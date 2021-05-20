import tensorflow as tf


class GlobalSTDPooling2D(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        self.data_format = tf.keras.backend.image_data_format()

    def call(self, inputs, **kwargs):
        if self.data_format == 'channels_last':
            return tf.keras.backend.std(inputs, axis=[1, 2])
        else:
            return tf.keras.backend.std(inputs, axis=[2, 3])
