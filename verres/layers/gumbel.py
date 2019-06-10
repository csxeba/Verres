import tensorflow as tf


class GumbelSoftmax(tf.keras.layers.Layer):

    def __init__(self, temperature=1., **kwargs):
        super().__init__(**kwargs)
        self.temperature = tf.Variable(temperature)

    def call(self, inputs, **kwargs):
        distro = tf.contrib.distributions.RelaxedOneHotCategorical(self.temperature, inputs)
        output = distro.sample(tf.shape(inputs)[0])
        return output
