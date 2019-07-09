import tensorflow as tf


class Correlation(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        mean_x, std_x = tf.nn.moments(inputs, axes=1, keep_dims=True)
        mx = tf.matmul(mean_x, tf.transpose(mean_x))
        vx = tf.matmul(inputs, tf.transpose(inputs)) / tf.cast(tf.shape(inputs)[0], tf.float32) + 1e-7
        cov_xx = vx - mx
        cor_xx = cov_xx / tf.matrix_diag_part(vx)
        return cor_xx
