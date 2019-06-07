import tensorflow as tf


def sse(y_true, y_pred):
    return tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred), axis=(1, 2, 3))
