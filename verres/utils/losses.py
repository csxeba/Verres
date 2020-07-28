import tensorflow as tf


def sse(y_true, y_pred):
    return tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred), axis=(1, 2, 3))


def sae(y_true, y_pred):
    return tf.keras.backend.sum(tf.keras.backend.abs(y_true - y_pred), axis=(1, 2, 3))


def mse(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))


def mae(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_pred))
