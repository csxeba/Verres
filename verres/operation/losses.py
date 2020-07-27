import tensorflow as tf


@tf.function
def sse(y_true, y_pred):
    d = tf.square(y_true - y_pred)
    d = tf.reduce_sum(d, axis=(1, 2, 3))
    return tf.reduce_mean(d)


@tf.function
def sae(y_true, y_pred):
    d = tf.abs(y_true - y_pred)
    d = tf.reduce_sum(d, axis=(1, 2, 3))
    return tf.reduce_mean(d)


@tf.function
def sum_of_cxent_sparse_from_logits(y_true, y_pred):
    xent = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True, axis=-1)
    assert len(xent.shape) == 3
    xent = tf.reduce_sum(xent, axis=(1, 2))
    return tf.reduce_mean(xent)
