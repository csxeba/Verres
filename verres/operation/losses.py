from typing import List

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


@tf.function(experimental_relax_shapes=True)
def sparse_vector_field_sae(y_true, y_pred, locations):
    pred_x = tf.gather_nd(y_pred[..., 0::2], locations)
    pred_y = tf.gather_nd(y_pred[..., 1::2], locations)
    d = tf.abs(y_true - tf.stack([pred_x, pred_y], axis=-1))
    d = tf.reduce_sum(d, axis=-1)
    return tf.reduce_mean(d)


@tf.function
def sum_of_cxent_sparse_from_logits(y_true, y_pred):
    xent = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True, axis=-1)
    assert len(xent.shape) == 3
    xent = tf.reduce_sum(xent, axis=(1, 2))
    return tf.reduce_mean(xent)


@tf.function
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.function
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


@tf.function
def mean_of_cxent_sparse_from_logits(y_true, y_pred):
    xent = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True, axis=-1)
    return tf.reduce_mean(xent)


class Tracker:

    def __init__(self, keys: List[str]):
        self.keys = keys
        self.variables = [tf.Variable(0., dtype=tf.float32, trainable=False) for _ in keys]
        self.step = tf.Variable(0., dtype=tf.float32, trainable=False)

    def record(self, data):
        for v, d in zip(self.variables, data):
            v.assign_add(d)
        self.step.assign_add(1.)
        return {k: v / self.step for k, v in zip(self.keys, self.variables)}

    def reset(self):
        for v in self.variables:
            v.assign(0.)
        self.step.assign(0.)
