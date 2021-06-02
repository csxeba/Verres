from typing import List

import tensorflow as tf


def sse(y_true, y_pred):
    d = tf.square(y_true - y_pred)
    d = tf.reduce_sum(d, axis=(1, 2, 3))
    return tf.reduce_mean(d)


def sae(y_true, y_pred):
    d = tf.abs(y_true - y_pred)
    d = tf.reduce_sum(d, axis=(1, 2, 3))
    return tf.reduce_mean(d)


def sparse_vector_field_sae(y_true, y_pred, locations):
    pred_x = tf.gather_nd(y_pred[..., 0::2], locations)
    pred_y = tf.gather_nd(y_pred[..., 1::2], locations)
    d = tf.abs(y_true - tf.stack([pred_x, pred_y], axis=-1))
    d = tf.reduce_sum(d, axis=-1)
    return tf.reduce_mean(d)


def sparse_vector_field_mae(y_true, y_pred, locations):
    pred_x = tf.gather_nd(y_pred[..., 0::2], locations)
    pred_y = tf.gather_nd(y_pred[..., 1::2], locations)
    d = tf.abs(y_true - tf.stack([pred_x, pred_y], axis=-1))
    d = tf.reduce_sum(d, axis=-1)
    N = tf.cast(tf.shape(locations)[0], tf.float32)
    N = tf.maximum(N, 1.)
    return d / N


def sum_of_cxent_sparse_from_logits(y_true, y_pred):
    xent = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True, axis=-1)
    assert len(xent.shape) == 3
    xent = tf.reduce_sum(xent, axis=(1, 2))
    return tf.reduce_mean(xent)


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def mean_of_cxent_sparse_from_logits(y_true, y_pred):
    xent = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True, axis=-1)
    return tf.reduce_mean(xent)


def focal_loss(y_true, y_pred, alpha, beta, from_logits=True):
    pos_mask = tf.stop_gradient(tf.cast(y_true == 1., tf.float32))
    neg_mask = tf.stop_gradient(tf.cast(y_true < 1., tf.float32))
    num_pos = tf.maximum(tf.reduce_sum(pos_mask), 1.)

    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    if from_logits:
        y_prob = tf.nn.sigmoid(y_pred)
    else:
        y_prob = y_pred

    pos_loss = bce * pos_mask * tf.pow(1. - y_prob, alpha)
    neg_loss = bce * neg_mask * tf.pow(y_prob, alpha) * tf.pow(1. - y_true, beta)

    loss = tf.reduce_sum(pos_loss + neg_loss) / num_pos

    return loss


class Tracker:

    def __init__(self, keys: List[str]):
        self.keys = keys
        self.variables = [tf.Variable(0., dtype=tf.float32, trainable=False, name=key + "_logger") for key in keys]
        self.step = tf.Variable(0., dtype=tf.float32, trainable=False, name="step_logger")

    def record(self, data):
        for v, d in zip(self.variables, data):
            v.assign_add(d)
        self.step.assign_add(1.)
        return {k: v / self.step for k, v in zip(self.keys, self.variables)}

    def reset(self):
        for v in self.variables:
            v.assign(0.)
        self.step.assign(0.)
