import functools
from typing import List, NamedTuple, Optional

import tensorflow as tf


def sse(y_true, y_pred):
    d = tf.square(y_true - y_pred)
    d = tf.reduce_sum(d, axis=(1, 2, 3))
    return tf.reduce_mean(d)


def sae(y_true, y_pred):
    d = tf.abs(y_true - y_pred)
    d = tf.reduce_sum(d, axis=(1, 2, 3))
    return tf.reduce_mean(d)


def sparse_sae(y_true, y_pred, locations):
    pred_x = tf.gather_nd(y_pred[..., 0::2], locations)
    pred_y = tf.gather_nd(y_pred[..., 1::2], locations)
    d = tf.abs(y_true - tf.stack([pred_x, pred_y], axis=-1))
    d = tf.reduce_sum(d, axis=-1)
    return tf.reduce_mean(d)


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def sparse_mae(y_true, y_pred, locations):
    pred_x = tf.gather_nd(y_pred[..., 0::2], locations)
    pred_y = tf.gather_nd(y_pred[..., 1::2], locations)
    d = tf.abs(y_true - tf.stack([pred_x, pred_y], axis=-1))
    return tf.reduce_mean(d)


def focal_loss(y_true, y_pred, alpha: float = 4., beta: float = 2.):
    pos_mask = y_true == 1
    neg_mask = tf.cast(tf.logical_not(pos_mask), tf.float32)
    pos_mask = tf.cast(pos_mask, tf.float32)
    num_peaks = tf.maximum(tf.reduce_sum(pos_mask), 1.)

    bxent = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    probabilities = tf.nn.sigmoid(y_pred)

    pos_loss = bxent * pos_mask * tf.pow(1. - probabilities, beta)
    neg_loss = bxent * neg_mask * tf.pow(probabilities, beta) * tf.pow(1. - y_true, alpha)

    loss = tf.reduce_sum(pos_loss, axis=(1, 2, 3)) + tf.reduce_sum(neg_loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss) / num_peaks

    return loss


class LossFunction(NamedTuple):

    name: str
    feature_name: str
    loss_fn: callable
    loss_weight: tf.Variable
    is_sparse_loss: bool
    sparse_location_feature_name: Optional[str]

    def call(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)


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


dense_losses = {"mse": mse,
                "mean_square_error": mse,
                "mean_squared_error": mse,
                "mae": mae,
                "mean_absolute_error": mae,
                "sse": sse,
                "sum_of_squared_errors": sse,
                "sae": sae,
                "sum_of_absolute_errors": sae,
                "focal_loss": focal_loss,
                "focal": focal_loss}

sparse_losses = {"sparse_sae": sparse_sae,
                 "sum_of_absolute_errors": sparse_sae,
                 "sum_of_absolute_error": sparse_sae,
                 "sae": sparse_sae,
                 "mean_absolute_error": sparse_mae,
                 "sparse_mean_absolute_error": sparse_mae,
                 "mae": sparse_mae}


def factory(params: dict):
    name = params.pop("name")
    feature = params.pop("feature", None)
    if feature is None:
        raise RuntimeError(f"`feature` must be specified in every loss spec. "
                           f"It is unspecified for {name}")

    weight = params.pop("weight", 1.)
    is_trainable_weight = params.pop("is_trainable_weight", False)
    is_sparse_loss = "sparse_location_feature" in params
    sparse_location_feature = params.pop("sparse_location_feature", None)
    if is_sparse_loss:
        loss_fn = sparse_losses.get(name, None)
    else:
        loss_fn = dense_losses.get(name, None)
    if loss_fn is None:
        raise RuntimeError(f"No such {('dense', 'sparse')[int(is_sparse_loss)]} loss function: {name}")

    loss_weight = tf.Variable([weight], trainable=is_trainable_weight)

    partial_loss_fn = functools.partial(loss_fn, **params)
    loss_object = LossFunction(name, feature, partial_loss_fn, loss_weight, is_sparse_loss, sparse_location_feature)
    print(f" [Verres.losses] - Factory built: {name}({str(params)})")
    return loss_object
