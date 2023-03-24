import dataclasses
from typing import List, NamedTuple, Optional

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy


def sse(y_true, y_pred):
    d = tf.square(y_true - y_pred)
    d = tf.reduce_sum(d)
    return d


def sae(y_true, y_pred):
    d = tf.abs(y_true - y_pred)
    d = tf.reduce_sum(d)
    return d


def sparse_sae(y_true, y_pred, locations):
    return sae(y_true, tf.gather_nd(y_pred, locations))


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def sparse_mae(y_true, y_pred, locations):
    return mae(y_true, tf.gather_nd(y_pred, locations))


def focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 2., beta: float = 4.):
    pos_inds = tf.cast(y_true == 1., tf.float32)
    neg_inds = tf.cast(y_true < 1., tf.float32)

    neg_weights = tf.pow(1. - y_true, beta)

    y_prob = tf.nn.sigmoid(y_pred)
    y_prob = tf.clip_by_value(y_prob, 0.01, 0.99)

    pos_loss = tf.math.log(y_prob) * tf.pow(1. - y_prob, alpha) * pos_inds
    neg_loss = tf.math.log(1. - y_prob) * tf.pow(y_prob, alpha) * neg_weights * neg_inds

    pos_loss_red = tf.reduce_sum(pos_loss)
    neg_loss_red = tf.reduce_sum(neg_loss)

    loss = pos_loss_red + neg_loss_red

    return -loss


@dataclasses.dataclass
class LossFunction:

    name: str
    feature_name: str
    loss_fn: callable
    loss_weight: tf.Variable
    is_sparse_loss: bool

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
                "focal": focal_loss,
                "binary_crossentropy": binary_crossentropy,
                "categorical_crossentropy": categorical_crossentropy}

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
    is_sparse_loss = params.pop("sparse", False)
    if is_sparse_loss:
        loss_fn = sparse_losses.get(name, None)
    else:
        loss_fn = dense_losses.get(name, None)
    if loss_fn is None:
        raise RuntimeError(f"No such {('dense', 'sparse')[int(is_sparse_loss)]} loss function: {name}")

    loss_weight = tf.Variable([weight], trainable=is_trainable_weight)

    loss_object = LossFunction(name, feature, loss_fn, loss_weight, is_sparse_loss)
    print(f" [Verres.losses] - Factory built: {name}({str(params)})")
    return loss_object
