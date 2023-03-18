from typing import Dict

import tensorflow as tf

import verres as V
from . import losses


class VRSCriteria:

    def __init__(self, config: V.Config, spec: dict):
        super().__init__()
        self.cfg = config
        self.spec = spec
        self.nan_guard = spec.get("nan_guard", False)
        feature_names = [loss_spec["feature"] for loss_spec in spec["losses"]]
        if len(set(feature_names)) != len(feature_names):
            raise RuntimeError(f"Multiple losses are defined for the same feature: {feature_names}")
        self.loss_functions: Dict[str, losses.LossFunction] = {}
        for loss_spec in spec["losses"]:
            feature_name = loss_spec["feature"]
            self.loss_functions[feature_name] = losses.factory(loss_spec)

    def call(
        self,
        y_true: Dict[str, tf.Tensor],
        y_pred: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        loss = 0.
        all_losses = {}

        object_locations = tf.where(y_true["heatmap"] == 1)[:, :3]
        N = tf.cast(tf.shape(object_locations)[0], tf.float32)

        for feature_name, loss_fn in self.loss_functions.items():
            if loss_fn.is_sparse_loss:
                current_loss = loss_fn.call(y_true=y_true[feature_name],
                                            y_pred=y_pred[feature_name],
                                            locations=object_locations)
            else:
                current_loss = loss_fn.call(y_true=y_true[feature_name],
                                            y_pred=y_pred[feature_name])
            if self.nan_guard:
                tf.assert_equal(tf.reduce_any(tf.math.is_nan(current_loss)), False,
                                f"NaN Guard for {feature_name}/{loss_fn.name}")

            loss += current_loss * loss_fn.loss_weight / N
            all_losses[feature_name] = current_loss

        total_loss = tf.reduce_mean(loss)
        all_losses["loss"] = total_loss
        return all_losses

    def get_output_keys(self):
        return list(self.loss_functions)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
