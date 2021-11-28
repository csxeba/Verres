from typing import Dict

import tensorflow as tf

import verres as V
from ..operation import losses


class VRSCriteria(tf.Module):

    def __init__(self, config: V.Config, spec: dict):
        super().__init__()
        self.cfg = config
        self.spec = spec
        self.nan_guard = spec.get("nan_guard", False)
        self.loss_functions: Dict[str, losses.LossFunction] = {
            losses_spec["feature"]: losses.factory(losses_spec) for losses_spec in spec["losses"]}

    def call(self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:

        loss = 0.
        all_losses = {}

        for feature_name, loss_fn in self.loss_functions.items():
            if loss_fn.is_sparse_loss:
                current_loss = loss_fn.call(y_true=y_true[feature_name],
                                            y_pred=y_pred[feature_name],
                                            locations=y_true[loss_fn.sparse_location_feature_name])
            else:
                current_loss = loss_fn.call(y_true=y_true[feature_name],
                                            y_pred=y_pred[feature_name])
            if self.nan_guard:
                tf.assert_equal(tf.logical_not(tf.reduce_any(tf.math.is_nan(current_loss))),
                                f"NaN Guard for {feature_name}/{loss_fn.name}")

            loss += current_loss * loss_fn.loss_weight
            all_losses[feature_name] = current_loss

        total_loss = loss / len(self.loss_functions)
        all_losses["loss"] = total_loss
        return all_losses

    def get_output_keys(self):
        return list(self.loss_functions)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
