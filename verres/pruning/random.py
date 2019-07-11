import numpy as np
import tensorflow as tf
from tfkerassurgeon import Surgeon

from .base import Pruner

K = tf.keras.backend


class RandomPruner(Pruner):

    def _produce_filter_pruning_mask(self, keep_ratio):
        num_filters = sum(K.int_shape(W)[-1] for W in self.weights_of_interest)
        drop_mask = np.random.random(num_filters) > keep_ratio
        result = []
        start = 0
        for weight in self.weights_of_interest:
            m = K.int_shape(weight)[-1]
            result.append(drop_mask[start:start + m])
            start += m
        return result

    def prune(self, keep_ratio):
        surgeon = Surgeon(self.model, copy=True)
        filter_masks = self._produce_filter_pruning_mask(keep_ratio)
        for filter_mask, layer in zip(filter_masks, self.model.layers):
            drop_indices = np.argwhere(filter_mask)[:, 0]
            if drop_indices.size == filter_mask.size:
                drop_indices = drop_indices[1:]
            if drop_indices.size:
                surgeon.add_job("delete_channels", layer, channels=drop_indices)
        return surgeon.operate()
