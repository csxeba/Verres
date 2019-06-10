from collections import defaultdict

import numpy as np

from keras.models import Model
from keras import backend as K
from kerassurgeon import Surgeon

from .base import Pruner


class SNIP(Pruner):

    def __init__(self, model: Model, loss, excluded_layer_names=()):
        super().__init__(model, excluded_layer_names)
        self.loss = loss
        self._get_saliencies = None
        self._build_saliency_function()

    def _build_saliency_function(self):
        ground_truth = K.placeholder(self.model.output_shape)
        prediction = self.model.output
        loss_tensor = self.loss(ground_truth, prediction)

        saliencies = []
        for weight in self.weights_of_interest:
            [grad] = self.model.optimizer.get_gradients(loss_tensor, [weight])
            saliencies.append(grad * weight)

        self._get_saliencies = K.function(inputs=[self.model.input, ground_truth], outputs=saliencies)

    def _reorder_saliencies(self, saliencies: list):
        result = defaultdict(list)
        for saliency, weight in zip(saliencies, self.weights_of_interest):
            layer_name = self.weight_to_layer[weight.name]
            result[layer_name].append(saliency)
        return result

    def _produce_filter_pruning_mask(self, saliencies, keep_ratio):
        filter_saliencies = []
        for saliency in saliencies:
            mean_axes = tuple(range(saliency.ndim - 1))
            s = np.mean(saliency, axis=mean_axes)
            filter_saliencies.append(s)
        cat_filter_saliencies = np.concatenate(filter_saliencies, axis=0)
        sorted_args = cat_filter_saliencies.argsort(kind="mergesort")
        drop_ratio = 1 - keep_ratio
        drop_num = int(len(cat_filter_saliencies) * drop_ratio)
        drop_mask = np.zeros_like(cat_filter_saliencies, dtype=bool)
        drop_mask[sorted_args[:drop_num]] = True
        assert np.all(cat_filter_saliencies[sorted_args[:drop_num]] <= cat_filter_saliencies[sorted_args[drop_num]])
        result = []
        start = 0
        for saliency in saliencies:
            num_filters = saliency.shape[-1]
            result.append(drop_mask[start:start+num_filters])
            start += num_filters
        return result, filter_saliencies

    def prune(self, x, y, keep_ratio):
        saliencies = self._get_saliencies([x, y])
        filter_masks, filter_saliencies = self._produce_filter_pruning_mask(saliencies, keep_ratio)
        surgeon = Surgeon(self.model, copy=True)
        for filter_mask, saliency, weight in zip(filter_masks, filter_saliencies, self.weights_of_interest):
            layer_name = self.weight_to_layer[weight.name]
            layer = self.model.get_layer(name=layer_name)
            drop_indices = np.squeeze(np.argwhere(filter_mask))
            if drop_indices.size == filter_mask.size:
                best_filter = saliency.argmax()
                drop_indices = np.delete(drop_indices, best_filter)
            if drop_indices.shape:
                surgeon.add_job("delete_channels", layer, channels=drop_indices)
        return surgeon.operate()
