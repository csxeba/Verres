from typing import List

import numpy as np
import tensorflow as tf

import verres as V
from .. import feature


def _concatenate_batch_index(regression_masks: List[np.ndarray]):
    for batch_index, tensor in enumerate(regression_masks):
        n = len(tensor)
        batch_indices = np.full([n, 1], batch_index, dtype=tensor.dtype)
        regression_masks[batch_index] = np.concatenate([batch_indices, tensor], axis=1)


class CollateBatch:

    def __init__(self, config: V.Config, features: List[feature.Feature]):
        self.cfg = config
        self.features = features

    def process(self, meta_list: List[dict]):
        result = {}
        for ftr in self.features:
            tensor_list = [meta[ftr.meta_field] for meta in meta_list]
            if ftr.sparse:
                if ftr.name == "regression_mask":
                    _concatenate_batch_index(tensor_list)
                collated = np.concatenate(tensor_list, axis=0)
            else:
                collated = np.stack(tensor_list, axis=0)
                if ftr.stride > 1:
                    collated = V.operation.padding.pad_output_tensor_to_stride(
                        collated,
                        self.cfg.model.maximum_stride,
                        tensor_stride=ftr.stride)
            result[ftr.name] = tf.cast(collated, ftr.dtype)
        return result
