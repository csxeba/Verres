from typing import List, Dict

import numpy as np
import tensorflow as tf

import verres as V
from verres import operation as ops
from ..sample import Sample
from ... import feature


def _concatenate_batch_index(regression_masks: List[np.ndarray]):
    for batch_index, tensor in enumerate(regression_masks):
        n = len(tensor)
        batch_indices = np.full([n, 1], batch_index, dtype=tensor.dtype)
        regression_masks[batch_index] = np.concatenate([batch_indices, tensor], axis=1)


class CollateBatch:

    def __init__(self, config: V.Config, features: List[feature.Feature]):
        self.cfg = config
        self.features = features

    def process(self, sample_list: List[Sample]) -> Dict[str, np.ndarray]:
        result = {}
        for ftr in self.features:
            tensor_list = [sample.encoded_tensors[ftr.meta_field] for sample in sample_list]
            if ftr.sparse:
                collated = np.concatenate(tensor_list, axis=0)
                if "object_center_locations" not in result:
                    result["object_center_locations"] = ops.numeric.get_object_center_locations(
                        sample_list,
                        stride=ftr.stride,
                    )
            else:
                collated = np.stack(tensor_list, axis=0)
                if ftr.stride > 1:
                    collated = ops.padding.pad_output_tensor_to_stride(
                        collated,
                        self.cfg.model.maximum_stride,
                        tensor_stride=ftr.stride)
            result[ftr.name] = tf.cast(collated, ftr.dtype)
        return result
