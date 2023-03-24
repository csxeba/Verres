from typing import Dict

import tensorflow as tf

from ..operation import numeric as ops
from ..data import Sample


def fake_od_network_outputs(sample: Sample) -> Dict[str, tf.Tensor]:
    encoded = sample.encoded_tensors
    label = sample.label
    tensor_shape = tuple(s // 4 for s in sample.input.shape_whc[:2][::-1])
    refinements = ops.scatter(label.object_centers, vectors=encoded["refinement"], tensor_shape_hw=tensor_shape)
    box_whs = ops.scatter(label.object_centers, vectors=encoded["box_wh"], tensor_shape_hw=tensor_shape)
    return {
        "heatmap": encoded["heatmap"],
        "refinement": refinements,
        "box_wh": box_whs
    }
