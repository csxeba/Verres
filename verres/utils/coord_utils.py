from typing import Union

import numpy as np
import tensorflow as tf


def transpose_coordinate_system(coords: Union[tf.Tensor, np.ndarray]):
    ndim = coords.shape[1]
    if ndim == 2:
        result = coords[:, ::-1]
    elif ndim == 4:
        concat_fn = np.concatenate if isinstance(coords, np.ndarray) else tf.concat
        result = concat_fn([coords[:, 0:2][:, ::-1], coords[:, 2:4][:, ::-1]], axis=1)
    else:
        raise NotImplementedError

    return result
