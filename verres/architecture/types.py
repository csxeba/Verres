from typing import NamedTuple, Dict

import tensorflow as tf


class IntermediateResult(NamedTuple):

    outputs: Dict[str, tf.Tensor]
    metrics: Dict[str, tf.Tensor]
    losses: Dict[str, tf.Tensor]

