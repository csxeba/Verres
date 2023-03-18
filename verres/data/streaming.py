import random
from typing import List

import numpy as np
import tensorflow as tf

import verres as V
from .pipeline import Pipeline


def stream(config: V.Config,
           pipelines: List[Pipeline],
           shuffle: bool,
           batch_size: int,
           sampling_probabilities: List[float] = "uniform",
           collate: callable = "default"):

    if sampling_probabilities == "uniform":
        N = len(pipelines)
        sampling_probabilities = [1. / N for _ in range(N)]

    if len(pipelines) < len(sampling_probabilities):
        raise RuntimeError("There should be a sampling probability for every pipeline passed to stream()")

    if collate == "default":
        collate = V.data.transformation.CollateBatch(config, features=pipelines[0].output_features)

    sampling_probabilities = np.array(sampling_probabilities)
    sampling_probabilities = sampling_probabilities / np.sum(sampling_probabilities)

    meta_iterators = [
        iter(pipeline.stream(shuffle=shuffle, batch_size=1, collate_batch=None))
        for pipeline in pipelines]

    while 1:
        meta_list = []
        while len(meta_list) < batch_size:
            iterator = random.choices(meta_iterators, sampling_probabilities)[0]
            meta_list.extend(next(iterator))

        if collate is not None:
            batch = collate.process(meta_list)
            yield batch
        else:
            yield meta_list


def get_tf_dataset(config: V.Config,
                   pipelines: List[Pipeline],
                   shuffle: bool,
                   batch_size: int,
                   sampling_probabilities: List[float] = "uniform",
                   collate: callable = "default"):

    features = [ftr for ftr in pipelines[0].transformation_list.output_features]
    types = {ftr.name: ftr.dtype for ftr in features}
    shapes = {}
    for ftr in features:
        if ftr.sparse:
            shape = ftr.shape + (ftr.depth,)
        else:
            shape = (batch_size,) + ftr.shape + (ftr.depth,)
        shapes[ftr.name] = shape
    dataset = tf.data.Dataset.from_generator(lambda: stream(config,
                                                            pipelines,
                                                            shuffle,
                                                            batch_size,
                                                            sampling_probabilities,
                                                            collate),
                                             output_types=types,
                                             output_shapes=shapes)
    dataset = dataset.prefetch(5)
    return dataset
