import random
from concurrent.futures import ThreadPoolExecutor
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

    if len(pipelines) != len(sampling_probabilities):
        raise RuntimeError("There should be a sampling probability for every pipeline passed to stream()")

    if collate == "default":
        collate = V.data.transformation.CollateBatch(config, features=pipelines[0].output_features)

    sampling_probabilities = np.array(sampling_probabilities)
    sampling_probabilities = sampling_probabilities / np.sum(sampling_probabilities)

    meta_iterators = [iter(pipeline.dataset.meta_stream(shuffle=shuffle, infinite=True)) for pipeline in pipelines]
    codecs = [pipeline.codec for pipeline in pipelines]

    # Encoding a sample is dominated by `cv2.imread`, which releases the GIL, so threads scale it.
    with ThreadPoolExecutor() as pool:
        while 1:
            chosen = random.choices(range(len(pipelines)), sampling_probabilities, k=batch_size)
            # `next()` stays on this thread: advancing one generator from two threads is an error.
            samples = [next(meta_iterators[index]) for index in chosen]
            samples = list(pool.map(lambda pair: pair[1].encode_sample(pair[0]),
                                    zip(samples, (codecs[index] for index in chosen))))

            if collate is not None:
                yield collate.process(samples)
            else:
                yield samples


def get_tf_dataset(config: V.Config,
                   pipelines: List[Pipeline],
                   shuffle: bool,
                   batch_size: int,
                   sampling_probabilities: List[float] = "uniform",
                   collate: callable = "default"):

    features = [ftr for ftr in pipelines[0].output_features]
    types = {ftr.name: ftr.dtype for ftr in features}
    shapes = {}
    for ftr in features:
        if ftr.sparse:
            shape = ftr.shape + (ftr.depth,)
            if "object_center_locations" not in types:
                types["object_center_locations"] = tf.int64
            if "object_center_locations" not in shapes:
                shapes["object_center_locations"] = (None, 3)
        else:
            shape = (batch_size,) + ftr.shape + (ftr.depth,)
        shapes[ftr.name] = shape
    dataset = tf.data.Dataset.from_generator(
        lambda: stream(
            config,
            pipelines,
            shuffle,
            batch_size,
            sampling_probabilities,
            collate,
        ),
        output_types=types,
        output_shapes=shapes,
    )
    dataset = dataset.prefetch(5)
    return dataset
