import math
from typing import List

import tensorflow as tf

import verres as V
from .dataset import Dataset
from .transformation import Transformation, TransformationList, CollateBatch


class Pipeline:

    def __init__(self,
                 config: V.Config,
                 dataset: Dataset,
                 transformations: List[Transformation]):

        self.cfg = config
        self.dataset = dataset
        self.transformation_list = TransformationList(config, transformations)

    @property
    def output_features(self):
        return self.transformation_list.output_features

    def steps_per_epoch(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self.cfg.training.batch_size
        return math.ceil(len(self.dataset) / batch_size)

    def as_tf_dataset(self,
                      shuffle: bool,
                      batch_size: int = None,
                      collate_batch="default",
                      prefetch: int = 5):

        features = [ftr for ftr in self.transformation_list.output_features]
        types = {ftr.name: ftr.dtype for ftr in features}
        shapes = {}
        for ftr in features:
            if ftr.sparse:
                shape = ftr.shape + (ftr.depth,)
            else:
                shape = (batch_size,) + ftr.shape + (ftr.depth,)
            shapes[ftr.name] = shape
        dataset = tf.data.Dataset.from_generator(lambda: self.stream(shuffle, batch_size, collate_batch),
                                                 output_types=types,
                                                 output_shapes=shapes)
        dataset = dataset.prefetch(prefetch)
        return dataset

    def stream(self,
               shuffle: bool,
               batch_size: int,
               collate_batch="default"):

        if collate_batch == "default":
            collate_batch = CollateBatch(
                self.cfg,
                features=self.output_features)

        stream = self.dataset.meta_stream(shuffle, infinite=True)

        while 1:
            meta_list = []
            for meta in stream:
                meta = self.transformation_list.process(meta)
                if not meta.get("_validity_flag"):
                    continue
                meta_list.append(meta)
                if len(meta_list) == batch_size:
                    break
            if collate_batch is not None:
                batch = collate_batch.process(meta_list)
                yield batch
            else:
                yield meta_list

    def __len__(self):
        return len(self.dataset)
