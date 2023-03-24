import math
from typing import List, Iterable, Optional, Dict

import tensorflow as tf

import verres as V
from .dataset import Dataset
from .sample import Sample
from .transformation import CollateBatch
from .codec import Codec


class Pipeline:

    def __init__(self,
                 config: V.Config,
                 dataset: Dataset,
                 codec: Codec):

        self.cfg = config
        self.dataset = dataset
        self.codec = codec

    @property
    def output_features(self):
        return self.codec.output_features

    def steps_per_epoch(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self.cfg.training.batch_size
        return math.ceil(len(self.dataset) / batch_size)

    def stream(
        self,
        shuffle: bool,
        batch_size: int,
        collate_batch: Optional[CollateBatch] = None,
    ) -> Iterable[List[Sample]]:
        if collate_batch == "default":
            collate_batch = CollateBatch(
                self.cfg,
                features=self.codec.output_features,
            )

        stream = self.dataset.meta_stream(shuffle, infinite=True)
        while 1:
            sample_list: List[Sample] = []
            for sample in stream:
                sample = self.codec.encode_sample(sample)
                sample_list.append(sample)
                if len(sample_list) == batch_size:
                    break
            if collate_batch is not None:
                batch = collate_batch.process(sample_list)
                yield batch
            else:
                yield sample_list

    def __len__(self):
        return len(self.dataset)
