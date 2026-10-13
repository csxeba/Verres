from typing import List, Dict

import tensorflow as tf

import verres as V
from verres.data import transformation
from verres.feature import Feature
from verres.data.sample import Sample, Label


class Codec:

    def __init__(self, config: V.Config, transformation_specs: List[dict]):
        super().__init__()
        self.transformation_list = transformation.factory(config, transformation_specs)
        self.config = config

    def encode_sample(self, sample: Sample) -> Sample:
        for trf in self.transformation_list:
            sample = trf.process_sample(sample)
        return sample

    def decode(self, predictions: Dict[str, tf.Tensor]) -> Label:
        detection_label = Label()
        for trf in self.transformation_list:
            detection_label = trf.decode(predictions, detection_label)
        return detection_label

    @property
    def output_features(self) -> List[Feature]:
        result = []
        for trf in self.transformation_list:
            result.extend(trf.output_features)
        return result
