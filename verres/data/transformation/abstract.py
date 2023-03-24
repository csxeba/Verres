import time
from typing import Union, Tuple, Optional, Dict

import numpy as np
import tensorflow as tf

import verres as V
from ... import feature
from ..sample import Sample, Label


def _as_tuple(value) -> tuple:
    if value is None:
        value = ()
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, tuple):
        value = value,
    return value


def _unpack_output_features(output_features: Tuple[feature.Feature]) -> Tuple[feature.Feature]:
    features = []
    for ftr in output_features:
        if ftr is None:
            continue
        if isinstance(ftr, feature.MultiFeature):
            features.extend(ftr.feature_list)
        else:
            features.append(ftr)

    return tuple(features)


class Transformation:

    def __init__(self,
                 config: V.Config,
                 transformation_spec: Optional[dict],
                 input_fields: Union[tuple, list, str],
                 output_features: Optional[Union[tuple, list, feature.Feature]] = None,
                 output_fields: Union[tuple, list, str] = "default"):

        self.cfg = config
        self.spec = transformation_spec
        self.input_fields: Tuple[str] = _as_tuple(input_fields)
        self.output_fields: Tuple[str] = _as_tuple(output_fields)
        self.output_features: Tuple[feature.Feature] = _unpack_output_features(_as_tuple(output_features))
        self._net_processing_time: float = 0.
        self._num_calls: int = 0

        if output_fields == "default":
            self.output_fields = tuple(ftr.meta_field for ftr in self.output_features)

    def _write_result_to_sample(self, sample: Sample, results: dict) -> Sample:
        for field in self.output_fields:
            sample.encoded_tensors[field] = results[field]
        return sample

    def process_sample(self, sample: Sample):
        process_start_timestamp = time.time()
        results = self.call(sample)
        self._net_processing_time += process_start_timestamp - time.time()
        self._num_calls += 1
        self._write_result_to_sample(sample, results)
        return sample

    def report_runtime(self):
        if self._num_calls == 0:
            return np.inf
        return self._net_processing_time / self._num_calls

    def call(self, sample: Sample) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def decode(self, tensors: Dict[str, tf.Tensor], label: Optional[Label] = None) -> Label:
        raise NotImplementedError
