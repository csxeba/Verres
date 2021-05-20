import time
from typing import Union, List, Tuple

import verres as V
from .. import feature


def _as_tuple(value) -> tuple:
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, tuple):
        value = value,
    return value


class Transformation:

    def __init__(self,
                 config: V.Config,
                 input_fields: Union[tuple, list, str],
                 output_features: Union[tuple, list, feature.Feature]):

        self.cfg = config
        self.input_fields: Tuple[str] = _as_tuple(input_fields)
        self._output_features: Tuple[feature.Feature] = _as_tuple(output_features)
        self._net_processing_time: float = 0.
        self._num_calls: int = 0

    @property
    def output_features(self):
        features = []
        for ftr in self._output_features:
            if isinstance(ftr, feature.MultiFeature):
                features.extend(ftr.feature_list)
            else:
                features.append(ftr)
        return features

    @classmethod
    def from_descriptors(cls, config: V.Config, data_descriptor, transformation_params):
        raise NotImplementedError

    def _read_parameters_from_metadata(self, metadata: dict):
        return [metadata[field] for field in self.input_fields]

    def _write_result_to_metadata(self, metadata: dict, results: list):
        results = _as_tuple(results)
        for i, ftr in enumerate(self.output_features):
            metadata[ftr.meta_field] = results[i]
        return metadata

    def process(self, metadata: dict):
        call_parameters = self._read_parameters_from_metadata(metadata)
        process_start_timestamp = time.time()
        results = self.call(*call_parameters)
        self._net_processing_time += process_start_timestamp - time.time()
        self._num_calls += 1
        metadata = self._write_result_to_metadata(metadata, results)
        return metadata

    def report_runtime(self):
        if self._num_calls == 0:
            return 0.
        return self._net_processing_time / self._num_calls

    def call(self, *args, **kwargs):
        raise NotImplementedError


class TransformationList(Transformation):

    def __init__(self, config: V.Config, transformation_list: List[Transformation]):
        super().__init__(config,
                         input_fields=(),
                         output_features=())
        self.transformation_list = transformation_list
        self._output_features = []
        for transformation in transformation_list:
            self._output_features.extend(list(transformation.output_features))

    @classmethod
    def from_descriptors(cls, config: V.Config, data_descriptor, feature_descriptor):
        raise NotImplementedError

    @property
    def output_features(self) -> List[feature.Feature]:
        result = []
        for transformation in self.transformation_list:
            result.extend(transformation.output_features)
        return result

    def process(self, metadata: dict):
        for transformation in self.transformation_list:
            metadata = transformation.process(metadata)
        return metadata

    def call(self, *args, **kwargs):
        raise NotImplementedError
