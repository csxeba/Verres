import time
from typing import Union, List, Tuple, Optional

import numpy as np

import verres as V
from .. import feature


class InvalidDataPoint(RuntimeError):
    ...


def _as_tuple(value) -> tuple:
    if value is None:
        value = ()
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, tuple):
        value = value,
    return value


def _unpack_output_features(output_features) -> Tuple[feature.Feature]:
    features = []
    for ftr in output_features:
        if ftr is None:
            continue
        if isinstance(ftr, feature.MultiFeature):
            features.extend(ftr.feature_list)
        else:
            features.append(ftr)

    # noinspection PyTypeChecker
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

    @classmethod
    def from_descriptors(cls, config: V.Config, data_descriptor, transformation_params):
        raise NotImplementedError

    def _read_parameters_from_metadata(self, metadata: dict):
        return [metadata[field] for field in self.input_fields]

    def _write_result_to_metadata(self, metadata: dict, results: list):
        results = _as_tuple(results)
        for i, field in enumerate(self.output_fields):
            metadata[field] = results[i]
        return metadata

    def process(self, metadata: dict):
        if not metadata.get("_validity_flag", True):
            return metadata
        call_parameters = self._read_parameters_from_metadata(metadata)
        process_start_timestamp = time.time()
        results = self.call(*call_parameters)
        self._net_processing_time += process_start_timestamp - time.time()
        self._num_calls += 1
        metadata = self._write_result_to_metadata(metadata, results)
        return metadata

    def report_runtime(self):
        if self._num_calls == 0:
            return np.inf
        return self._net_processing_time / self._num_calls

    def call(self, *args, **kwargs):
        raise NotImplementedError


class TransformationList(Transformation):

    def __init__(self, config: V.Config, transformation_list: List[Transformation]):
        super().__init__(config,
                         transformation_spec=None,
                         input_fields=(),
                         output_features=())
        self.transformation_list = transformation_list
        self.output_features = []
        for transformation in transformation_list:
            self.output_features.extend(list(transformation.output_features))

    @classmethod
    def from_descriptors(cls, config: V.Config, data_descriptor, feature_descriptor):
        raise NotImplementedError

    def process(self, metadata: dict):
        for transformation in self.transformation_list:
            metadata = transformation.process(metadata)
        return metadata

    def call(self, *args, **kwargs):
        raise NotImplementedError
