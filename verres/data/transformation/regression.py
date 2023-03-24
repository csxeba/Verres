from typing import Dict

import numpy as np
import tensorflow as tf

import verres as V
from verres.operation import numeric as T
from ... import feature
from ..sample import Sample, Label
from .abstract import Transformation


class RefinementTensor(Transformation):

    def __init__(self, config: V.Config, transformation_spec: dict):
        output_feature = feature.Feature(
            name="refinement",
            stride=transformation_spec["stride"],
            sparse=True,
            dtype="float32",
            depth=2,
            shape=(None,),
        )
        super().__init__(
            config,
            transformation_spec,
            input_fields=[],
            output_features=[output_feature])
        self.stride: int = transformation_spec["stride"]

    def call(self, sample: Sample) -> Dict[str, np.ndarray]:
        label_shape = tuple(s // self.stride for s in sample.input.shape_whc[:2])
        scale_01_precise_coords = sample.label.object_centers
        scale_01_quantized_coords = T.quantize_coordinates(scale_01_precise_coords, label_shape)
        scale_01_refinements = scale_01_precise_coords - scale_01_quantized_coords
        return {self.output_fields[0]: scale_01_refinements}

    def decode(self, tensors: Dict[str, tf.Tensor], label: Label) -> Label:
        if len(label.object_centers) == 0:
            return label

        rreg = tensors[self.output_fields[0]]
        assert len(rreg.shape) == 3

        centers = label.object_centers  # Format: image
        output_shape = tf.cast(rreg.shape[:2], tf.float32)  # format: matrix
        object_locations = tf.cast(tf.floor(centers[:, ::-1] * output_shape[None, :]), tf.int64)  # format: matrix
        refinements = tf.gather_nd(rreg, object_locations).numpy()  # format: image
        refined_centroids = centers + refinements  # format: image
        label.object_centers = refined_centroids  # format: image
        return label


class BoxWHTensor(Transformation):

    def __init__(self, config: V.Config, transformation_spec: dict):
        output_feature = feature.Feature(
            name="box_wh",
            stride=transformation_spec["stride"],
            sparse=True,
            dtype="float32",
            depth=2,
            shape=(None,),
        )
        super().__init__(
            config,
            transformation_spec,
            input_fields=[],
            output_features=[output_feature],
        )
        self.stride = transformation_spec["stride"]

    def call(self, sample: Sample) -> Dict[str, np.ndarray]:
        box_corners = sample.label.object_keypoint_coords  # format: image
        box_whs = box_corners[:, 2:] - box_corners[:, :2]
        return {self.output_fields[0]: box_whs}

    def decode(self, tensors: Dict[str, tf.Tensor], label: Label) -> Label:
        if len(label.object_centers) == 0:
            return label

        tensor = tensors[self.output_fields[0]]
        assert len(tensor.shape) == 3

        centers = label.object_centers  # format: image
        output_shape = tf.cast(tensor.shape[:2], tf.float32)  # format: matrix
        object_locations = tf.cast(tf.floor(centers[..., ::-1] * output_shape), tf.int32)  # format: matrix
        box_params = tf.gather_nd(tensor, object_locations).numpy()  # format: image
        box_corners = tf.concat([centers + box_params / 2., centers - box_params / 2.], axis=1)  # format: image
        label.object_keypoint_coords = box_corners
        return label


class BoxCornerTensor(Transformation):

    def __init__(self, config: V.Config, transformation_spec: dict):
        output_feature = feature.Feature(
            name="box_corners",
            stride=transformation_spec["stride"],
            sparse=True,
            dtype="float32",
            depth=4,
            shape=(),
        )
        super().__init__(
            config,
            transformation_spec,
            input_fields=[],
            output_features=[output_feature],
        )
        self.stride = transformation_spec["stride"]

    def call(self, sample: Sample) -> Dict[str, np.ndarray]:
        return {"box_corners": sample.label.object_keypoint_coords}

    def decode(self, tensors: Dict[str, tf.Tensor], label: Label) -> Label:
        if len(label.object_centers) == 0:
            return label
        tensor = tensors[self.output_fields[0]]
        assert len(tensor.shape) == 3

        centers = label.object_centers  # format: image
        output_shape = tf.cast(tensor.shape[:2], tf.float32)  # format: matrix
        object_locations = tf.cast(tf.floor(centers[..., ::-1] * output_shape), tf.int32)  # format: matrix
        box_corners = tf.gather_nd(tensor, object_locations)  # format: image
        label.object_keypoint_coords = box_corners
        return label
