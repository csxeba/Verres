from typing import Dict

import numpy as np
import cv2
import tensorflow as tf

import verres as V
from verres.operation import numeric as T
from ... import feature
from ..sample import Sample, Label
from .abstract import Transformation
from verres.operation import masking


class SemanticSegmentationTensor(Transformation):

    def __init__(self, config: V.Config, transformation_spec: dict):
        output_feature = feature.Feature(
            name="semantic_segmentation",
            stride=transformation_spec["stride"],
            dtype="uint8",
            shape=V.operation.padding.calculate_padded_output_shape(
                input_shape=config.model.input_shape_hw,
                model_stride=config.model.maximum_stride,
                tensor_stride=transformation_spec["stride"]),
            depth=1,
            sparse=False,
        )
        super().__init__(
            config,
            transformation_spec,
            input_fields=[],
            output_features=[output_feature],
        )

        output_shape = tuple(s // transformation_spec["stride"] for s in config.model.input_shape_hw)
        self.stride = transformation_spec["stride"]
        self.num_classes = config.class_mapping.num_classes
        self.semantic_tensor_shape = output_shape + (1,)

    def call(self, sample: Sample) -> Dict[str, np.ndarray]:
        canvas = np.zeros(self.semantic_tensor_shape, dtype="int64")
        target_wh = self.semantic_tensor_shape[1::-1]
        for class_idx, segmentation_repr in zip(sample.label.object_types, sample.label.segmentation_repr):
            instance_mask = np.squeeze(masking.mask_from_representation(
                segmentation_repr, sample.input.shape_whc[:2][::-1]))  # rasterise at full input resolution
            if instance_mask.shape[::-1] != target_wh:
                # Masks are native, targets live on the model-input grid.
                instance_mask = cv2.resize(instance_mask, target_wh, interpolation=cv2.INTER_NEAREST)
            canvas[instance_mask] = class_idx+1
        return {self.output_fields[0]: canvas}

    def decode(self, tensors: Dict[str, tf.Tensor], label: Label) -> Label:
        pixel_logits = tensors[self.output_fields[0]]
        label.semantic_segmentation = tf.argmax(pixel_logits, axis=-1)
        return label


class InstanceSegmentationTensor(Transformation):

    def __init__(self, config: V.Config, transformation_spec: dict):
        output_feature = feature.Feature(
            name="instance_segmentation",
            stride=transformation_spec["stride"],
            dtype="float32",
            shape=V.operation.padding.calculate_padded_output_shape(
                input_shape=config.model.input_shape_hw,
                model_stride=config.model.maximum_stride,
                tensor_stride=transformation_spec["stride"]),
            depth=2,
            sparse=False,
        )
        super().__init__(
            config,
            transformation_spec,
            input_fields=[],
            output_features=[output_feature],
        )
        self.stride = transformation_spec["stride"]
        self.offset_error_threshold = transformation_spec.get("offset_error_threshold", 2.0)

    def call(self, sample: Sample) -> Dict[str, np.ndarray]:
        # Targets live on the model-input grid, masks are rasterised at the native resolution first.
        canvas_shape = tuple(s // self.stride for s in self.cfg.model.input_shape_hw)  # Format: Matrix (HW)
        canvas_shape_np = np.array(canvas_shape[::-1])  # Format: Image (WH)
        instance_canvas = np.zeros(canvas_shape + (2,), dtype="float32")
        object_centers = sample.label.object_centers  # Format: Image (WH)

        for centroid_scale_01, segmentation_repr in zip(object_centers, sample.label.segmentation_repr):
            instance_mask = np.squeeze(masking.mask_from_representation(
                segmentation_repr, sample.input.shape_whc[:2][::-1]))  # Matrix (HW)
            if instance_mask.shape != canvas_shape:
                instance_mask = cv2.resize(instance_mask, canvas_shape_np, interpolation=cv2.INTER_NEAREST)
            coords_scale_01 = np.argwhere(instance_mask)[:, ::-1] / canvas_shape_np[None, :]  # Format: Image (WH)
            instance_canvas[instance_mask] = centroid_scale_01[None, :] - coords_scale_01
        return {self.output_fields[0]: instance_canvas}

    def decode(self, tensors: Dict[str, tf.Tensor], label: Label) -> Label:
        if len(label.object_centers) == 0:
            return label
        iseg = tensors[self.output_fields[0]]
        assert len(iseg.shape) == 3

        semseg = label.semantic_segmentation
        centroids_scale_01 = label.object_centers

        # Image (WH) order, as in the encoder: mesh_xy is (x, y), the shape is reversed to (w, h).
        grid_wh = tf.cast(tf.convert_to_tensor(semseg.shape[:2]), tf.float32)[::-1]
        non_bg = semseg > 0
        coords_non_bg_scale_01 = T.meshgrid(semseg.shape[:2], dtype=tf.float32)[non_bg] / grid_wh
        iseg_offset_scale_01 = iseg[non_bg] + coords_non_bg_scale_01

        D = tf.reduce_sum(  # [M, 1, 2] - [N, 1, 2] -> [M, N, 2]
            tf.square(iseg_offset_scale_01[:, None, :] - centroids_scale_01[None, :, :]), axis=2)  # -> [M, N]
        affiliations = tf.argmin(D, axis=1)
        offset_errors = tf.reduce_min(D, axis=1)

        offset_error_ok = offset_errors < self.offset_error_threshold

        label.instance_pixel_coords = coords_non_bg_scale_01[offset_error_ok]
        label.instance_pixel_affiliations = affiliations[offset_error_ok]

        return label
