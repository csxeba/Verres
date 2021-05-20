from typing import Tuple

import numpy as np
import cv2

from .. import feature
from .abstract import Transformation
from verres.operation import masking


class SemanticSegmentationTensor(Transformation):

    def __init__(self,
                 image_shape: Tuple[int, int],
                 output_stride: int,
                 num_classes: int):

        super().__init__(
            input_fields=["types", "segmentations"],
            output_features=feature.SEMANTIC_SEG)

        output_shape = tuple(s // output_stride for s in image_shape)
        self.image_shape = image_shape
        self.stride = output_stride
        self.num_classes = num_classes
        self.semantic_tensor_shape = output_shape + (1,)
        self.instance_tensor_shape = output_shape + (2,)

    @classmethod
    def from_descriptors(cls, data_descriptor, feature_descriptor):
        stride = feature_descriptor["stride"]
        return cls(image_shape=data_descriptor["image_shape"],
                   output_stride=stride,
                   num_classes=data_descriptor["num_classes"])

    def call(self, types, segmentations):
        canvas = np.zeros(self.semantic_tensor_shape, dtype="int64")
        for class_idx, segmentation_repr in zip(types, segmentations):
            instance_mask = np.squeeze(masking.mask_from_representation(segmentation_repr, self.image_shape))
            if self.stride > 1:
                instance_mask = cv2.resize(instance_mask, (0, 0), fx=1/self.stride, fy=1/self.stride,
                                           interpolation=cv2.INTER_NEAREST)
            canvas[instance_mask] = class_idx+1


class PanopticSegmentationTensor(Transformation):

    def __init__(self,
                 image_shape: Tuple[int, int],
                 output_stride: int,
                 num_classes: int):

        super().__init__(
            input_fields=["types", "segmentations"],
            output_features=[feature.SEMANTIC_SEG, feature.INSTANCE_SEG])

        output_shape = tuple(s // output_stride for s in image_shape[:2])
        self.image_shape = image_shape[:2]
        self.stride = output_stride
        self.num_classes = num_classes
        self.semantic_tensor_shape = output_shape
        self.instance_tensor_shape = output_shape + (2,)

    @classmethod
    def from_descriptors(cls, data_descriptor, feature_descriptor):
        stride = feature_descriptor["stride"]
        return cls(image_shape=data_descriptor["image_shape"],
                   output_stride=stride,
                   num_classes=data_descriptor["num_classes"])

    def call(self, types, segmentations):
        semantic_canvas = np.zeros(self.semantic_tensor_shape, dtype="int64")
        instance_canvas = np.zeros(self.instance_tensor_shape, dtype="float32")

        for [class_idx], segmentation_repr in zip(types, segmentations):
            instance_mask = np.squeeze(masking.mask_from_representation(segmentation_repr, self.image_shape))
            if self.stride > 1:
                instance_mask = cv2.resize(instance_mask, (0, 0), fx=1/self.stride, fy=1/self.stride,
                                           interpolation=cv2.INTER_NEAREST)
            coords = np.argwhere(instance_mask)  # type: np.ndarray

            semantic_canvas[instance_mask] = class_idx+1
            instance_canvas[instance_mask] = coords.mean(axis=0, keepdims=True) - coords

        return [semantic_canvas, instance_canvas]
