import numpy as np
import cv2

import verres as V
import verres.operation.padding
from .. import feature
from . import abstract


class HeatmapProcessor(abstract.Transformation):

    def __init__(self,
                 config: V.Config,
                 transformation_spec: dict,
                 num_classes: int):

        padded_shape = verres.operation.padding.calculate_padded_output_shape(
            input_shape=config.model.input_shape,
            model_stride=config.model.maximum_stride,
            tensor_stride=transformation_spec["stride"])

        output_feature = feature.Feature(
            name="heatmap",
            stride=transformation_spec["stride"],
            sparse=False,
            dtype="float32",
            depth=num_classes,
            shape=padded_shape)

        super().__init__(config,
                         input_fields=["bboxes", "types"],
                         output_features=[output_feature])

        self.stride = transformation_spec["stride"]
        self.num_classes = num_classes
        self.tensor_depth = self.num_classes + 1  # bg
        self.full_tensor_shape = (config.model.input_shape[0] // self.stride,
                                  config.model.input_shape[1] // self.stride,
                                  num_classes)

    @classmethod
    def from_descriptors(cls, config, data_descriptor, transformation_params):
        return cls(config,
                   num_classes=data_descriptor["num_classes"],
                   transformation_spec=transformation_params)

    def call(self, bboxes, types):
        shape = np.array(self.full_tensor_shape)
        heatmap_tensor = np.zeros(shape, dtype="float32")
        if bboxes is None or len(bboxes) == 0:
            return heatmap_tensor

        assert len(bboxes) == len(types)

        hit = 0
        for bbox, class_idx in zip(bboxes, types):
            hit = 1
            box = np.array(bbox) / self.stride
            centroid = box[:2] + box[2:] / 2
            centroid_rounded = np.clip(np.round(centroid).astype(int), [0, 0], shape[:2][::-1]-1)
            heatmap_tensor[centroid_rounded[1], centroid_rounded[0], class_idx] = 1

        if hit:
            kernel_size = 5
            heatmap_tensor = cv2.GaussianBlur(heatmap_tensor, (kernel_size, kernel_size), 0, borderType=cv2.BORDER_CONSTANT)
            heatmap_tensor /= heatmap_tensor.max()

        heatmap_tensor[..., -1] = np.max(heatmap_tensor[..., :-1], axis=-1)  # add background

        return heatmap_tensor
