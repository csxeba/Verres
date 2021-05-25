import numpy as np
import cv2

import verres as V
from .. import feature
from . import abstract


class _HeatmapProcessor(abstract.Transformation):

    def __init__(self,
                 config: V.Config,
                 transformation_spec: dict,
                 num_classes: int):

        padded_shape = V.operation.padding.calculate_padded_output_shape(
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
                                  num_classes)  # Format: image

    @classmethod
    def from_descriptors(cls, config, data_descriptor, transformation_params):
        return cls(config,
                   num_classes=data_descriptor["num_classes"],
                   transformation_spec=transformation_params)

    def call(self, *args, **kwargs):
        raise NotImplementedError


class UniformSigmaHeatmapProcessor(_HeatmapProcessor):

    def call(self, bboxes, types):
        """
        :param bboxes: np.ndarray
            Stacked MSCOCO-format (image csystem) bounding box representations [x0, y0, w, h]
        :param types: np.ndarray
            MSCOCO category_ids of interest mapped to 0..N
        """
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
            centroid_rounded = np.round(centroid).astype(int)[::-1]
            assert centroid_rounded[0] <= shape[0] and centroid_rounded[1] <= shape[1]
            centroid_rounded = np.clip(centroid_rounded, [0, 0], shape[:2]-1)
            heatmap_tensor[centroid_rounded[0], centroid_rounded[1], class_idx] = 1

        if hit:
            kernel_size = 5
            heatmap_tensor = cv2.GaussianBlur(heatmap_tensor, (kernel_size, kernel_size), 0, borderType=cv2.BORDER_CONSTANT)
            heatmap_tensor /= heatmap_tensor.max()

        return heatmap_tensor


class VariableSigmaHeatmapProcessor(_HeatmapProcessor):

    @staticmethod
    def gaussian2D(shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]

        h = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    @staticmethod
    def calculate_radius(bbox, min_overlap=0.7):
        width, height = np.ceil(bbox[2:])

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def draw_gaussian(self, heatmap, center, radius):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y-top:y+bottom, x-left:x+right]
        masked_gaussian = gaussian[radius-top:radius+bottom, radius-left:radius+right]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

        return heatmap

    def call(self, bboxes, types):
        shape = np.array(self.full_tensor_shape[-1:] + self.full_tensor_shape[:-1])
        heatmap_tensor = np.zeros(shape, dtype="float32")
        for box, type_id in zip(bboxes, types):
            x0y0 = box[:2]
            center = x0y0 + box[2:] / 2
            radius = self.calculate_radius(box)
            self.draw_gaussian(heatmap_tensor[type_id], center, radius)

        return heatmap_tensor.transpose((1, 2, 0))
