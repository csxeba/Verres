from typing import Dict, Optional

import numpy as np
import tensorflow as tf

import verres as V
from verres.operation import numeric as T
from ... import feature
from ..sample import Sample, Label
from . import abstract


class UniformSigmaHeatmapProcessor(abstract.Transformation):

    def __init__(self, config: V.Config, transformation_spec: dict):

        padded_shape = V.operation.padding.calculate_padded_output_shape(
            input_shape=config.model.input_shape_hw,
            model_stride=config.model.maximum_stride,
            tensor_stride=transformation_spec["stride"])

        output_feature = feature.Feature(
            name="heatmap",
            stride=transformation_spec["stride"],
            sparse=False,
            dtype=config.context.float_precision,
            depth=config.class_mapping.num_classes,
            shape=padded_shape)

        super().__init__(config,
                         transformation_spec,
                         input_fields=["bboxes", "types"],
                         output_features=[output_feature])

        self.stride = transformation_spec["stride"]
        self.num_classes = config.class_mapping.num_classes
        self.tensor_depth = self.num_classes + 1  # bg
        self.full_tensor_shape = (config.model.input_shape_hw[0] // self.stride,
                                  config.model.input_shape_hw[1] // self.stride,
                                  self.num_classes)  # Format: matrix

    def call(self, sample: Sample) -> Dict[str, np.ndarray]:
        gauss = np.zeros(self.full_tensor_shape, dtype=np.float32)
        centers = np.floor(T.upscale_coordinates(sample.label.object_centers[:, ::-1], self.full_tensor_shape[:2]))
        for type_id in range(self.full_tensor_shape[-1]):
            object_mask = sample.label.object_types == type_id
            if not np.any(object_mask):
                continue
            centers_of_type = centers[object_mask]
            sigma = np.full_like(centers_of_type, fill_value=1.)
            gauss[..., type_id] = T.gauss_2D(center_xy=centers_of_type, sigma_xy=sigma, tensor_shape=gauss.shape[:2])
        if self.cfg.context.debug:
            x, y, c = tuple(centers[:, 0].astype(int)), tuple(centers[:, 1].astype(int)), sample.label.object_types
            np.testing.assert_array_equal(gauss[x, y, c], 1)
            assert not np.any(np.isnan(gauss))
        return {self.output_fields[0]: gauss}

    def decode(self, tensors: Dict[str, tf.Tensor], label_instance: Optional[Label] = None) -> Label:
        hmap = tensors[self.output_fields[0]]

        peaks, scores = T.peakfind(hmap, 0.1)

        output_shape = tf.cast(tf.shape(hmap)[:2], tf.float32)

        centroids = tf.cast(peaks[:, :2], tf.float32) / output_shape
        types = peaks[:, 2]

        if label_instance is None:
            label_instance = Label()

        label_instance.object_centers = centroids.numpy()[..., ::-1]
        label_instance.object_types = types.numpy()
        label_instance.object_scores = scores.numpy()

        return label_instance


class VariableSigmaHeatmapProcessor(UniformSigmaHeatmapProcessor):

    @staticmethod
    def gaussian2D(shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]

        h = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    @staticmethod
    def calculate_radius(bbox, min_overlap=0.7):
        width, height = np.ceil(bbox)

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
        return int(min(r1, r2, r3))

    def draw_gaussian(self, heatmap, center, radius):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[:2]

        left = int(min(x, radius))
        right = int(min(width - x, radius + 1))
        top = int(min(y, radius))
        bottom = int(min(height - y, radius + 1))

        masked_heatmap = heatmap[y-top:y+bottom, x-left:x+right]
        masked_gaussian = gaussian[radius-top:radius+bottom, radius-left:radius+right]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

        return heatmap

    def call(self, sample: Sample) -> Dict[str, np.ndarray]:
        shape = np.array(self.full_tensor_shape)
        heatmap_tensor = np.zeros(shape, dtype=self.cfg.context.float_precision)

        centers = sample.label.object_centers
        types = sample.label.object_types
        box_whs = sample.label.object_keypoint_coords[:, 2:] - sample.label.object_keypoint_coords[:, :2]
        for center, whs, type_id in zip(centers, box_whs, types):
            scale_xy = heatmap_tensor.shape[:2][::-1]
            radius = self.calculate_radius(whs * scale_xy)
            heatmap_tensor[..., type_id] = self.draw_gaussian(heatmap_tensor[..., type_id], center * scale_xy, radius)
        return {self.output_fields[0]: heatmap_tensor}

    def decode(self, tensors: Dict[str, tf.Tensor], label: Optional[Label] = None) -> Label:
        hm_out = tensors[self.output_fields[0]]

        outshape = tf.shape(hm_out)  # Format: matrix
        width = outshape[1]
        cat = outshape[2]

        hmax = tf.nn.max_pool2d(hm_out[None, ...], (3, 3), strides=(1, 1), padding='SAME')[0]
        keep = tf.cast(hmax == hm_out, tf.float32)
        hm = hm_out * keep

        _hm = tf.reshape(tf.transpose(hm, (1, 2, 0)), (-1,))

        _scores, _inds = tf.math.top_k(_hm, k=100, sorted=True)
        _classes = _inds % cat
        _inds = tf.cast(_inds / cat, tf.int32)
        _xs = tf.cast(_inds % width, tf.float32)
        _ys = tf.cast(tf.floor(_inds / width), tf.float32)  # xy format: Image

        scx = hm_out.shape[1]
        scy = hm_out.shape[0]  # note: transpose

        _xs = _xs / scx
        _ys = _ys / scy

        centroids = tf.stack([_xs, _ys], axis=1)

        if label is None:
            label = Label()

        valid_score = _scores > 0.1

        label.object_centers = centroids[valid_score]
        label.object_types = _classes[valid_score]
        label.object_scores = _scores[valid_score]

        return label
