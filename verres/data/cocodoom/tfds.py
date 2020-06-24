import os
from typing import Tuple

import tensorflow as tf

from .loader import COCODoomLoader
from .config import COCODoomStreamConfig


class DatasetConfig:

    def __init__(self,
                 batch_size: int,
                 heatmap_kernel_size: int,
                 heatmap_tensor_shape: Tuple[int, int],
                 num_categories: int):

        self.batch_size = batch_size
        self.heatmap_kernel_size = heatmap_kernel_size
        self.heatmap_tensor_shape = heatmap_tensor_shape
        self.num_categories = num_categories


class DatasetFactory:

    def __init__(self):


class PutGaussian:

    def __init__(self, kernel_size, tensor_shape, num_categories):
        self.kernel_size = kernel_size
        self.stdev = 0.3*((kernel_size-1)*0.5 - 1) + 0.8
        self.variance_2 = 2 * self.stdev**2
        self.gauss_integral = 1.
        self.tensor_shape = tensor_shape
        self.num_categories = num_categories
        self.full_tensor_shape = tf.convert_to_tensor([tensor_shape[0], tensor_shape[1], num_categories])
        self.kernel = self._make_kernel()

    def _make_kernel(self):
        x = self.kernel_size
        blob_center = tf.cast((x-1) / 2, tf.float32)
        xx = tf.range(x, dtype=tf.float32)
        exp_x = tf.exp(-(xx - blob_center) ** 2 / self.variance_2)  # [x] - [] -> x
        exp = exp_x[:, None] * exp_x[None, :]  # [x] x [x] -> [x, x]
        exp = tf.stack([exp] * self.num_categories, axis=2)[..., None]  # Repeat channel times + append multiplier
        return exp

    def process(self, coords: tf.Tensor, types: tf.Tensor):
        locations = tf.concat([coords, types], axis=-1)
        hard_heatmaps = tf.scatter_nd(locations, tf.fill([len(coords)], self.gauss_integral),
                                      shape=self.full_tensor_shape)
        soft_heatmaps = tf.nn.depthwise_conv2d(
            hard_heatmaps[None, ...], self.kernel, strides=[1, 1, 1, 1], padding="SAME")[0]

        background = tf.reduce_max(soft_heatmaps, axis=-1, keepdims=True)
        soft_heatmaps = tf.concat([soft_heatmaps, background], axis=-1)
        return soft_heatmaps


def build_box(stream_config: COCODoomStreamConfig, data_loader: COCODoomLoader):
    ...
