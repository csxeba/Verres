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


def build_box(stream_config: COCODoomStreamConfig, data_loader: COCODoomLoader):
    ...
