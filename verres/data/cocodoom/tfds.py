import os

import numpy as np
import tensorflow as tf
from tensorflow import data as tfd

from .loader import COCODoomLoader
from .config import COCODoomStreamConfig

from verres.utils import cocodoom_utils


def _read_and_process_image(path, normalization_constant, what):
    png_kw = {"depth": {"channels": 1, "dtype": tf.uint16},
              "image": {"channels": 3, "dtype": tf.uint8}}[what]
    raw = tf.read_file(path)
    data = tf.image.decode_png(raw, **png_kw)
    data = tf.cast(data, tf.float32)
    data /= normalization_constant
    return data


def _preprocess_depth_tuple(image_paths, depth_paths):
    images = _read_and_process_image(image_paths, normalization_constant=2. ** 8. - 1., what="image")
    depth_maps = _read_and_process_image(depth_paths, normalization_constant=2. ** 16. - 1., what="depth")
    return images, depth_maps


def _make_heatmap_tfd(meta: dict, config: COCODoomStreamConfig, loader: COCODoomLoader):
    canvas = tf.zeros([meta["height"] // loader.cfg.stride,
                       meta["width"] // loader.cfg.stride,
                       loader.num_classes], dtype="float32")


def build_depth(stream_config: COCODoomStreamConfig, data_loader: COCODoomLoader):
    meta_stream = cocodoom_utils.apply_filters(data_loader.image_meta.values(), stream_config, data_loader)
    metae = sorted(meta_stream, key=lambda meta: meta["id"])

    image_paths_py = [os.path.join(data_loader.cfg.images_root, meta["file_name"]) for meta in metae]
    image_paths = tfd.Dataset.from_tensor_slices(np.array(image_paths_py))
    depth_paths = tfd.Dataset.from_tensor_slices(
        np.array([path.replace("/rgb/", "/depth/") for path in image_paths_py]))

    combo_paths = tfd.Dataset.zip((image_paths, depth_paths)).repeat()
    if stream_config.shuffle:
        combo_paths = combo_paths.shuffle(buffer_size=len(metae))
    combo_paths = combo_paths.prefetch(buffer_size=tfd.experimental.AUTOTUNE)

    dataset = combo_paths.map(_preprocess_depth_tuple).batch(stream_config.batch_size)
    return dataset


def build_box(stream_config: COCODoomStreamConfig, data_loader: COCODoomLoader):
    ...
