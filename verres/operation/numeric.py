from typing import Tuple

import numpy as np
import tensorflow as tf


def correlate(x):
    var = np.var(x, axis=1)
    cov = np.cov(x)
    cor = cov / var[None, ...]
    return cor


def meshgrid_np(shape: Tuple[int, int], dtype=None):
    m = np.stack(
        np.meshgrid(
            np.arange(shape[1]),
            np.arange(shape[0])
        ), axis=2).astype(dtype)
    return m


@tf.function
def peakfind(hmap, peak_nms):
    hmap_max = tf.nn.max_pool2d(hmap, (3, 3), strides=(1, 1), padding="SAME")

    peak = hmap_max[0] == hmap[0]
    over_nms = hmap[0] > peak_nms
    peak = tf.logical_and(peak, over_nms)
    peaks = tf.where(peak)
    scores = tf.gather_nd(hmap[0], peaks)

    return peaks, scores


def untensorize(tensor):
    if isinstance(tensor, tf.Tensor):
        tensor = tensor.numpy()
    if tensor.ndim == 4:
        tensor = tensor[0]
    return tensor


def gather_and_refine(peaks, rreg):
    refinements = tf.gather_nd(rreg, peaks)
    coords = tf.cast(peaks[:, :2], tf.float32) + refinements
    types = peaks[:, 2]
    return coords, types


def meshgrid(shape, dtype=tf.int64):
    m = tf.stack(
        tf.meshgrid(
            tf.range(shape[1]),
            tf.range(shape[0])
        ), axis=2)
    return tf.cast(m, dtype=dtype)


def quantize_coordinates(coordinates_01_scale_precise: np.ndarray, target_resolution: Tuple[int, int]):
    coordinates_target_scale_precise = upscale_coordinates(coordinates_01_scale_precise, target_resolution)
    coordinates_target_scale_rounded = np.floor(coordinates_target_scale_precise)
    return downscale_coordinates(coordinates_target_scale_rounded, target_resolution)


def upscale_coordinates(coordinates_01_scale: np.ndarray, target_resolution: Tuple[int, int]):
    target_resolution_np = np.array(target_resolution)[None, :]
    coordinates_target_scale = coordinates_01_scale * target_resolution_np
    return coordinates_target_scale


def downscale_coordinates(coordinates_target_scale: np.ndarray, full_resolution: Tuple[int, int]):
    full_resolution_np = np.array(full_resolution)[None, :]
    coordinates_01_scale = coordinates_target_scale / full_resolution_np
    return coordinates_01_scale


def _gauss_1D(center: np.ndarray, x: np.ndarray, sigma: np.ndarray, min_thresh: float) -> np.ndarray:
    """
    :param center: 1D np.ndarray[float32], shape: [num_obj]
    :param x: 1D np.ndarray[float32], shape: [tensor_shape[i]]
    :param sigma: 1D np.ndarray[float32], shape: [num_obj]
    :param min_thresh: 0D float
    :returns 2D np.ndarray[float32], shape: [num_obj, tensor_shape[i]]
    """
    C = 1.
    d = (x - center[:, None]) / sigma[:, None]  # [num_obj, tensor_shape[i]]
    e = C * np.exp(-0.5 * d**2.)
    return e * (e > min_thresh).astype(e.dtype)


def gauss_2D(
    center_xy: np.ndarray,
    sigma_xy: np.ndarray,
    tensor_shape: Tuple[int, int],
    min_thresh: float = 0.1,
) -> np.ndarray:
    range_x = np.arange(tensor_shape[0], dtype=np.float32)
    range_y = np.arange(tensor_shape[1], dtype=np.float32)
    ex = _gauss_1D(center_xy[:, 0], range_x, sigma_xy[:, 0], min_thresh)  # [num_obj, s1]
    ey = _gauss_1D(center_xy[:, 1], range_y, sigma_xy[:, 1], min_thresh)  # [num_obj, s2]
    per_object_2D_gaussians = ex[:, :, None] * ey[:, None, :]  # [num_obj, s1, s2]
    assert not np.any(np.isnan(ex)) and not np.any(np.isnan(ey))
    result = np.max(per_object_2D_gaussians, axis=0)  # [s1, s2]
    return result
