import tensorflow as tf


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


def gather_vectors(peaks, tensor):
    return tf.stack([
        tf.gather_nd(tensor[..., 1::2], peaks),
        tf.gather_nd(tensor[..., 0::2], peaks)], axis=-1)


def gather_and_refine(peaks, rreg):
    refinements = gather_vectors(peaks, rreg)
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
