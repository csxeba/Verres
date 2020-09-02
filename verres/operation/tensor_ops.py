import tensorflow as tf


@tf.function
def peakfind_and_refine(hmap, rreg, peak_nms, stride):
    hmap_max = tf.nn.max_pool2d(hmap, (3, 3), strides=(1, 1), padding="SAME")

    peak = hmap_max[0] == hmap[0]
    over_nms = hmap[0] > peak_nms
    peak = tf.logical_and(peak, over_nms)
    peaks = tf.where(peak)
    scores = tf.gather_nd(hmap[0], peaks)

    refinements = tf.stack([
        tf.gather_nd(rreg[0, ..., 1::2], peaks),
        tf.gather_nd(rreg[0, ..., 0::2], peaks)], axis=-1)

    refined_centroids = (tf.cast(peaks[:, :2], tf.float32) + refinements) * stride
    centroids = tf.cast(tf.round(refined_centroids[:, ::-1]), tf.int64)

    return centroids, peak[:, 2], scores
