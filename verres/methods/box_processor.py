import tensorflow as tf


def decode(heatmaps: tf.Tensor, refinements: tf.Tensor, whs: tf.Tensor, stride: tf.Tensor):

    hmax = tf.nn.max_pool2d(heatmaps, ksize=(5, 5), strides=(1, 1), padding='SAME')
    keep = tf.cast(tf.equal(hmax, heatmaps), heatmaps.dtype)
    hm: tf.Tensor = heatmaps * keep

    hm_shape = hm.shape
    reg_shape = refinements.shape
    wh_shape = whs.shape
    batch, width, cat = hm_shape[0], hm_shape[2], hm_shape[3]

    hm_flat = tf.reshape(hm, (batch, -1))
    reg_flat = tf.reshape(refinements, (reg_shape[0], -1, reg_shape[-1]))
    wh_flat = tf.reshape(whs, (wh_shape[0], -1, wh_shape[-1]))

    detections = []

    for _hm, _reg, _wh in zip(hm_flat, reg_flat, wh_flat):
        _scores, _inds = tf.math.top_k(_hm, k=100, sorted=True)
        _classes = tf.cast(_inds % cat, tf.float32)
        _inds = tf.cast(_inds / cat, tf.int32)
        _xs = tf.cast(_inds % width, tf.float32)
        _ys = tf.cast(tf.cast(_inds / width, tf.int32), tf.float32)
        _wh = tf.gather(_wh, _inds)
        _reg = tf.gather(_reg, _inds)

        _xs = _xs + _reg[..., 0]
        _ys = _ys + _reg[..., 1]

        _x1 = _xs - _wh[..., 0] / 2
        _y1 = _ys - _wh[..., 1] / 2
        _x2 = _xs + _wh[..., 0] / 2
        _y2 = _ys + _wh[..., 1] / 2

        # rescale to image coordinates
        _x1 = stride * _x1
        _y1 = stride * _y1
        _x2 = stride * _x2
        _y2 = stride * _y2

        detections.append(tf.stack([_x1, _y1, _x2, _y2, _scores, _classes], -1))

    return detections
