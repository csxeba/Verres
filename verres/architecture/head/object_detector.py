from typing import List

import tensorflow as tf

import verres as V
from verres.operation import numeric as T
from ..layers import block
from .base import VRSHead
from ..backbone import FeatureSpec


class OD(VRSHead):

    def __init__(self, config: V.Config, input_features: List[FeatureSpec]):
        super().__init__()
        spec = config.model.head_spec.copy()
        if len(input_features) == 1:
            input_features = [input_features[0], input_features[0]]
        self.centroid_feature_spec, self.box_feature_spec = input_features
        nearest_po2 = V.utils.numeric_utils.ceil_to_nearest_power_of_2(spec["num_classes"])
        self.hmap_head = block.VRSHead(
            spec.get("head_convolution_width", nearest_po2),
            output_width=spec["num_classes"])
        self.rreg_head = block.VRSHead(
            spec.get("head_convolution_width", nearest_po2*2),
            output_width=spec["num_classes"]*2)
        self.boxx_head = block.VRSHead(
            spec.get("head_convolution_width", nearest_po2*2),
            output_width=spec["num_classes"]*2)

        self.peak_nms = spec.get("peak_nms", 0.1)

    def call(self, inputs, training=None, mask=None):
        centroid_features, box_features = inputs
        hmap = self.hmap_head(centroid_features, training=training)
        rreg = self.rreg_head(centroid_features, training=training)
        boxx = self.boxx_head(centroid_features, training=training)
        hmap = tf.cond(training,
                       true_fn=lambda: hmap,
                       false_fn=lambda: tf.nn.sigmoid(hmap))
        return {"heatmap": hmap, "box_wh": boxx, "refinement": rreg}

    def postprocess_network_output(self, predictions):
        hmap = predictions["heatmap"]
        rreg = predictions["refinement"]
        bbox = predictions["box_wh"]

        peaks, scores = T.peakfind(hmap, self.peak_nms)

        output_shape = tf.cast(tf.shape(hmap)[1:3], tf.float32)

        refinements = tf.stack([
            tf.gather_nd(rreg[0, ..., 0::2], peaks),
            tf.gather_nd(rreg[0, ..., 1::2], peaks)], axis=-1)

        box_params = tf.stack([
            tf.gather_nd(bbox[0, ..., 0], peaks[:, :2]),
            tf.gather_nd(bbox[0, ..., 1], peaks[:, :2])], axis=-1)

        refined_centroids = (tf.cast(peaks[:, :2], tf.float32) + refinements) / output_shape
        box_params = box_params / output_shape
        types = peaks[:, 2]

        boxes = tf.concat([refined_centroids, box_params], axis=1)

        result = {"boxes": boxes, "types": types, "scores": scores}

        return result


class CTDet(VRSHead):

    def __init__(self, config: V.Config):
        super().__init__()
        spec = config.model.head_spec.copy()
        self.hmap_head = block.VRSHead(spec.get("head_convolution_width", 32), output_width=spec["num_classes"])
        self.rreg_head = block.VRSHead(spec.get("head_convolution_width", 32), output_width=spec["num_classes"]*2)
        self.boxx_head = block.VRSHead(spec.get("head_convolution_width", 32), output_width=spec["num_classes"]*2)
        self.peak_nms = spec.get("peak_nms", 0.1)

    def call(self, inputs, training=None, mask=None):
        centroid_features, box_features = inputs
        hmap = self.hmap_head(centroid_features, training=training)
        rreg = self.rreg_head(centroid_features, training=training)
        boxx = self.boxx_head(centroid_features, training=training)
        return {"heatmap": hmap, "box_wh": boxx, "refinement": rreg}

    def postprocess_network_output(self, predictions):

        hm_out = predictions["heatmap"]
        reg_out = predictions["refinement"]
        wh_out = predictions["box_wh"]

        if self.cfg.model_params.heatmap_has_background:
            hm_out = hm_out[..., :-1]

        outshape = tf.shape(hm_out)  # Format: matrix
        width = outshape[2]
        cat = outshape[3]

        hmax = tf.nn.max_pool2d(hm_out, (3, 3), strides=(1, 1), padding='SAME')
        keep = tf.cast(hmax == hm_out, tf.float32)
        hm = hm_out * keep

        _hm = tf.reshape(hm[0], (-1,))
        _reg = tf.reshape(reg_out[0], (-1, reg_out.shape[-1]))
        _wh = tf.reshape(wh_out[0], (-1, wh_out.shape[-1]))

        _scores, _inds = tf.math.top_k(_hm, k=self.k, sorted=True)
        _classes = tf.cast(_inds % cat, tf.float32)
        _inds = tf.cast(_inds / cat, tf.int32)
        _xs = tf.cast(_inds % width, tf.float32)
        _ys = tf.cast(tf.floor(_inds / width), tf.float32)  # xy format: Image
        _wh = tf.gather(_wh, _inds) / 2  # xy format: Image (eg. width-first)
        _reg = tf.gather(_reg, _inds)  # xy format: Image

        _xs = _xs + _reg[..., 0]
        _ys = _ys + _reg[..., 1]

        _x1 = _xs - _wh[..., 0]
        _y1 = _ys - _wh[..., 1]
        _x2 = _xs + _wh[..., 0]
        _y2 = _ys + _wh[..., 1]

        scx = hm_out.shape[3]
        scy = hm_out.shape[2]  # note: transpose

        # rescale to 0-1
        _x1 = _x1 / scx
        _y1 = _y1 / scy
        _x2 = _x2 / scx
        _y2 = _y2 / scy

        boxes = tf.stack([_x1, _y1, _x2, _y2], axis=-1)

        return {"boxes": boxes, "types": _classes, "scores": _scores}
