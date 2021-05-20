import tensorflow as tf

import verres as V
from verres.operation import numeric as T
from ..layers import block
from .base import VRSHead


class OD(VRSHead):

    def __init__(self, config: V.Config):
        super().__init__()
        spec = config.model.head_spec.copy()
        self.hmap_head = block.VRSHead(spec.get("head_convolution_width", 32), output_width=spec["num_classes"])
        self.rreg_head = block.VRSHead(spec.get("head_convolution_width", 32), output_width=spec["num_classes"]*2)
        self.boxx_head = block.VRSHead(spec.get("head_convolution_width", 32), output_width=spec["num_classes"]*2)
        self.peak_nms = spec.get("peak_nms", 0.1)

    def call(self, inputs, training=None, mask=None):
        centroid_features, box_features = inputs
        hmap = self.hmap_head(centroid_features)
        rreg = self.rreg_head(centroid_features)
        x = tf.concat([box_features, hmap, rreg], axis=-1)
        boxx = self.boxx_head(x)
        return {"heatmap": hmap, "box_wh": boxx, "refinement": rreg}

    def postprocess_network_output(self, predictions):
        hmap = predictions["heatmap"]
        rreg = predictions["refinement"]
        bbox = predictions["box_wh"]

        peaks, scores = T.peakfind(hmap, self.peak_nms)

        output_shape = tf.cast(tf.shape(hmap)[1:3], tf.float32)

        refinements = tf.stack([
            tf.gather_nd(rreg[0, ..., 1::2], peaks),
            tf.gather_nd(rreg[0, ..., 0::2], peaks)], axis=-1)

        box_params = tf.stack([
            tf.gather_nd(bbox[0, ..., 0], peaks[:, :2]),
            tf.gather_nd(bbox[0, ..., 1], peaks[:, :2])], axis=-1)

        refined_centroids = (tf.cast(peaks[:, :2], tf.float32) + refinements) / output_shape
        box_params = box_params / output_shape
        types = peaks[:, 2]

        boxes = tf.concat([refined_centroids, box_params], axis=1)

        result = {"boxes": boxes, "types": types, "scores": scores}

        return result
