import tensorflow as tf

from verres.architecture import backbone as _backbone, head as _head
from verres.operation import numeric as T
from verres.optim import losses as L


class PanopticSegmentor(tf.keras.Model):
    # coord[valid], affiliations[valid], centroids, types, scores

    _EMPTY = (
        tf.zeros((0, 2), dtype=tf.int64),
        tf.zeros((0,), dtype=tf.int64),
        tf.zeros((0, 2), dtype=tf.float32),
        tf.zeros((0,), dtype=tf.int64),
        tf.zeros((0,), dtype=tf.float32)
    )

    def __init__(self,
                 num_classes: int,
                 backbone: _backbone.VRSBackbone,
                 peak_nms: float = 0.3,
                 offset_nms: float = 5.,
                 weights: str = None,
                 sparse_detection: bool = True):

        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.peak_nms = peak_nms
        self.offset_nms = offset_nms ** 2
        self.sparse_detection = sparse_detection
        self.detector = _head.Panoptic(num_classes)
        self.loss_tracker = L.Tracker(
            ["loss/train", "HMap/train", "RReg/train", "ISeg/train", "SSeg/train", "Acc/train"])

        if weights is not None:
            s = max(fs.working_stride for fs in self.backbone.feature_specs)
            self(tf.zeros((1, s, s, 3)))
            self.load_weights(weights)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        ftr1, ftr2, ftr4, ftr8 = self.backbone(inputs, training=training, mask=mask)
        hmap, rreg, iseg, sseg = self.detector([ftr1, ftr2, ftr4, ftr8])
        return hmap, rreg, iseg, sseg

    def get_centroids(self, hmap, rreg):
        peaks, scores = T.peakfind(hmap, peak_nms=self.peak_nms)
        centroids, types = T.gather_and_refine(peaks, rreg[0])
        centroids = centroids[:, ::-1] * 8
        return centroids, types, scores

    @staticmethod
    def get_offsetted_coords(sseg, iseg):
        hard_sseg = tf.argmax(sseg[0], axis=-1)
        coord = T.meshgrid(hard_sseg.shape[:2], dtype=tf.float32)
        non_bg = hard_sseg > 0
        coords_non_bg = coord[non_bg]
        iseg_offset = iseg[0][non_bg][:, ::-1] + coords_non_bg
        return coords_non_bg, iseg_offset

    @staticmethod
    def get_affiliations(iseg_offset, centroids):
        D = tf.reduce_sum(  # [M, 1, 2] - [N, 1, 2] -> [M, N, 2]
            tf.square(iseg_offset[:, None, :] - centroids[None, :, :]), axis=2)  # -> [M, N]
        affiliations = tf.argmin(D, axis=1)
        offset_scores = tf.reduce_min(D, axis=1)
        return affiliations, offset_scores

    def get_filtered_result(self, coords_non_bg, affiliations, offset_scores):
        valid = offset_scores < self.offset_nms
        return coords_non_bg[valid], affiliations[valid]

    def postprocess(self, hmap, rreg, iseg, sseg):
        centroids, types, scores = self.get_centroids(hmap, rreg)

        if centroids.shape[0] == 0:
            return self._EMPTY

        coords_non_bg, iseg_offset = self.get_offsetted_coords(sseg, iseg)
        if iseg_offset.shape[0] == 0:
            return self._EMPTY

        affiliations, offset_scores = self.get_affiliations(iseg_offset, centroids)
        coords_non_bg, affiliations = self.get_filtered_result(coords_non_bg, affiliations, offset_scores)

        # masks = self.scatter_result(coords_non_bg, affiliations, iseg[0].shape[:2])
        return coords_non_bg, affiliations, centroids, types, scores

    def detect(self, inputs):
        hmap, rreg, iseg, sseg = self(inputs)
        coords, affiliations, centroids, types, scores = self.postprocess(hmap, rreg, iseg, sseg)
        return coords, affiliations, centroids, types, scores

    @tf.function
    def train_step(self, data):
        img, hmap_gt, locations, rreg_sparse, iseg_gt, sseg_gt = data[0]
        iseg_mask = tf.cast(iseg_gt != 0, tf.float32)
        locations = tf.stack([locations[:, 0], locations[:, 2], locations[:, 1], locations[:, 3]],
                             axis=1)

        with tf.GradientTape() as tape:
            hmap, rreg, iseg, sseg = self(img)

            hmap_loss = L.sse(hmap_gt, hmap)
            rreg_loss = L.sparse_vector_field_sae(rreg_sparse, rreg, locations)
            iseg_loss = L.mae(iseg_gt, iseg * iseg_mask)
            sseg_loss = L.mean_of_cxent_sparse_from_logits(sseg_gt, sseg)

            total_loss = hmap_loss + rreg_loss + iseg_loss + sseg_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(sseg_gt, sseg))

        return self.loss_tracker.record([total_loss, hmap_loss, rreg_loss, iseg_loss, sseg_loss, acc])
