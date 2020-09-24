import tensorflow as tf

from . import detector, backbone as _backbone
from ..operation import losses as L, tensor_ops as T


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
        self.detector = detector.Panoptic(num_classes)
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


class ObjectDetector(tf.keras.Model):

    def __init__(self,
                 backbone: _backbone.VRSBackbone,
                 num_classes: int,
                 stride: int,
                 refinement_stages: int = 1,
                 weights: str = None,
                 peak_nms: float = 0.1):

        super().__init__()
        self.backbone = backbone
        if len(backbone.feature_specs) != 2:
            print(" [Verres.OD] - Single backbone mode is active.")
            self.single_backbone_mode = True
        else:
            self.single_backbone_mode = False
        self.detectors = [detector.OD(num_classes, stride) for _ in range(refinement_stages)]
        self.loss_tracker = L.Tracker(["loss/train", "HMap/train", "RReg/train", "BBox/train"])
        self.peak_nms = peak_nms
        self.stride = stride
        self.num_classes = num_classes
        if weights is not None:
            self.build((None, None, None, 3))
            self.load_weights(weights)

    def call(self, inputs, training=None, mask=None):
        outputs = []
        features = self.backbone(inputs)
        if self.single_backbone_mode:
            features = [features[0], features[0]]

        hmap_features, boxx_features = features

        for det in self.detectors:
            hmap, rreg, boxx = det(features)
            outputs.extend([hmap, rreg, boxx])
            features = [tf.concat([hmap_features, hmap], axis=-1),
                        tf.concat([boxx_features, boxx], axis=-1)]
        return outputs

    def postprocess(self, outputs):
        hmap, rreg, bbox = outputs[-3:]
        peaks, scores = T.peakfind(hmap, self.peak_nms)

        refinements = tf.stack([
            tf.gather_nd(rreg[0, ..., 1::2], peaks),
            tf.gather_nd(rreg[0, ..., 0::2], peaks)], axis=-1)

        box_params = tf.stack([
            tf.gather_nd(bbox[0, ..., 0], peaks[:, :2]),
            tf.gather_nd(bbox[0, ..., 1], peaks[:, :2])], axis=-1)

        refined_centroids = (tf.cast(peaks[:, :2], tf.float32) + refinements) * self.stride
        centroids = tf.cast(tf.round(refined_centroids[:, ::-1]), tf.int64)
        box_params = box_params * self.stride
        types = peaks[:, 2]

        return centroids, box_params, types, scores

    @tf.function
    def detect(self, inputs):
        outputs = self(inputs, training=False)
        centroids, whs, types, scores = self.postprocess(outputs)
        return centroids, whs, types, scores

    @tf.function
    def train_step(self, data):
        image, hmap_gt, locations, rreg_values, boxx_values = data[0]
        locations = tf.concat([locations[:, 0:1], locations[:, 2:3], locations[:, 1:2], locations[:, 3:4]], axis=1)
        with tf.GradientTape() as tape:
            outputs = self(image)
            for i in range(0, len(self.detectors)*3, 3):
                hmap, rreg, boxx = outputs[i:i+3]

                hmap_loss = L.sse(hmap_gt, hmap)
                rreg_loss = L.sparse_vector_field_sae(rreg_values, rreg, locations)
                boxx_loss = L.sparse_vector_field_sae(boxx_values, boxx, locations)

                total_loss = hmap_loss + rreg_loss * 10 + boxx_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return total_loss, hmap_loss, rreg_loss, boxx_loss


class TimePriorizedObjectDetector(ObjectDetector):

    def __init__(self,
                 backbone: _backbone.VRSBackbone,
                 num_classes: int,
                 stride: int,
                 refinement_stages: int = 1,
                 weights: str = None,
                 peak_nms: float = 0.1):

        super().__init__(backbone, num_classes, stride, refinement_stages, weights, peak_nms)
        self.loss_tracker = L.Tracker(["loss", "noprio_loss", "prio_loss"])

    def call(self, inputs, training=None, mask=None):
        images, priors = inputs
        outputs = []
        features = self.backbone(images)
        if self.single_backbone_mode:
            features = [features[0], features[0]]
        hmap_features, boxx_features = features
        hmap_features = tf.concat([hmap_features, priors[0]], axis=3)
        boxx_features = tf.concat([hmap_features, priors[1]], axis=3)
        for head in self.detectors:
            hmap, rreg, boxx = head(features)
            outputs.extend([hmap, rreg, boxx])
            features = [tf.concat([hmap_features, hmap], axis=-1),
                        tf.concat([boxx_features, boxx], axis=-1)]
        return outputs

    def _generate_empty_priors(self, image_shape):
        s = image_shape[0], image_shape[1] // self.stride, image_shape[2] // self.stride, self.num_classes
        no_prior = [tf.stop_gradient(tf.zeros(s, dtype=tf.float32)),
                    tf.stop_gradient(tf.zeros((s[0], s[1], s[2], s[3]*2), dtype=tf.float32))]
        return no_prior

    def detect(self, inputs):
        past_image, present_image = inputs
        no_prior = self._generate_empty_priors(tf.shape(past_image))
        hmap, rreg, boxx = self([past_image, no_prior])

        prior = [hmap, boxx]
        outputs = self([present_image, prior])

        result = super().postprocess(outputs)
        return result

    @tf.function
    def train_step(self, data):
        (image1, hmap_gt1, locations1, rreg_values1, boxx_values1,
         image2, hmap_gt2, locations2, rreg_values2, boxx_values2) = data[0]

        locations1 = tf.stop_gradient(tf.stack(
            [locations1[:, 0], locations1[:, 2], locations1[:, 1], locations1[:, 3]], axis=1))
        locations2 = tf.stop_gradient(tf.stack(
            [locations2[:, 0], locations2[:, 2], locations2[:, 1], locations2[:, 3]], axis=1))

        no_prior = self._generate_empty_priors(tf.shape(image1))

        with tf.GradientTape() as tape:

            hmap1, rreg1, boxx1 = self([image1, no_prior])
            hmap2, rreg2, boxx2 = self([image2, no_prior])

            hmap_loss = L.sse(hmap_gt1, hmap1) + L.sse(hmap_gt2, hmap2)
            rreg_loss = (L.sparse_vector_field_sae(rreg_values1, rreg1, locations1) +
                         L.sparse_vector_field_sae(rreg_values2, rreg2, locations2))
            boxx_loss = (L.sparse_vector_field_sae(boxx_values1, boxx1, locations1) +
                         L.sparse_vector_field_sae(boxx_values2, boxx2, locations2))

            priors = [hmap1, boxx1]
            hmapp, rregp, boxxp = self([image2, priors])

            phmap_loss = L.sse(hmap_gt2, hmapp)
            prreg_loss = L.sparse_vector_field_sae(rreg_values2, rregp, locations2)
            pboxx_loss = L.sparse_vector_field_sae(boxx_values2, boxxp, locations2)

            no_prior_loss = (hmap_loss + rreg_loss * 10. + boxx_loss) / 2.
            with_prior_loss = phmap_loss + prreg_loss * 10. + pboxx_loss

            total_loss = no_prior_loss + with_prior_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return self.loss_tracker.record([total_loss, no_prior_loss, with_prior_loss])
