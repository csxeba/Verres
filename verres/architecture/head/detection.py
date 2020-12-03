import tensorflow as tf

from verres.architecture import backbone as _backbone, detector
from verres.operation import losses as L, tensor_ops as T


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

        logs = {"loss": total_loss, "hmap": hmap_loss, "rreg": rreg_loss, "boxx": boxx_loss}

        return logs


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
        no_prior = [tf.random.uniform(s, minval=0, maxval=1e-2, dtype=tf.float32),
                    tf.random.uniform((s[0], s[1], s[2], s[3]*2), minval=0, maxval=1e-2, dtype=tf.float32)]
        return no_prior

    def detect(self, inputs):
        past_image, present_image = inputs
        past_image = past_image[None, ...]
        present_image = present_image[None, ...]
        no_prior = self._generate_empty_priors(tf.shape(past_image))
        hmap, rreg, boxx = self([past_image, no_prior])

        prior = [hmap, boxx]
        outputs = self([present_image, prior])

        result = super().postprocess(outputs)
        return result

    @tf.function(experimental_relax_shapes=True)
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