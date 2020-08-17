from typing import List

import tensorflow as tf

from . import detector
from ..operation import losses as L


class PanopticSegmentor(tf.keras.Model):

    def __init__(self,
                 num_classes: int,
                 backbone: tf.keras.Model,
                 feature_layer_names: List[str]):

        super().__init__()
        self.backbone = self._wrap_backbone(backbone, feature_layer_names)
        self.detector = detector.Panoptic(num_classes)
        self.train_steps = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.train_metric_keys = ["loss/train", "HMap/train", "RReg/train", "ISeg/train", "SSeg/train", "Acc/train"]
        self.train_metrics = {n: tf.Variable(0, dtype=tf.float32, trainable=False) for n in self.train_metric_keys}

    def _wrap_backbone(self, base_model, feature_layer_names):
        output_tensors = [base_model.get_layer(layer).output for layer in feature_layer_names]
        return tf.keras.Model(inputs=base_model.input, outputs=output_tensors)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        ftr1, ftr2, ftr4, ftr8 = self.backbone(inputs, training=training, mask=mask)
        hmap, rreg, iseg, sseg = self.detector([ftr1, ftr2, ftr4, ftr8])
        return hmap, rreg, iseg, sseg

    def _save_and_report_losses(self, total_loss, hmap_loss, rreg_loss, iseg_loss, sseg_loss, acc):
        self.train_metrics["loss/train"].assign_add(total_loss)
        self.train_metrics["HMap/train"].assign_add(hmap_loss)
        self.train_metrics["RReg/train"].assign_add(rreg_loss)
        self.train_metrics["ISeg/train"].assign_add(iseg_loss)
        self.train_metrics["SSeg/train"].assign_add(sseg_loss)
        self.train_metrics["Acc/train"].assign_add(acc)
        self.train_steps.assign_add(1)
        return {n: self.train_metrics[n] / self.train_steps for n in self.train_metric_keys}

    @tf.function
    def train_step(self, data):
        img, hmap_gt, rreg_gt, iseg_gt, sseg_gt = data[0]
        rreg_mask = tf.cast(rreg_gt > 0, tf.float32)
        iseg_mask = tf.cast(iseg_gt > 0, tf.float32)

        with tf.GradientTape() as tape:
            hmap, rreg, iseg, sseg = self(img)

            hmap_loss = L.mse(hmap_gt, hmap)
            rreg_loss = L.mae(rreg_gt, rreg * rreg_mask)
            iseg_loss = L.mae(iseg_gt, iseg * iseg_mask)
            sseg_loss = L.mean_of_cxent_sparse_from_logits(sseg_gt, sseg)
            l2 = tf.reduce_sum(self.losses)

            total_loss = hmap_loss + rreg_loss + iseg_loss + sseg_loss + l2

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(sseg_gt, sseg))

        return self._save_and_report_losses(total_loss, hmap_loss, rreg_loss, iseg_loss, sseg_loss, acc)

    @tf.function
    def test_step(self, data):
        img, hmap_gt, rreg_gt, iseg_gt, sseg_gt = data[0]
        rreg_mask = tf.cast(rreg_gt > 0, tf.float32)
        iseg_mask = tf.cast(iseg_gt > 0, tf.float32)

        hmap, rreg, iseg, sseg = self(img)
        hmap_loss = L.mse(hmap_gt, hmap)
        rreg_loss = L.mae(rreg_gt, rreg * rreg_mask)
        iseg_loss = L.mae(iseg_gt, iseg * iseg_mask)
        sseg_loss = L.mean_of_cxent_sparse_from_logits(sseg_gt, sseg)

        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(sseg_gt, sseg))

        return {"HMap/val": hmap_loss, "RReg/val": rreg_loss, "ISeg/val": iseg_loss,
                "SSeg/val": sseg_loss, "Acc/val": acc}


class ObjectDetector(tf.keras.Model):

    def __init__(self,
                 backbone: tf.keras.Model,
                 num_classes: int,
                 stride: int):

        super().__init__()
        self.backbone = backbone
        self.detector = detector.OD(num_classes, stride)
        self.train_steps = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.train_metric_keys = ["loss/train", "HMap/train", "RReg/train", "BBox/train"]
        self.train_metrics = {n: tf.Variable(0, dtype=tf.float32, trainable=False) for n in self.train_metric_keys}

    def call(self, inputs, training=None, mask=None):
        features = self.backbone(inputs)
        hmap, rreg, boxx = self.detector(features)
        return hmap, rreg, boxx

    def detect(self, inputs):
        hmap, rreg, bbox = self(inputs)

        peaks, scores = self.peak_finder(hmap)

        refinements = tf.stack([
            tf.gather(rreg[..., ::2], peaks),
            tf.gather(rreg[..., 1::2], peaks)], axis=-1)

        box_params = tf.stack([
            tf.gather(bbox[..., 0], bbox[:, :3]),
            tf.gather(bbox[..., 1], bbox[:, :3])], axis=-1)

        refined_centroids = (peaks + refinements) * self.stride
        refined_box_params = box_params * self.stride

        return refined_centroids, refined_box_params, scores

    def _save_and_report_losses(self, total_loss, hmap_loss, rreg_loss, bbox_loss):
        self.train_metrics["loss/train"].assign_add(total_loss)
        self.train_metrics["HMap/train"].assign_add(hmap_loss)
        self.train_metrics["RReg/train"].assign_add(rreg_loss)
        self.train_metrics["BBox/train"].assign_add(bbox_loss)
        self.train_steps.assign_add(1)
        return {n: self.train_metrics[n] / self.train_steps for n in self.train_metric_keys}

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data):
        image, hmap_gt, locations, rreg_values, boxx_values = data[0]
        with tf.GradientTape() as tape:
            hmap, rreg, boxx = self(image)

            hmap_loss = L.sse(hmap_gt, hmap)
            rreg_loss = L.sparse_vector_field_sae(rreg_values, rreg, locations)
            boxx_loss = L.sparse_vector_field_sae(boxx_values, boxx, locations)

            total_loss = hmap_loss + rreg_loss * 10 + boxx_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return self._save_and_report_losses(total_loss, hmap_loss, rreg_loss, boxx_loss)

    def test_step(self, data):
        image, hmap_gt, rreg_gt, boxx_gt = data[0]
        hmap, rreg, boxx = self(image)
        hmap_loss = L.sse(hmap_gt, hmap)
        rreg_loss = L.sae(rreg_gt, rreg * tf.cast(rreg_gt > 0, tf.float32))
        bbox_loss = L.sae(boxx_gt, boxx * tf.cast(boxx_gt > 0, tf.float32))

        return {"HMap/val": hmap_loss, "RReg/val": rreg_loss, "BBox/val": bbox_loss}
