from typing import List

import tensorflow as tf
from tensorflow.keras import layers as tfl

from verres.layers import block
from verres.operation import losses as vrsloss
from verres.utils import layer_utils


class StageBody(tf.keras.Model):

    def __init__(self, width, num_blocks=5, skip_connect=True):
        super().__init__()
        self.layer_objects = [block.VRSConvBlock(width, depth=3, skip_connect=True, batch_normalize=True,
                                                 activation="leakyrelu") for _ in range(num_blocks)]
        self.skip_connect = skip_connect

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layer_objects:
            x = layer(x, training=training, mask=mask)
        if self.skip_connect:
            x = tf.concat([inputs, x], axis=3)
        return x


class Head(tf.keras.Model):

    def __init__(self, width: int, num_outputs: int, activation: str = "leakyrelu"):
        super().__init__()
        self.conv = tfl.Conv2D(width, kernel_size=3, padding="same")
        self.act = layer_utils.get_activation(activation, as_layer=True)
        self.out = tfl.Conv2D(num_outputs, kernel_size=1)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.act(x)
        return self.out(x)


class Detector(tf.keras.Model):

    def __init__(self, num_classes: int):
        super().__init__()
        self.body8 = StageBody(width=64, num_blocks=5, skip_connect=True)
        self.body4 = StageBody(width=32, num_blocks=1, skip_connect=True)
        self.body2 = StageBody(width=16, num_blocks=1, skip_connect=True)
        self.upsc8_4 = block.VRSUpscale(num_stages=1, width_base=64, batch_normalize=True, activation="leakyrelu")
        self.upsc4_2 = block.VRSUpscale(num_stages=1, width_base=32, batch_normalize=True, activation="leakyrelu")
        self.upsc2_1 = block.VRSUpscale(num_stages=1, width_base=16, batch_normalize=True, activation="leakyrelu")
        self.hmap = Head(width=64, num_outputs=num_classes)
        self.rreg = Head(width=64, num_outputs=num_classes*2)
        self.sseg = Head(width=8, num_outputs=num_classes+1)
        self.iseg = Head(width=8, num_outputs=2)

    @tf.function
    def call(self, inputs, training=None, mask=None):

        ftr1, ftr2, ftr4, ftr8 = inputs

        ftr8 = self.body8(ftr8, training=training, mask=mask)

        ftr4 = tf.concat([self.upsc8_4(ftr8, training=training, mask=mask), ftr4], axis=-1)
        ftr4 = self.body4(ftr4, training=training, mask=mask)

        ftr2 = tf.concat([self.upsc4_2(ftr4, training=training, mask=mask), ftr2], axis=-1)
        ftr2 = self.body2(ftr2, training=training, mask=mask)

        ftr1 = tf.concat([self.upsc2_1(ftr2, training=training, mask=mask), ftr1], axis=-1)

        hmap = self.hmap(ftr8, training=training, mask=mask)
        rreg = self.rreg(ftr8, training=training, mask=mask)
        iseg = self.iseg(ftr1, training=training, mask=mask)
        sseg = self.sseg(ftr1, training=training, mask=mask)

        return hmap, rreg, iseg, sseg


class Segmentor(tf.keras.Model):

    def __init__(self,
                 num_classes: int,
                 backbone: tf.keras.Model,
                 feature_layer_names: List[str]):

        super().__init__()
        self.backbone = self._wrap_backbone(backbone, feature_layer_names)
        self.detector = Detector(num_classes)
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

            hmap_loss = vrsloss.mse(hmap_gt, hmap)
            rreg_loss = vrsloss.mae(rreg_gt, rreg * rreg_mask)
            iseg_loss = vrsloss.mae(iseg_gt, iseg * iseg_mask)
            sseg_loss = vrsloss.mean_of_cxent_sparse_from_logits(sseg_gt, sseg)
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
        hmap_loss = vrsloss.mse(hmap_gt, hmap)
        rreg_loss = vrsloss.mae(rreg_gt, rreg * rreg_mask)
        iseg_loss = vrsloss.mae(iseg_gt, iseg * iseg_mask)
        sseg_loss = vrsloss.mean_of_cxent_sparse_from_logits(sseg_gt, sseg)

        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(sseg_gt, sseg))

        return {"HMap/val": hmap_loss, "RReg/val": rreg_loss, "ISeg/val": iseg_loss,
                "SSeg/val": sseg_loss, "Acc/val": acc}
