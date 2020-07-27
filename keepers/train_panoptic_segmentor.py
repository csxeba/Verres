import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfl

from verres.data import cocodoom
from verres.layers import block
from verres.utils import layer_utils
from verres.operation import losses as vrsloss
from verres.artifactory import Artifactory


class StageBody(tf.keras.Model):

    def __init__(self, width, num_blocks=5, skip_connect=True):
        super().__init__()
        self.layer_objects = [block.VRSConvBlock(width, depth=3, skip_connect=True, batch_normalize=True,
                                                 activation="leakyrelu") for _ in range(num_blocks)]
        self.skip_connect = skip_connect

    @tf.function(experimental_relax_shapes=True)
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
        self.conv = tfl.Conv2D(width, kernel_size=1)
        self.act = layer_utils.get_activation(activation, as_layer=True)
        self.out = tfl.Conv2D(num_outputs, kernel_size=1)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.act(x)
        return self.out(x)


class Segmentor(tf.keras.Model):

    INPUT_SHAPE = (200, 320, 3)
    RESNET_FTR_STRIDES = [1, 2, 4, 8]
    RESNET_FTR_LAYER_NAMES = reversed(["input_1", "conv1_relu", "conv2_block3_out", "conv3_block4_out"])

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = self._make_backbone()
        self.body8 = StageBody(width=64, num_blocks=5, skip_connect=True)
        self.body4 = StageBody(width=32, num_blocks=5, skip_connect=True)
        self.body2 = StageBody(width=16, num_blocks=5, skip_connect=True)
        self.body1 = StageBody(width=8, num_blocks=3, skip_connect=True)
        self.upsc8_4 = block.VRSUpscale(num_stages=1, width_base=64, batch_normalize=True, activation="leakyrelu")
        self.upsc4_2 = block.VRSUpscale(num_stages=1, width_base=32, batch_normalize=True, activation="leakyrelu")
        self.upsc2_1 = block.VRSUpscale(num_stages=1, width_base=16, batch_normalize=True, activation="leakyrelu")
        self.hmap = Head(width=64, num_outputs=num_classes)
        self.rreg = Head(width=64, num_outputs=num_classes*2)
        self.sseg = Head(width=8, num_outputs=num_classes)
        self.iseg = Head(width=8, num_outputs=2)

    def _make_backbone(self):
        resnet = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=self.INPUT_SHAPE)
        output_tensors = [resnet.get_layer(lyr).output for lyr in self.RESNET_FTR_LAYER_NAMES]
        return tf.keras.Model(inputs=resnet.input, outputs=output_tensors)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, mask=None):
        ftr8, ftr4, ftr2, ftr1 = self.backbone(inputs, training=training, mask=mask)
        ftr8 = self.body8(ftr8, training=training, mask=mask)

        ftr4 = tf.concat([self.upsc8_4(ftr8, training=training, mask=mask), ftr4], axis=-1)
        ftr4 = self.body4(ftr4, training=training, mask=mask)

        ftr2 = tf.concat([self.upsc4_2(ftr4, training=training, mask=mask), ftr2], axis=-1)
        ftr2 = self.body2(ftr2, training=training, mask=mask)

        ftr1 = tf.concat([self.upsc2_1(ftr2, training=training, mask=mask), ftr1], axis=-1)
        ftr1 = self.body1(ftr1, training=training, mask=mask)

        hmap = self.hmap(ftr8, training=training, mask=mask)
        rreg = self.rreg(ftr8, training=training, mask=mask)
        iseg = self.iseg(ftr1, training=training, mask=mask)
        sseg = self.sseg(ftr1, training=training, mask=mask)

        return hmap, rreg, iseg, sseg

    @tf.function
    def train_step(self, data):
        img, hmap_gt, rreg_gt, iseg_gt, sseg_gt = data[0]
        rreg_mask = tf.cast(rreg_gt > 0, tf.float32)
        iseg_mask = tf.cast(iseg_gt > 0, tf.float32)

        with tf.GradientTape() as tape:
            hmap, rreg, iseg, sseg = self(img)

            hmap_loss = vrsloss.sse(hmap_gt, hmap)
            rreg_loss = vrsloss.sae(rreg_gt, rreg * rreg_mask)
            iseg_loss = vrsloss.sae(iseg_gt, iseg * iseg_mask)
            sseg_loss = vrsloss.sum_of_cxent_sparse_from_logits(sseg_gt, sseg)

            total_loss = hmap_loss * 64 + rreg_loss * 64 + iseg_loss + sseg_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(sseg_gt, sseg))

        return {"HMap": hmap_loss, "RReg": rreg_loss, "ISeg": iseg_loss, "SSeg": sseg_loss, "Acc": acc}


def log_loss(epoch_no, step_no, total_steps, losses_dict):
    logstr = [f"E {epoch_no:>4} - {step_no / total_steps:>7.4%}"]
    for key, val in losses_dict.items():
        logstr.append(f"{key}: {np.mean(val):.4f}")
    print("\r",  " - ".join(logstr), end="")


EPOCHS = 30
BATCH_SIZE = 2

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/map-train.json",
        images_root="/data/Datasets/cocodoom",
        stride=8
    )
)
streamcfg = cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.PANSEG,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          min_no_visible_objects=2)
stream = cocodoom.COCODoomSequence(streamcfg, loader)

artifactory = Artifactory.get_default("panseg")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(os.path.join(artifactory.tensorboard, "latest.h5"),
                                       save_freq=1, save_weights_only=True),
    tf.keras.callbacks.TensorBoard(artifactory.tensorboard),
    tf.keras.callbacks.CSVLogger(artifactory.logfile_path, append=True)
]

model = Segmentor(num_classes=loader.num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(2e-5))
model.fit(stream,
          epochs=EPOCHS,
          steps_per_epoch=stream.steps_per_epoch())
