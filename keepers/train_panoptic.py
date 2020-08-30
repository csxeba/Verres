import os

import tensorflow as tf

from verres.data import cocodoom
from verres.tf_arch import backbone, vision
from verres.artifactory import Artifactory

EPOCHS = 120
BATCH_SIZE = 10
VIF = 4
BACKBONE = "MobileNet"
FEATURE_STRIDES = [1, 2, 4, 8]

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

output_types = tuple(tf.float32 for _ in range(5)),
output_shapes = (
    (BATCH_SIZE, 200, 320, 3),
    (BATCH_SIZE, 25, 40, loader.num_classes),
    (BATCH_SIZE, 25, 40, loader.num_classes*2),
    (BATCH_SIZE, 200, 320, 2),
    (BATCH_SIZE, 200, 320, 1)
),

train_ds = tf.data.Dataset.from_generator(lambda: stream,
                                          output_types=output_types,
                                          output_shapes=output_shapes)

artifactory = Artifactory.get_default(experiment_name="panseg")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(os.path.join(artifactory.checkpoints, "latest.h5"),
                                       save_freq=1, save_weights_only=True),
    tf.keras.callbacks.TensorBoard(artifactory.tensorboard, profile_batch=0)
]

backbone = backbone.SmallFCNN(strides=FEATURE_STRIDES, width_base=16)
model = vision.PanopticSegmentor(
    num_classes=loader.num_classes,
    backbone=backbone)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

model.fit(train_ds.prefetch(10),
          epochs=EPOCHS * VIF,
          steps_per_epoch=stream.steps_per_epoch() // VIF,
          callbacks=callbacks)
