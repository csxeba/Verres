import os

import tensorflow as tf

from verres.data import cocodoom
from verres.tf_arch.panoptic import Segmentor
from verres.artifactory import Artifactory

EPOCHS = 30
BATCH_SIZE = 2
VIF = 4

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

val_stream = cocodoom.COCODoomSequence(
    stream_config=cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.PANSEG,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                min_no_visible_objects=0),
    data_loader=cocodoom.COCODoomLoader(
        config=cocodoom.COCODoomLoaderConfig(
            data_json="/data/Datasets/cocodoom/map-val.json",
            images_root="/data/Datasets/cocodoom",
            stride=8,
        )
    )
)
val_ds = tf.data.Dataset.from_generator(lambda: val_stream,
                                        output_types=output_types,
                                        output_shapes=output_shapes)

artifactory = Artifactory.get_default(experiment_name="panseg")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(os.path.join(artifactory.tensorboard, "latest.h5"),
                                       save_freq=1, save_weights_only=True),
    tf.keras.callbacks.TensorBoard(artifactory.tensorboard),
    tf.keras.callbacks.CSVLogger(artifactory.logfile_path, append=True)
]

model = Segmentor(num_classes=loader.num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(2e-5 * 64 * 4))

model.fit(train_ds.prefetch(10),
          epochs=EPOCHS * VIF,
          steps_per_epoch=stream.steps_per_epoch() // VIF,
          validation_data=val_ds.prefetch(10),
          validation_steps=val_stream.steps_per_epoch())
