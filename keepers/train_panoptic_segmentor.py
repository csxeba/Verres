import os

import tensorflow as tf

from verres.data import cocodoom
from verres.tf_arch.panoptic import Segmentor
from verres.artifactory import Artifactory

EPOCHS = 30
BATCH_SIZE = 10

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

val_stream = cocodoom.COCODoomSequence(
    stream_config=cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.PANSEG,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                min_no_visible_objects=0),
    data_loader=cocodoom.COCODoomLoader(
        config=cocodoom.COCODoomLoaderConfig(
            data_json="/data/Datasets/cocodoom/map-val.json",
            images_root="/data/Datasets/cocodoom",
            stride=8
        )
    )
)

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
          steps_per_epoch=10,
          validation_data=val_stream,
          validation_steps=10)
