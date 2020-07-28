import os

import numpy as np
import tensorflow as tf

from verres.data import cocodoom
from verres.tf_arch.panoptic import Segmentor
from verres.artifactory import Artifactory


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
