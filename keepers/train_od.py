import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import callbacks as tfcb

from verres.data import cocodoom
from verres.tf_arch import backbone as vrsbackbone, vision
from verres.artifactory import Artifactory
from verres.utils import keras_callbacks as vcb, cocodoom_utils

cocodoom_utils.generate_enemy_dataset()

EPOCHS = 120
BATCH_SIZE = 8
VIF = 2

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/map-train.json",
        images_root="/data/Datasets/cocodoom",
        stride=8,
        input_shape=None
    )
)
streamcfg = cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.DETECTION,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          min_no_visible_objects=2)
stream = cocodoom.COCODoomSequence(streamcfg, loader)

output_types = (tf.float32, tf.float32, tf.int64, tf.float32, tf.float32),
output_shapes = ((None, 200, 320, 3),
                 (None, 25, 40, loader.num_classes),
                 (None, 4),
                 (None, 2),
                 (None, 2)),

dataset = tf.data.Dataset.from_generator(
    lambda: stream,
    output_types=output_types,
    output_shapes=output_shapes
)

val_loader = cocodoom.COCODoomLoader(
    config=cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/enemy-map-val.json",
        images_root="/data/Datasets/cocodoom",
        stride=8,
        input_shape=None))

# artifactory = Artifactory(root="/drive/My Drive/artifactory", experiment_name="od", add_now=False)
artifactory = Artifactory.get_default(experiment_name="od", add_now=False)
latest_checkpoint = artifactory.root / "latest.h5"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(str(latest_checkpoint), save_weights_only=True),
    vcb.ObjectMAP(val_loader, artifactory, checkpoint_best=True),
    tf.keras.callbacks.TensorBoard(artifactory.tensorboard, profile_batch=0)]

backbone = vrsbackbone.SmallFCNN(width_base=8)

model = vision.ObjectDetector(num_classes=loader.num_classes,
                              backbone=backbone,
                              stride=8)
# keras_utils.inject_regularizer(model, kernel_regularizer=tf.keras.regularizers.l2(5e-4))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
model.train_step(next(stream))

callbacks = tfcb.CallbackList(callbacks, add_history=True, add_progbar=False, model=model)
iterator = iter(dataset.prefetch(10))
data_times = []
model_times = []
steps = stream.steps_per_epoch() // VIF

for epoch in range(1, EPOCHS+1):

    for i in range(1, steps + 1):

        start = time.time()
        data = next(iterator)
        data_times.append(time.time() - start)

        start = time.time()
        logs = model.train_step(data)
        model_times.append(time.time() - start)

        print(f"\rEpoch {epoch:3>} - P: {i / steps:>7.2%} "
              f"L: {logs['loss/train'].numpy():.2f} "
              f"DTime: {np.mean(data_times[-10:]):>6.4f} s "
              f"MTime: {np.mean(model_times[-10:]):>6.4f} s",
              end="")

    print()


model.fit(dataset.prefetch(10),
          epochs=EPOCHS * VIF,
          steps_per_epoch=stream.steps_per_epoch() // VIF,
          callbacks=callbacks)
