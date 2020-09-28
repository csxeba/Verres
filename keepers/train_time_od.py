import tensorflow as tf
from tensorflow.keras import callbacks as kcallbacks

import verres as vrs
from verres.data import cocodoom
from verres.tf_arch import backbone as vrsbackbone, vision
from verres.utils import keras_callbacks as vrscallbacks

STRIDE = 8
BATCH_SIZE = 8
VIF = 4

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/enemy-time-map-full-train.json",
        images_root="/data/Datasets/cocodoom",
        stride=STRIDE))
val_loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/enemy-time-map-full-val-ds4.json",
        images_root="/data/Datasets/cocodoom",
        stride=STRIDE))

stream = cocodoom.COCODoomTimeSequence(
    cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.DETECTION,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  min_no_visible_objects=2),
    time_data_loader=loader,
)

output_types = (tf.float32, tf.float32, tf.int64, tf.float32, tf.float32,
                tf.float32, tf.float32, tf.int64, tf.float32, tf.float32,),
output_shapes = ((None, 200, 320, 3),
                 (None, 25, 40, loader.num_classes),
                 (None, 4),
                 (None, 2),
                 (None, 2),
                 (None, 200, 320, 3),
                 (None, 25, 40, loader.num_classes),
                 (None, 4),
                 (None, 2),
                 (None, 2)),

dataset = tf.data.Dataset.from_generator(
    lambda: stream,
    output_types=output_types,
    output_shapes=output_shapes)

backbone = vrsbackbone.SmallFCNN(width_base=16, strides=(2, 4, 8))
fusion = vrsbackbone.FeatureFuser(backbone, final_stride=8, base_width=8, final_width=64)

model = vision.TimePriorizedObjectDetector(num_classes=loader.num_classes,
                                           backbone=fusion,
                                           stride=STRIDE,
                                           refinement_stages=1)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
model.train_step(next(stream))

artifactory = vrs.artifactory.Artifactory.get_default(
    experiment_name="time_od", add_now=True
)

callbacks = [
    kcallbacks.TensorBoard(artifactory.tensorboard, profile_batch=0, write_graph=False),
    kcallbacks.CSVLogger(artifactory.logfile_path),
    kcallbacks.ModelCheckpoint(artifactory.make_checkpoint_template().format("latest"), save_weights_only=True)]

model.fit(dataset.prefetch(10),
          epochs=120 * VIF,
          steps_per_epoch=stream.steps_per_epoch() // VIF,
          callbacks=callbacks)

