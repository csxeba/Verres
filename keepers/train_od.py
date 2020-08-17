import os

import tensorflow as tf

from verres.data import cocodoom
from verres.tf_arch.backbone import ApplicationBackbone, FeatureSpec
from verres.tf_arch.vision import ObjectDetector
from verres.artifactory import Artifactory

EPOCHS = 120
BATCH_SIZE = 12
VIF = 4
BACKBONE = "MobileNet"
FEATURE_LAYER_NAMES = ["conv_pw_5_relu"]
FEATURE_STRIDES = [8]
feature_specs = [FeatureSpec(name, stride) for name, stride in zip(FEATURE_LAYER_NAMES, FEATURE_STRIDES)]

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/map-train.json",
        images_root="/data/Datasets/cocodoom",
        stride=FEATURE_STRIDES[-1]
    )
)
streamcfg = cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.DETECTION,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          min_no_visible_objects=2)
stream = cocodoom.COCODoomSequence(streamcfg, loader)

val_stream = cocodoom.COCODoomSequence(
    stream_config=cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.DETECTION,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                min_no_visible_objects=0),
    data_loader=cocodoom.COCODoomLoader(
        config=cocodoom.COCODoomLoaderConfig(
            data_json="/data/Datasets/cocodoom/map-val.json",
            images_root="/data/Datasets/cocodoom",
            stride=8,
        )
    )
)

artifactory = Artifactory.get_default(experiment_name="panseg")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(os.path.join(artifactory.checkpoints, "latest.h5"),
                                       save_freq=1, save_weights_only=True),
    tf.keras.callbacks.TensorBoard(artifactory.tensorboard, profile_batch=0)
]

backbone = ApplicationBackbone(BACKBONE,
                               feature_specs=feature_specs,
                               input_shape=(200, 320, 3),
                               fixed_batch_size=BATCH_SIZE)
model = ObjectDetector(num_classes=loader.num_classes,
                       backbone=backbone,
                       stride=FEATURE_STRIDES[-1])

# keras_utils.inject_regularizer(model, kernel_regularizer=tf.keras.regularizers.l2(5e-4))
model.compile(optimizer=tf.keras.optimizers.Adam(2e-4))

# for data in stream:
#     model.train_step(data)

model.fit(stream,
          epochs=EPOCHS * VIF,
          steps_per_epoch=stream.steps_per_epoch() // VIF,
          validation_data=val_stream,
          validation_steps=val_stream.steps_per_epoch() // VIF,
          callbacks=callbacks)
