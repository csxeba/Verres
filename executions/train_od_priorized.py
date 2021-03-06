import tensorflow as tf

import verres.architecture.head.detection
from verres.data import cocodoom
from verres.architecture import backbone as vrsbackbone
from verres.architecture.head import vision
from verres.artifactory import Artifactory
from verres.utils import keras_callbacks as vcb, cocodoom_utils

cocodoom_utils.generate_enemy_dataset()

EPOCHS = 120
BATCH_SIZE = 8
VIF = 2
STRIDE = 8

base_loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/cocodoom/map-train.json",
        images_root="/data/cocodoom",
        stride=STRIDE,
        input_shape=None
    )
)
full_loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/cocodoom/map-full-train.json",
        images_root="/data/cocodoom",
        stride=STRIDE,
        input_shape=None
    )
)
streamcfg = cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.DETECTION,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          min_no_visible_objects=2)
stream = cocodoom.COCODoomTimeSequence(streamcfg, full_loader)

output_types = tuple([tf.float32,  # image
                      tf.float32,  # heatmap
                      tf.int64,  # location
                      tf.float32,  # rreg
                      tf.float32]*2),  # boxx
output_shapes = tuple([
    [None, None, None, 3],
    [None, None, None, stream.loader.num_classes],
    [None, 4],
    [None, 2],
    [None, 2]]*2),

dataset = tf.data.Dataset.from_generator(
    lambda: stream,
    output_types=output_types,
    output_shapes=output_shapes
)

val_loader = cocodoom.COCODoomLoader(
    config=cocodoom.COCODoomLoaderConfig(
        data_json="/data/cocodoom/enemy-map-val.json",
        images_root="/data/cocodoom",
        stride=STRIDE,
        input_shape=None))

# artifactory = Artifactory(root="/drive/My Drive/artifactory", experiment_name="od", add_now=False)
artifactory = Artifactory.get_default(experiment_name="od_priorized", add_now=True)
latest_checkpoint = artifactory.root / "latest.h5"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(str(latest_checkpoint), save_weights_only=True),
    vcb.ObjectMAP(val_loader, artifactory, checkpoint_best=True),
    tf.keras.callbacks.TensorBoard(artifactory.tensorboard, profile_batch=0),
    tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda *args, **kwargs: model.reset_metrics())]

backbone = vrsbackbone.SmallFCNN(width_base=16, strides=(2, 4, 8))
fusion = vrsbackbone.FeatureFuser(backbone, final_stride=8, base_width=8, final_width=64)

model = verres.architecture.head.detection.TimePriorizedObjectDetector(num_classes=base_loader.num_classes,
                                                                       backbone=fusion,
                                                                       stride=STRIDE,
                                                                       refinement_stages=1)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
model.train_step(next(stream))

model.fit(dataset.prefetch(10),
          epochs=EPOCHS * VIF,
          steps_per_epoch=stream.steps_per_epoch() // VIF,
          callbacks=callbacks)
