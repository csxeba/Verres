import tensorflow as tf

from verres.data import cocodoom
from verres.tf_arch import backbone as vrsbackbone, vision
from verres.artifactory import Artifactory
from verres.utils import keras_callbacks as vcb, cocodoom_utils

cocodoom_utils.generate_enemy_dataset()

EPOCHS = 120
BATCH_SIZE = 10
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
    vcb.ObjectMAP(val_loader, artifactory, checkpoint_best=True),
    tf.keras.callbacks.TensorBoard(artifactory.tensorboard, profile_batch=0),
    tf.keras.callbacks.ModelCheckpoint(str(latest_checkpoint), save_weights_only=True)]

backbone = vrsbackbone.SmallFCNN(width_base=8)

model = vision.ObjectDetector(num_classes=loader.num_classes,
                              backbone=backbone,
                              stride=8)
# keras_utils.inject_regularizer(model, kernel_regularizer=tf.keras.regularizers.l2(5e-4))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
model.train_step(next(stream))

model.fit(stream,
          epochs=EPOCHS * VIF,
          steps_per_epoch=stream.steps_per_epoch() // VIF,
          callbacks=callbacks)
