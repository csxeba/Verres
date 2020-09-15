import tensorflow as tf

from verres.data import cocodoom
from verres.tf_arch import backbone as vrsbackbone, vision

STRIDE = 8
BATCH_SIZE = 8

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/map-val.json",
        images_root="/data/Datasets/cocodoom",
        stride=STRIDE
    )
)
full_loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/map-full-val.json",
        images_root="/data/Datasets/cocodoom",
        stride=STRIDE
    )
)
stream = cocodoom.COCODoomTimeSequence(
    cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.DETECTION,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  min_no_visible_objects=2),
    data_loader=loader,
    full_data_loader=full_loader
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
    output_shapes=output_shapes
)

backbone = vrsbackbone.SmallFCNN(width_base=16, strides=(2, 4, 8))
fusion = vrsbackbone.FeatureFuser(backbone, final_stride=8, base_width=8, final_width=64)

model = vision.TimePriorizedObjectDetector(num_classes=loader.num_classes,
                                           backbone=fusion,
                                           stride=STRIDE,
                                           refinement_stages=1)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
model.train_step(next(stream))

model.fit(dataset.prefetch(10),
          epochs=120 * 2,
          steps_per_epoch=stream.steps_per_epoch() // 2)
