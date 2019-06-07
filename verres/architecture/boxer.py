import tensorflow as tf

from ..data import cocodoom


def _build_backbone(input_shape):
    if len(input_shape) == 3:
        input_shape = (None,) + input_shape
    if input_shape == "cocodoom":
        input_shape = (None, 200, 320, 3)

    inputs = tf.keras.layers.Input(batch_shape=input_shape, name="images")

    down_stage1 = tf.keras.layers.Conv2D(8, 3, padding="same")(inputs)
    down_stage1 = tf.keras.layers.BatchNormalization()(down_stage1)
    down_stage1 = tf.keras.layers.ReLU()(down_stage1)

    down_stage2 = tf.keras.layers.MaxPool2D()(down_stage1)  # 100
    down_stage2 = tf.keras.layers.Conv2D(16, 3, padding="same")(down_stage2)
    down_stage2 = tf.keras.layers.BatchNormalization()(down_stage2)
    down_stage2 = tf.keras.layers.ReLU()(down_stage2)

    down_stage3 = tf.keras.layers.MaxPool2D()(down_stage2)  # 50
    down_stage3 = tf.keras.layers.Conv2D(16, 3, padding="same")(down_stage3)
    down_stage3 = tf.keras.layers.BatchNormalization()(down_stage3)
    down_stage3 = tf.keras.layers.ReLU()(down_stage3)

    down_stage4 = tf.keras.layers.MaxPool2D()(down_stage3)  # 25
    down_stage4 = tf.keras.layers.Conv2D(16, 3, padding="same")(down_stage4)
    down_stage4 = tf.keras.layers.BatchNormalization()(down_stage4)
    down_stage4 = tf.keras.layers.ReLU()(down_stage4)

    down_stage4 = tf.keras.layers.Conv2D(16, 3, padding="same")(down_stage4)
    down_stage4 = tf.keras.layers.BatchNormalization()(down_stage4)
    down_stage4 = tf.keras.layers.ReLU()(down_stage4)

    return inputs, down_stage4


def _attach_box_regression_inference_head(num_output_classes, features):
    x = tf.keras.layers.Conv2D(num_output_classes+2+2, 5, padding="same")(features)
    return x


def _attach_box_regression_training_head(num_output_classes, features):
    num_maps = num_output_classes+2+2
    mask = tf.keras.Input(batch_shape=tf.keras.backend.int_shape(features)[:3] + (num_maps,))
    x = tf.keras.layers.Conv2D(num_maps, 5, padding="same")(features)
    x = tf.keras.layers.Lambda(lambda xx: xx[0] * xx[1])([x, mask])
    return x, mask


def build(task: cocodoom.TASK,
          input_shape: tuple=(None, 200, 320, 3),
          num_output_classes=None):

    inputs, backbone_stage = _build_backbone(input_shape)
    if task == cocodoom.TASK.DETECTION_INFERENCE:
        x = _attach_box_regression_inference_head(num_output_classes, features=backbone_stage)
    elif task == cocodoom.TASK.DETECTION_TRAINING:
        x, mask = _attach_box_regression_training_head(num_output_classes, features=backbone_stage)
        inputs = [inputs, mask]
    else:
        assert False

    return tf.keras.models.Model(inputs, x)

