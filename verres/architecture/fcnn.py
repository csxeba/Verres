import tensorflow as tf

from verres.data import cocodoom


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
    down_stage4 = tf.keras.layers.Conv2D(16, (2, 1), padding="valid")(down_stage4)  # 24
    down_stage4 = tf.keras.layers.BatchNormalization()(down_stage4)
    down_stage4 = tf.keras.layers.ReLU()(down_stage4)

    down_stage4 = tf.keras.layers.Conv2D(16, 3, padding="same")(down_stage4)  # 24
    down_stage4 = tf.keras.layers.BatchNormalization()(down_stage4)
    down_stage4 = tf.keras.layers.ReLU()(down_stage4)

    down_stage5 = tf.keras.layers.MaxPool2D()(down_stage4)  # 12
    down_stage5 = tf.keras.layers.Conv2D(32, 3, padding="same")(down_stage5)
    down_stage5 = tf.keras.layers.BatchNormalization()(down_stage5)
    down_stage5 = tf.keras.layers.ReLU()(down_stage5)

    down_stage5 = tf.keras.layers.Conv2D(64, 3, padding="same")(down_stage5)
    down_stage5 = tf.keras.layers.BatchNormalization()(down_stage5)
    down_stage5 = tf.keras.layers.ReLU()(down_stage5)

    down_stage5 = tf.keras.layers.Conv2D(32, 3, padding="same")(down_stage5)  # 12 x 20 x 64
    down_stage5 = tf.keras.layers.BatchNormalization()(down_stage5)
    down_stage5 = tf.keras.layers.ReLU()(down_stage5)

    return inputs, down_stage1, down_stage2, down_stage3, down_stage4, down_stage5


def _attach_upscaling_branch(backbone_stages):
    down_stage1, down_stage2, down_stage3, down_stage4, down_stage5 = backbone_stages

    x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu", padding="same", strides=2)(down_stage5)  # 12, 20
    x = tf.keras.layers.concatenate([down_stage4, x])
    x = tf.keras.layers.Conv2D(32, 5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.concatenate([down_stage4, x])
    x = tf.keras.layers.Conv2DTranspose(32, (2, 1), padding="valid")(x)  # 25, 40
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(32, 5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu", padding="same", strides=2)(x)  # 50, 80
    x = tf.keras.layers.concatenate([down_stage3, x])
    x = tf.keras.layers.Conv2D(16, 5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu", padding="same", strides=2)(x)  # 100, 160
    x = tf.keras.layers.concatenate([down_stage2, x])
    x = tf.keras.layers.Conv2D(8, 5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu", padding="same", strides=2)(x)  # 200, 320

    return x


def _attach_segmentation_head(num_output_classes, features):
    x = tf.keras.layers.Conv2D(num_output_classes+1, 5, padding="same")(features)
    x = tf.keras.layers.Softmax()(x)
    return x


def _attach_depth_head(features):
    x = tf.keras.layers.Conv2D(1, 5, padding="same")(features)
    return x


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

    inputs, *backbone_stages = _build_backbone(input_shape)
    if task == cocodoom.TASK.SEGMENTATION:
        features = _attach_upscaling_branch(backbone_stages)
        x = _attach_segmentation_head(num_output_classes, features)
    elif task == cocodoom.TASK.DEPTH:
        features = _attach_upscaling_branch(backbone_stages)
        x = _attach_depth_head(features)
    elif task == cocodoom.TASK.DETECTION_INFERENCE:
        x = _attach_box_regression_inference_head(num_output_classes, features=backbone_stages[-2])
    elif task == cocodoom.TASK.DETECTION_TRAINING:
        x, mask = _attach_box_regression_training_head(num_output_classes, features=backbone_stages[-2])
        inputs = [inputs, mask]
    else:
        assert False

    return tf.keras.models.Model(inputs, x)
