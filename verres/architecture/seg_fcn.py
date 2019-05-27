import tensorflow as tf

from verres.data import cocodoom


TASK = cocodoom.config.TASK


def build(task: TASK=TASK.SEGMENTATION,
          input_shape: tuple=(None, 200, 320, 3),
          num_output_classes=None):

    inputs = tf.keras.layers.Input(batch_shape=input_shape)

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

    down_stage5 = tf.keras.layers.Conv2D(32, 3, padding="same")(down_stage5)
    down_stage5 = tf.keras.layers.BatchNormalization()(down_stage5)
    down_stage5 = tf.keras.layers.ReLU()(down_stage5)

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
    if task == TASK.SEGMENTATION:
        x = tf.keras.layers.Conv2D(num_output_classes+1, 5, padding="same")(x)
        x = tf.keras.layers.Softmax()(x)
    elif task == TASK.DEPTH:
        x = tf.keras.layers.Conv2D(1, 5, padding="same")(x)

    return tf.keras.models.Model(inputs, x)


if __name__ == '__main__':
    build(num_output_classes=10)
