import tensorflow as tf


class CNN:

    def __init__(self):
        self.model = None  # type: tf.keras.Model

    def build_for_cifar(self, num_classes=10):

        inputs = tf.keras.Input((32, 32, 3))

        x = tf.keras.layers.Conv2D(16, 5)(inputs)  # 28
        x = tf.keras.layers.MaxPool2D()(x)  # 12
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(32, 3)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(32, 3)(x)  # 8
        x = tf.keras.layers.MaxPool2D()(x)  # 4
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        self.model = tf.keras.Model(inputs, x)
        self.model.compile("adam", "categorical_crossentropy", metrics=["acc"])
        return self
