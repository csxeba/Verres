import tensorflow as tf


class CNN:

    def __init__(self, batch_norm=True):
        self.model: tf.keras.Model = None
        self.batch_norm = batch_norm

    def basic_backbone(self, inputs):
        x = tf.keras.layers.Conv2D(16, 5)(inputs)  # 28
        if self.batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D()(x)  # 12

        x = tf.keras.layers.Conv2D(32, 3)(x)
        if self.batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(32, 3)(x)  # 8
        if self.batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D()(x)  # 4
        return x

    def classifier_head(self, features, num_classes):
        x = tf.keras.layers.Flatten()(features)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        if self.batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        return x

    def build_for_cifar(self, num_classes=10):
        inputs = tf.keras.Input((32, 32, 3))
        features = self.basic_backbone(inputs)
        outputs = self.classifier_head(features, num_classes)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile("adam", "categorical_crossentropy", metrics=["acc"])
        return self
