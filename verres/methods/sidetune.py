from typing import List

import tensorflow as tf


class SideModel(tf.keras.Model):

    def __init__(self, base_model: tf.keras.Model, alpha: float):
        super().__init__()
        self.base_model = base_model
        self.alpha = tf.Variable(alpha, dtype=tf.float32)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.base_model(inputs)


class Sidetune(tf.keras.Model):

    def __init__(self,
                 base_model: tf.keras.Model,
                 side_models: List[tf.keras.Model]):

        super().__init__()
        self.base_model = base_model
        self.side_models = side_models

    @tf.function
    def call(self, inputs, training=None, mask=None):
        features = self.base_model(inputs)
        sided_features = []
        for base_feature, model in zip(features, self.side_models):
            side_residuals = model(inputs)
            side_feature = base_feature + side_residuals
            sided_features.append(side_feature)
        return sided_features
