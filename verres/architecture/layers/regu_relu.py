import tensorflow as tf


class TargetedL2(tf.keras.regularizers.Regularizer):

    def __init__(self, target_value, ord="euclidean"):
        self.target_value = target_value
        self.ord = ord

    def __call__(self, x):
        return tf.norm(self.target_value - x, ord=self.ord)


class LinearPReLUInitializer(tf.keras.initializers.Initializer):

    def __call__(self, shape, dtype=None, partition_info=None):
        return -tf.keras.backend.ones(shape, dtype)


class ReguReLU(tf.keras.layers.PReLU):

    name_counter = 1

    def __init__(self, regularization_ord="euclidean", **prelu_kwargs):
        super().__init__(alpha_initializer=LinearPReLUInitializer(),
                         alpha_regularizer=TargetedL2(target_value=-1, ord=regularization_ord),
                         **prelu_kwargs)
