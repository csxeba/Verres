import tensorflow as tf


class MLP:

    def __init__(self, input_shape, hiddens, outputs, activation="relu", output_activation="linear"):
        self.model = tf.keras.models.Sequential()  # type: tf.keras.models.Sequential
        self.model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        if isinstance(hiddens, int):
            hiddens = (hiddens,)
        for h in hiddens:
            self.model.add(tf.keras.layers.Dense(units=h, activation=activation))
        self.model.add(tf.keras.layers.Dense(units=outputs, activation=output_activation))

    @classmethod
    def build_classifier(cls, input_dim, output_dim):
        mlp = cls(input_dim, 32, output_dim, activation="tanh", output_activation="softmax")
        mlp.model.compile("adam", "categorical_crossentropy")
        return mlp

    @classmethod
    def build_regressor(cls, input_dim, output_dim):
        mlp = cls(input_dim, 32, output_dim, activation="tanh", output_activation="linear")
        mlp.model.compile("adam", "mse")
        return mlp

    @classmethod
    def build_for_mnist(cls):
        mlp = cls.build_classifier([28, 28, 1], 10)
        return mlp
