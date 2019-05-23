import tensorflow as tf


class MLP:

    def __init__(self, input_dim, *layers, activation="relu"):
        raise RuntimeError("Buggy, fix the output layer addition!")
        self.engine = get_engine(ann_engine)
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=layers[0], input_dim=input_dim, activation=activation))
        for unit in layers[1:-1]:
            self.model.add(tf.keras.layers.Dense(units=unit, activation=activation))

    @classmethod
    def build_classifier(cls, input_dim, output_dim, ann_engine=None):
        engine = get_engine(ann_engine)
        mlp = cls(input_dim, 32, output_dim, activation="tanh", ann_engine=ann_engine)
        mlp.model.add(engine.layers.Activation("softmax"))
        mlp.model.compile("adam", "categorical_crossentropy")
        return mlp

    @classmethod
    def build_regressor(cls, input_dim, output_dim, ann_engine=None):
        mlp = cls(input_dim, 32, output_dim, activation="tanh", ann_engine=ann_engine)
        mlp.model.compile("adam", "mse")
        return mlp
