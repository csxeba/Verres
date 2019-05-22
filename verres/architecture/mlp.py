from keras import Sequential
from keras.layers import Dense

from verres.keras_engine import get_engine


class MLP:

    def __init__(self, input_dim, *layers, activation="relu", ann_engine=None):
        raise RuntimeError("Buggy, fix the output layer addition!")
        self.engine = get_engine(ann_engine)
        self.model = self.engine.models.Sequential()
        self.model.add(self.engine.layers.Dense(units=layers[0], input_dim=input_dim, activation=activation))
        for unit in layers[1:-1]:
            self.model.add(self.engine.layers.Dense(units=unit, activation=activation))

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
