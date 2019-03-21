from keras import Sequential
from keras.layers import Dense


class MLP:

    def __init__(self, input_dim, *dimensions, activation="relu"):
        self.model = Sequential()
        self.model.add(Dense(units=dimensions[0], input_dim=input_dim, activation=activation))
        for unit in dimensions[1:]:
            self.model.add(Dense(units=unit, activation=activation))

    @classmethod
    def build_basic(cls, input_dim, output_dim):
        return cls(input_dim, 32, output_dim, activation="tanh")
