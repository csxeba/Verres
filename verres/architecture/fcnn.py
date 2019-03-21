from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Activation


class FCNN:

    def __init__(self):
        self.model = None  # type: Sequential

    def build_for_mnist(self):
        self.model = Sequential([
            Conv2D(32, (5, 5), strides=2, input_shape=(28, 28, 1), padding="same", activation="relu"),
            Conv2D(64, (3, 3), activation="relu"),
            Conv2D(64, (3, 3), activation="relu"),
            Conv2D(128, (5, 5), strides=2, padding="same", activation="relu"),
            Conv2D(128, (3, 3), activation="relu"),
            Conv2D(10, (1, 1), name="logits"),
            GlobalAveragePooling2D(),
            Activation("softmax")
        ])
        self.model.compile("adam", "categorical_crossentropy", metrics=["acc"])
