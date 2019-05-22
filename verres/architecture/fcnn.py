from verres.keras_engine import get_engine


class FCNN:

    def __init__(self, ann_engine=None):
        self.engine = get_engine(ann_engine)
        self.model = None

    def build_simple_classifier(self, input_shape, output_dim):
        self.model = self.engine.models.Sequential([
            self.engine.layers.Conv2D(32, (5, 5), strides=2, input_shape=input_shape, padding="same", activation="relu"),
            self.engine.layers.Conv2D(64, (3, 3), activation="relu"),
            self.engine.layers.Conv2D(64, (3, 3), activation="relu"),
            self.engine.layers.Conv2D(128, (5, 5), strides=2, padding="same", activation="relu"),
            self.engine.layers.Conv2D(128, (3, 3), activation="relu"),
            self.engine.layers.Conv2D(output_dim, (1, 1), name="logits"),
            self.engine.layers.GlobalAveragePooling2D(),
            self.engine.layers.Activation("softmax")
        ])
        self.model.compile("adam", "categorical_crossentropy", metrics=["acc"])

    def build_for_mnist(self):
        self.build_simple_classifier((28, 28, 1), 10)

    def build_for_cifar(self, num_classes=10):
        self.build_simple_classifier((32, 32, 3), num_classes)
