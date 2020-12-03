import tensorflow as tf

from verres.layers import local_error as le_layers
from verres.methods.local_error_model_factory import LocalErrorModelFactory


class LocalErrorCNN:

    def __init__(self,
                 input_shape=(32, 32, 3),
                 output_dim=10,
                 output_activation="softmax",
                 output_loss="categorical_crossentropy",
                 optimizer="adam",
                 use_label_prediction_loss=True,
                 use_similarity_loss=True,
                 use_gradient_barrier=True,
                 backbone_trainable=True,
                 alpha=0.5):

        self.backbone_trainable = backbone_trainable
        self.use_gradient_barrier = use_gradient_barrier
        self.use_similarity_loss = use_similarity_loss
        self.use_label_prediction_loss = use_label_prediction_loss
        self.optimizer = optimizer
        self.output_loss = output_loss
        self.output_activation = output_activation
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.local_error_model_factory = None
        self.alpha = alpha

    def build_experiment_default(self):
        inputs = tf.keras.Input(self.input_shape)

        llkwargs = dict(use_gradient_barrier=self.use_gradient_barrier,
                        use_label_prediction_loss=self.use_label_prediction_loss,
                        use_similarity_loss=self.use_similarity_loss,
                        num_output_classes=self.output_dim,
                        label_prediction_activation=self.output_activation,
                        trainable=self.backbone_trainable)

        layers = [
            le_layers.LocalErrorConvContainer(
                32, (5, 5), padding="same", activation="relu", strides=2, **llkwargs),
            le_layers.LocalErrorConvContainer(
                32, (3, 3), padding="same", activation="relu", **llkwargs),
            le_layers.LocalErrorConvContainer(
                64, (5, 5), padding="same", activation="relu", strides=2, **llkwargs),
            le_layers.LocalErrorConvContainer(
                64, (3, 3), padding="same", activation="relu", **llkwargs),
            le_layers.LocalErrorConvContainer(
                128, (5, 5), padding="same", activation="relu", strides=2, **llkwargs),
            tf.keras.layers.GlobalAveragePooling2D(),
            le_layers.LocalErrorDenseContainer(
                32, activation="relu", **llkwargs),
            tf.keras.layers.Dense(self.output_dim, activation=self.output_activation)
        ]

        x = inputs
        for layer in layers:
            x = layer(x)

        self.local_error_model_factory = LocalErrorModelFactory(
            input_tensor=inputs,
            hidden_layers=layers[:-1],
            output_layer=layers[-1]
        )

        self.local_error_model_factory.compile(
            optimizer=self.optimizer, loss=self.output_loss, metrics=["acc"], alpha=self.alpha
        )

    def build_vgg8b(self):
        inputs = tf.keras.Input(self.input_shape)

        llkwargs = dict(use_gradient_barrier=self.use_gradient_barrier,
                        use_label_prediction_loss=self.use_label_prediction_loss,
                        use_similarity_loss=self.use_similarity_loss,
                        num_output_classes=self.output_dim,
                        label_prediction_activation=self.output_activation,
                        trainable=self.backbone_trainable)

        layers = [
            le_layers.LocalErrorConvContainer(
                128, (3, 3), padding="same", activation="relu", **llkwargs),
            le_layers.LocalErrorConvContainer(
                256, (3, 3), padding="same", activation="relu", **llkwargs),
            tf.keras.layers.MaxPool2D(),
            le_layers.LocalErrorConvContainer(
                256, (3, 3), padding="same", activation="relu", **llkwargs),
            le_layers.LocalErrorConvContainer(
                512, (3, 3), padding="same", activation="relu", **llkwargs),
            tf.keras.layers.MaxPool2D(),
            le_layers.LocalErrorConvContainer(
                512, (3, 3), padding="same", activation="relu", **llkwargs),
            tf.keras.layers.MaxPool2D(),
            le_layers.LocalErrorConvContainer(
                512, (3, 3), padding="same", activation="relu", **llkwargs),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            le_layers.LocalErrorDenseContainer(
                1024, activation="relu", **llkwargs),
            tf.keras.layers.Dense(self.output_dim, activation=self.output_activation)]

        x = inputs
        for layer in layers:
            x = layer(x)

        self.local_error_model_factory = LocalErrorModelFactory(inputs, layers[:-1], output_layer=layers[-1])
        self.local_error_model_factory.compile(
            optimizer=self.optimizer, loss=self.output_loss, metrics=["acc"], alpha=self.alpha
        )

    def fit_generator(self,
                      generator,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True,
                      initial_epoch=0):

        kwargs = locals()
        if "self" in kwargs:
            kwargs.pop("self")
        kwargs["generator"] = self.local_error_model_factory.adapt_data_generator(
            kwargs["generator"]
        )
        if kwargs["validation_data"] is not None:
            kwargs["validation_data"] = self.local_error_model_factory.adapt_data_generator(
                kwargs["validation_data"]
            )
        return self.local_error_model_factory.training_model.fit_generator(**kwargs)

    def predict(self, x, **kwargs):
        return self.local_error_model_factory.inference_model.predict(x, **kwargs)
