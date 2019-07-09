import tensorflow as tf

from verres.layers import local_error as le_layers
from verres.methods.local_error_model_factory import LocalErrorModelFactory


class LocalErrorCNN:

    def __init__(self,
                 input_shape=(32, 32, 3),
                 output_dim=10,
                 output_activation="softmax",
                 output_loss="categorical_crossentropy",
                 optimizer="adam"):

        inputs = tf.keras.Input(input_shape)

        self.c1 = le_layers.LocalErrorConvContainer(
            32, (5, 5), padding="same", activation="relu", strides=2,
            use_gradient_barrier=True,
            use_label_prediction_loss=True,
            use_similarity_loss=True,
            num_output_classes=output_dim,
            label_prediction_activation=output_activation
        )
        self.c2 = le_layers.LocalErrorConvContainer(
            32, (3, 3), padding="same", activation="relu",
            use_gradient_barrier=True,
            use_label_prediction_loss=True,
            use_similarity_loss=True,
            num_output_classes=output_dim,
            label_prediction_activation=output_activation

        )
        self.c3 = le_layers.LocalErrorConvContainer(
            64, (5, 5), padding="same", activation="relu", strides=2,
            use_gradient_barrier=True,
            use_label_prediction_loss=True,
            use_similarity_loss=True,
            num_output_classes=output_dim,
            label_prediction_activation=output_activation

        )
        self.c4 = le_layers.LocalErrorConvContainer(
            64, (3, 3), padding="same", activation="relu",
            use_gradient_barrier=True,
            use_label_prediction_loss=True,
            use_similarity_loss=True,
            num_output_classes=output_dim,
            label_prediction_activation=output_activation

        )
        self.c5 = le_layers.LocalErrorConvContainer(
            128, (5, 5), padding="same", activation="relu", strides=2,
            use_gradient_barrier=True,
            use_label_prediction_loss=True,
            use_similarity_loss=True,
            num_output_classes=output_dim,
            label_prediction_activation=output_activation
        )

        self.d1 = le_layers.LocalErrorDenseContainer(
            32, activation="relu",
            use_gradient_barrier=True,
            use_label_prediction_loss=True,
            use_similarity_loss=True,
            num_output_classes=output_dim,
            label_prediction_activation=output_activation
        )
        self.d2 = tf.keras.layers.Dense(output_dim, activation=output_activation)

        x = self.c1(inputs)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.d1(x)
        x = self.d2(x)

        self.local_error_model_factory = LocalErrorModelFactory(
            input_tensor=inputs,
            hidden_layers=[self.c1, self.c2, self.c3, self.c4, self.c5, self.d1],
            output_layer=self.d2
        )

        self.local_error_model_factory.compile(
            optimizer=optimizer, loss=output_loss, metrics=["acc"]
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
