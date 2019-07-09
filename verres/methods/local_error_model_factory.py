import tensorflow as tf

from verres.layers import local_error as le_layers
from verres.operation import numpy_ops


class LocalErrorModelFactory:

    def __init__(self, input_tensor, hidden_layers, output_layer):

        self.correlation_outputs = []
        self.prediction_outputs = []
        self.losses = []

        if isinstance(output_layer, le_layers.LocalErrorBase):
            raise ValueError("Model output layer should not be a local error layer!")

        self.input_tensor = input_tensor
        self.output_layer = output_layer

        for h in hidden_layers:
            if not isinstance(h, le_layers.LocalErrorBase):
                continue
            if h.correlation_output is not None:
                self.correlation_outputs.append(h.correlation_output)
            if h.label_prediction_output is not None:
                self.prediction_outputs.append(h.label_prediction_output)

        self._training_model = None
        self._inference_model = None

    def compile(self, optimizer, loss, metrics=None):
        num_corr = len(self.correlation_outputs)
        num_pred = len(self.prediction_outputs)

        model = tf.keras.Model(self.input_tensor,
                               self.correlation_outputs + self.prediction_outputs + [self.output_layer.output])
        model.compile(
            optimizer, ["mse"]*num_corr + [loss]*num_pred + [loss],
            metrics={self.output_layer.name: metric for metric in metrics}
        )
        self._training_model = model
        self._inference_model = tf.keras.Model(self.input_tensor, self.output_layer.output)

    @property
    def training_model(self) -> tf.keras.Model:
        if self._training_model is None:
            raise RuntimeError("Please compile this model first!")
        return self._training_model

    @property
    def inference_model(self) -> tf.keras.Model:
        return self._inference_model

    def adapt_data_generator(self, generator=None):
        if generator is None:
            return generator
        num_corr = len(self.correlation_outputs)
        num_pred = len(self.prediction_outputs)
        for x, y in generator:
            y_correl = numpy_ops.correlate(y)
            new_y = [y_correl] * num_corr + [y] * num_pred + [y]
            yield x, new_y
