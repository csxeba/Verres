from keras import backend as K


class Pruner:

    def __init__(self, model, excluded_layer_names):
        self.model = model
        self.excluded_layer_names = excluded_layer_names
        self.weights_of_interest = []
        self.weight_to_layer = {}
        self._determine_layers_of_interest()

    def _determine_layers_of_interest(self):
        for layer in self.model.layers:
            if len(layer.trainable_weights) == 0:
                continue
            if layer.name in self.excluded_layer_names:
                continue
            for weight in layer.weights:
                if K.ndim(weight) == 1:
                    continue
                self.weights_of_interest.append(weight)
                self.weight_to_layer[weight.name] = layer.name
