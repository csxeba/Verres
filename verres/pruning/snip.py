import numpy as np
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Conv2D
from keras import backend as K
from kerassurgeon import Surgeon


class SnippedConv2D(Conv2D):

    def build(self, input_shape):
        super().build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        self.C = K.variable(np.ones(kernel_shape, dtype="float32"))

    def call(self, inputs):
        outputs = K.conv2d(
            inputs,
            self.C * self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        return outputs


def _snippify(model, exclude):
    surgeon = Surgeon(model, copy=False)
    for layer in model.layers:
        if not isinstance(layer, Conv2D) or layer in exclude:
            continue
        new_layer = SnippedConv2D.from_config(layer.get_config())
        surgeon.add_job("replace_layer", layer, new_layer=new_layer)
    new_model = surgeon.operate()
    new_model.compile(optimizer=model.optimizer, loss=model.loss)
    return new_model


def _compute_saliency(new_model, x, y, loss):
    C = [layer.C for layer in new_model.layers if isinstance(layer, SnippedConv2D)]

    ground_truth = K.placeholder(new_model.output_shape, dtype=y.dtype)
    prediction = new_model.output
    grads = K.gradients(loss(ground_truth, prediction), C)
    get_saliencies = K.function(inputs=[new_model.input, ground_truth], outputs=grads)
    return get_saliencies([x, y])


def _compute_saliency_better(model: Model, x, y, loss, exclude):
    ground_truth = K.placeholder(model.output_shape)
    prediction = model.output
    loss_tensor = loss(ground_truth, prediction)

    weights_of_interest = [layer.weights[0] for layer in model.layers if isinstance(layer, Conv2D) or layer in exclude]
    grads = model.optimizer.get_gradients(loss_tensor, weights_of_interest)
    saliencies = [g * w for g, w in zip(grads, weights_of_interest)]
    get_saliencies = K.function(inputs=[model.input, ground_truth], outputs=saliencies)
    return get_saliencies([x, y])


def _snip(model: Model, saliencies):
    surgeon = Surgeon(model, copy=True)
    for layer in model.layers:
        if not isinstance(layer, Conv2D):
            continue


def snip(model: Model, x, y, loss, exclude):
    # new_model = _snippify(model, exclude)
    saliencies = _compute_saliency_better(model, x, y, loss, exclude)
    filter_saliencies = [np.mean(np.abs(s), axis=(0, 1, 2)) for s in saliencies]
    for i, fs in enumerate(filter_saliencies):
        x = np.empty(len(fs))
        x[:] = i
        fs = np.sqrt(fs)
        plt.scatter(x, fs, marker=".", color="b")
        # plt.scatter(x, fn, marker=".", color="red")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    from verres.data import load_mnist
    from verres.architecture import FCNN
    lX, lY, tX, tY = load_mnist()
    net = FCNN()
    net.build_for_mnist()
    net.model.summary()
    # net.model.fit(lX, lY, batch_size=64, epochs=10, validation_data=(tX, tY))

    snip(net.model, lX[:512], lY[:512], loss=K.categorical_crossentropy,
         exclude=(net.model.get_layer(name="logits"),))
