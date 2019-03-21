from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Conv2D
from keras import backend as K
from kerassurgeon import Surgeon


def _compute_saliency(model: Model, x, y, loss, excluded_names):
    ground_truth = K.placeholder(model.output_shape)
    prediction = model.output
    loss_tensor = loss(ground_truth, prediction)

    saliencies = []
    for layer in model.layers:
        if layer.name in excluded_names:
            continue
        for w in layer.weights:
            if not w.trainable or K.ndim(w) == 1:
                continue
            [grad] = model.optimizer.get_gradients(loss_tensor, [w])
            saliencies.append(grad * w)

    get_saliencies = K.function(inputs=[model.input, ground_truth], outputs=saliencies)
    return get_saliencies([x, y])


def _snip(model: Model, saliencies):
    surgeon = Surgeon(model, copy=True)
    for layer in model.layers:
        if not isinstance(layer, Conv2D):
            continue


def snip(model: Model, x, y, loss, exclude):
    saliencies = _compute_saliency(model, x, y, loss, exclude)
    unit_saliences = []
    for sw in saliencies:
        unit_saliences.append(np.abs(sw).mean(axis=(0, 1, 2)))
    for i, us in enumerate(unit_saliences):
        x = np.empty(len(us))
        x[:] = i
        us = np.sqrt(us)
        plt.scatter(x, us, marker=".", color="b")
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
         exclude=("logits",))
