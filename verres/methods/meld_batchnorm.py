import numpy as np


def meld(convolution_params, batchnorm_params, epsilon):
    new_weights = []
    for c, b in zip(convolution_params, batchnorm_params):
        gamma, beta, mean, std = b
        W = c[0]

        scale = gamma / (std + epsilon)

        new_weights.append(
            [W * scale]
        )

        if len(c) >= 2:
            new_weights[-1].append(
                (c[1] - mean) * scale + beta
            )
    return new_weights
