def meld(convolution_params, batchnorm_params):
    new_weights = []
    for c, b in zip(convolution_params, batchnorm_params):
        mean, var, gamma, beta = b
        W = c[0]

        scale = gamma / var

        new_weights.append(
            [W * scale]
        )

        if len(c) >= 2:
            new_weights.append(
                [(c[1] - mean) * scale + beta]
            )
    return new_weights
