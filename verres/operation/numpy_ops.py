import numpy as np


def correlate(x):
    var = np.var(x, axis=1)
    cov = np.cov(x)
    cor = cov / var[None, ...]
    return cor
