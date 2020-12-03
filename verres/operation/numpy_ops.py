from typing import Tuple

import numpy as np


def correlate(x):
    var = np.var(x, axis=1)
    cov = np.cov(x)
    cor = cov / var[None, ...]
    return cor


def meshgrid(shape: Tuple[int, int], dtype=None):
    m = np.stack(
        np.meshgrid(
            np.arange(shape[1]),
            np.arange(shape[0])
        ), axis=2).astype(dtype)
    return m
