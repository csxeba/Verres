import numpy as np


def convert_box_representation(centroids, whs, types, stride=1):
    boxes = np.concatenate([centroids * stride,
                            whs * stride,
                            types[:, None]], axis=1)
    return boxes

