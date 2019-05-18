import cv2
import numpy as np


def decode_poly(poly, shape):
    full_mask = np.zeros(shape, dtype="uint8")
    pts = [np.round(np.array(p).reshape(-1, 2)).astype(int) for p in poly]
    return cv2.fillPoly(full_mask, pts, color=1).astype(bool)


def decode_rle(rle, shape):
    full_mask = np.zeros(np.prod(shape), dtype=bool)
    fill = False
    start = 0
    for num in rle["counts"]:
        end = start + num
        full_mask[start:end] = fill
        fill = not fill
        start = end
    return full_mask.reshape(shape[::-1]).T


def get_mask(annotation, image_shape):
    if isinstance(annotation["segmentation"], list):
        return decode_poly(annotation["segmentation"], image_shape).astype(bool)
    elif "counts" in annotation["segmentation"]:
        return decode_rle(annotation["segmentation"], image_shape)
