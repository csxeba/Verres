import numpy as np
import cv2


class Warper:

    def __init__(self, original_shape: np.array, target_shape: np.array):
        self.scales = (target_shape[:2] / original_shape[:2])[::-1]

    def warp_coordinates(self, data):
        coords = data[:, :2] * self.scales
        return np.concatenate([coords, data[:, 2:]], axis=1)

    def warp_box(self, box: np.ndarray):
        pt1 = box[:2] * self.scales
        pt2 = (box[:2] + box[2:]) * self.scales
        return np.concatenate([pt1, pt2 - pt1], axis=0)

    def warp_image(self, image):
        return cv2.resize(image, (0, 0), fx=float(self.scales[0]), fy=float(self.scales[1]))
