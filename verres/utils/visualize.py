import os
from typing import Tuple, Union

import tensorflow as tf
import numpy as np
import cv2

from . import colors as c


class Visualizer:

    ENEMY_TYPES = [
        "POSSESSED", "SHOTGUY", "VILE", "UNDEAD", "FATSO", "CHAINGUY", "TROOP", "SERGEANT", "HEAD", "BRUISER",
        "KNIGHT", "SKULL", "SPIDER", "BABY", "CYBORG", "PAIN", "WOLFSS"
    ]
    COLORS = [
        c.RED, c.BLUE, c.RED, c.BLUE, c.WHITE, c.GREEN, c.YELLOW, c.PINK, c.RED, c.GREEN, c.GREY, c.RED,
        c.WHITE, c.WHITE, c.WHITE, c.WHITE, c.BLUE
    ]

    def __init__(self, n_classes):
        self.n_classes = n_classes

    @staticmethod
    def deprocess_image(image):
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        if image.ndim == 4:
            image = image[0]
        image = image * 255.
        image = np.clip(image, 0, 255).astype("uint8")
        return image

    def _colorify_sparse_mask(self, y):
        segmentation = np.zeros(y.shape[:2] + (3,), dtype="uint8")
        for i in range(1, self.n_classes):
            segmentation[y[..., i] > 0.5] = self.COLORS[i]
        return segmentation

    def _colorify_dense_mask(self, y):
        segmentation = np.zeros(y.shape[:2] + (3,), dtype="uint8")
        for i in range(1, self.n_classes):
            indices = np.where(y == i)
            segmentation[indices[0], indices[1]] = self.COLORS[i]
        return segmentation

    def colorify_segmentation_mask(self, y):
        if y.ndim == 4:
            y = y[0]
        if y.shape[-1] == 1:
            return self._colorify_dense_mask(y)
        else:
            return self._colorify_sparse_mask(y)

    def overlay_segmentation_mask(self, x, y, alpha=0.3):
        colored = self.colorify_segmentation_mask(y)
        mask = colored > 0
        x[mask] = alpha * x[mask] + (1 - alpha) * colored[mask]
        return x

    @staticmethod
    def overlay_instance_mask(image, mask, alpha=0.3):
        angles = np.linalg.norm(mask, axis=-1, ord=1)
        angles /= np.max(angles)
        angles = (angles * 255).astype("uint8")
        angles = cv2.cvtColor(angles, cv2.COLOR_GRAY2BGR)
        angles = cv2.cvtColor(angles, cv2.COLOR_BGR2HSV)
        image = alpha * image + (1 - alpha) * angles
        return image.astype("uint8")

    @staticmethod
    def overlay_vector_field(image, field, alpha=0.3):
        result = image.copy()
        for x, y in np.argwhere(np.linalg.norm(field, ord=1, axis=-1)):
            dx, dy = field[x, y].astype(int)
            canvas = cv2.arrowedLine(image, (y, x), (y+dy, x+dx), color=(0, 0, 255), thickness=1)
            result = result * alpha + canvas * (1 - alpha)
        return result.astype("uint8")

    def overlay_heatmap(self, image, hmap, alpha=0.3):
        if isinstance(image, tf.Tensor):
            image = self.deprocess_image(image)
        if isinstance(hmap, tf.Tensor):
            hmap = hmap.numpy()
        if hmap.ndim == 4:
            hmap = hmap[0]

        canvas = image.copy()
        heatmap = np.max(hmap, axis=-1)
        heatmap_max = np.max(heatmap)
        if heatmap_max > 1.:
            heatmap /= np.max(heatmap)
        heatmap *= 255
        heatmap = heatmap.astype("uint8")
        heatmap = np.stack([np.zeros_like(heatmap)]*2 + [heatmap], axis=-1)
        if hmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, image.shape[:2][::-1])
        mask = heatmap > 25
        canvas[mask] = alpha * image[mask] + (1 - alpha) * heatmap[mask]
        return canvas

    def overlay_box(self, image: np.ndarray, box: np.ndarray, stride):
        half_wh = (box[2:4] / 2) * stride
        pt1 = tuple(map(int, box[:2] * stride - half_wh))
        pt2 = tuple(map(int, box[:2] * stride + half_wh))
        color = int(box[-1])
        canvas = np.copy(image)
        canvas = cv2.rectangle(canvas, pt1, pt2, self.COLORS[color], thickness=3)
        return canvas

    def overlay_boxes(self, image, boxes: np.ndarray, stride: int = 1):
        if isinstance(image, tf.Tensor):
            image = self.deprocess_image(image)
        for box in boxes:
            image = self.overlay_box(image, box, stride)
        return image


class CV2Screen:

    def __init__(self, window_name="CV2Screen", fps=None, scale=1.):
        self.name = window_name
        if fps is None:
            fps = 1000
        self.spf = 1000 // fps
        self.online = False
        self.scale = scale

    def blit(self, frame):
        if not self.online:
            self.online = True
        if self.scale != 1:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(self.name, frame)
        cv2.waitKey(self.spf)

    def teardown(self):
        if self.online:
            cv2.destroyWindow(self.name)
        self.online = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()

    def __del__(self):
        self.teardown()


class CV2VideoWriter:

    def __init__(self, file_name: str, fps: int, size: Tuple[int, int] = (200, 320)):
        self.file_name = file_name
        assert os.path.splitext(file_name)[-1][-3:] == "mp4"
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        self.size = size
        self.device: Union[cv2.VideoWriter, None] = None
        self._in_context = False

    def __enter__(self):
        self._in_context = True
        self.device = cv2.VideoWriter(self.file_name, self.fourcc, float(self.fps), self.size)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.device.release()
        self.device = None
        self._in_context = False

    def write(self, frame):
        if not self._in_context:
            raise RuntimeError("Please run in a `with` context!")
        self.device.write(frame)