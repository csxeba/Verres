import numpy as np
import cv2

from . import colors as c


class COCOSegVisualizer:

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

    def _handle_sparse(self, y):
        segmentation = np.zeros(y.shape[:2] + (3,), dtype="uint8")
        for i in range(1, self.n_classes):
            segmentation[y[..., i] > 0.5] = self.COLORS[i]
        return segmentation

    def _handle_dense(self, y):
        segmentation = np.zeros(y.shape[:2] + (3,), dtype="uint8")
        for i in range(1, self.n_classes):
            indices = np.where(y == i)
            segmentation[indices[0], indices[1]] = self.COLORS[i]
        return segmentation

    def colorify_mask(self, y):
        if y.ndim == 4:
            y = y[0]
        if y.shape[-1] == 1:
            return self._handle_dense(y)
        else:
            return self._handle_sparse(y)

    def overlay(self, x, y, alpha=0.3):
        colored = self.colorify_mask(y)  # type: np.ma.MaskedArray
        mask = colored > 0
        x[mask] = alpha * x[mask] + (1 - alpha) * colored[mask]
        return x


class CV2Screen:

    def __init__(self, window_name="CV2Screen", FPS=None):
        self.name = window_name
        if FPS is None:
            FPS = 1000
        self.spf = 1000 // FPS
        self.online = False

    def blit(self, frame):
        if not self.online:
            self.online = True
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
