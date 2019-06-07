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

    def colorify_mask(self, y):
        if y.ndim == 4:
            y = y[0]
        if y.shape[-1] == 1:
            return self._colorify_dense_mask(y)
        else:
            return self._colorify_sparse_mask(y)

    def overlay_mask(self, x, y, alpha=0.3):
        colored = self.colorify_mask(y)  # type: np.ma.MaskedArray
        mask = colored > 0
        x[mask] = alpha * x[mask] + (1 - alpha) * colored[mask]
        return x

    def overlay_heatmap(self, x, y, alpha=0.3):
        heatmap = np.zeros_like(x)
        if y.shape[:2] != x.shape[:2]:
            y = cv2.resize(y, x.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        heatmap[..., 0] = heatmap[..., 1] = y*255
        mask = heatmap > 0.1
        x[mask] = (1 - alpha) * x[mask] + (alpha) * heatmap[mask]
        return x

    def overlay_box(self, image: np.ndarray, box: np.ndarray):
        pt1 = tuple(map(int, box[:2]))
        pt2 = tuple(map(int, box[:2] + box[2:4]))
        c = int(box[-1])
        canvas = np.copy(image)
        canvas = cv2.rectangle(canvas, pt1, pt2, self.COLORS[c], thickness=3)
        return canvas

    def overlay_boxes(self, image: np.ndarray, boxes: np.ndarray):
        for box in boxes:
            image = self.overlay_box(image, box)
        return image


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
