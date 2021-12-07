from typing import Dict

import tensorflow as tf
import numpy as np
import cv2

import verres as V
from verres.utils import colors as c
from verres.operation import numeric
from . import device


class DataVisualizer:

    ENEMY_TYPES = [
        "POSSESSED", "SHOTGUY", "VILE", "UNDEAD", "FATSO", "CHAINGUY", "TROOP", "SERGEANT", "HEAD", "BRUISER",
        "KNIGHT", "SKULL", "SPIDER", "BABY", "CYBORG", "PAIN", "WOLFSS"
    ]
    COLORS = [
        c.RED, c.BLUE, c.RED, c.BLUE, c.WHITE, c.GREEN, c.YELLOW, c.PINK, c.RED, c.GREEN, c.GREY, c.RED,
        c.WHITE, c.WHITE, c.WHITE, c.WHITE, c.BLUE
    ]

    n_classes = len(ENEMY_TYPES)

    BG = np.zeros((200, 320, 2), dtype="uint8")

    @staticmethod
    def deprocess_image(image):
        image = numeric.untensorize(image)
        return image

    def _colorify_dense_mask(self, y):
        return self._colorify_sparse_mask(y)

    def _colorify_sparse_mask(self, y):
        segmentation = np.zeros(y.shape[:2] + (3,), dtype="uint8")
        for i in range(1, self.n_classes):
            indices = np.where(y == i)
            segmentation[indices[0], indices[1]] = self.COLORS[i]
        return segmentation

    def overlay_segmentation_mask(self, image, semantic_mask, alpha=0.3):
        semantic_mask = numeric.untensorize(semantic_mask)
        if semantic_mask.shape[-1] > 1:
            semantic_mask = np.argmax(semantic_mask, axis=-1)

        image = self.deprocess_image(image)

        for i in range(1, self.n_classes):
            x, y = np.where(semantic_mask == i)
            image[x, y] = self.COLORS[i] * alpha + image[x, y] * (1 - alpha)

        return image

    def overlay_instance_mask(self, image, mask, alpha=0.3):
        image = self.deprocess_image(image)
        if isinstance(mask, tf.Tensor):
            mask = mask.numpy()
        if mask.ndim == 4:
            mask = mask[0]

        data_present = np.any(mask > 0, axis=-1)

        norms = np.linalg.norm(mask, axis=-1, ord=2)
        norms /= max(np.max(norms), 1)
        norms *= 255
        norms = np.clip(norms, 0, 255).astype("uint8")

        image[..., 2][data_present] = (1 - alpha) * image[..., 2][data_present] + alpha * norms[data_present]
        return image.astype("uint8")

    def overlay_panoptic(self, image, coords, affils, alpha=0.3):
        image = self.deprocess_image(image)
        x, y = image.shape[:2]
        coords = tf.cast(coords, tf.int64)
        masks = tf.scatter_nd(coords[:, ::-1], affils, [x, y])
        masks = masks.numpy()
        fg = tf.scatter_nd(coords[:, ::-1], tf.ones(len(coords)), [x, y])
        fg = fg.numpy().astype(bool)

        colorized = np.zeros([x, y, 3], dtype="uint8")

        for i, affil in enumerate(np.unique(affils)):
            mask = masks == affil
            color = c.COLOR[i % len(c.COLOR)]
            colorized[mask] += color[None, :]

        image[fg] = (image[fg] * (1. - alpha) + colorized[fg] * alpha).astype("uint8")
        return image

    @staticmethod
    def overlay_vector_field(image, field, alpha=0.3):
        result = image.copy()
        for x, y in np.argwhere(np.linalg.norm(field, ord=1, axis=-1)):
            dx, dy = field[x, y].astype(int)
            canvas = cv2.arrowedLine(image, (y, x), (y+dy, x+dx), color=(0, 0, 255), thickness=1)
            result = result * alpha + canvas * (1 - alpha)
        return result.astype("uint8")

    def overlay_box_tensor(self, image, bbox, alpha=0.3):
        bbox = numeric.untensorize(bbox)

        if len(image.shape) == 4:
            w = image.shape[1]
        elif len(image.shape) == 3:
            w = image.shape[0]
        else:
            assert False

        stride = w // bbox.shape[0]
        ws, hs = bbox[..., 0::2], bbox[..., 1::2]
        valid = (ws * hs) > 25
        locations = np.argwhere(valid)
        whs = np.stack([ws[valid], hs[valid]], axis=1)
        boxes = boxutil.convert_box_representation(locations[..., :2], whs, locations[..., 2], stride)
        return self.overlay_boxes(image, boxes, stride=1, alpha=alpha)

    def overlay_heatmap(self, image, hmap, alpha=0.3):
        image = self.deprocess_image(image)
        hmap = numeric.untensorize(hmap)

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
        result = np.concatenate([canvas, heatmap], axis=0)
        return result

    def overlay_box(self, image: np.ndarray, box: np.ndarray, type_id: int, alpha: float):
        scale = np.array(image.shape[:2])
        half_wh = (box[2:4] / 2.) * scale
        pt1 = tuple(map(int, box[:2] * scale - half_wh))
        pt2 = tuple(map(int, box[:2] * scale + half_wh))
        image = np.copy(image)
        color = tuple(map(int, self.COLORS[type_id]))
        canvas = cv2.rectangle(image, pt1[::-1], pt2[::-1], color=color, thickness=2)
        canvas = cv2.addWeighted(image, 1. - alpha, canvas, alpha, 1.)
        return canvas

    def overlay_boxes(self, image, boxes: np.ndarray, types: np.ndarray, alpha=0.4):
        image = self.deprocess_image(image)
        for box, type_id in zip(boxes, types):
            image = self.overlay_box(image, box, type_id, alpha=alpha)
        return image


class PredictionVisualizer:

    def __init__(self, config: V.Config):

        self.cfg = config
        self.device = device.output_device_factory(fps=config.inference.fps,
                                                   scale=config.inference.output_upscale_factor,
                                                   to_screen=config.inference.to_screen,
                                                   output_file=config.inference.output_video_path)
        self.visualizer = DataVisualizer()

    def draw_detection(self,
                       image: tf.Tensor,
                       model_output: Dict[str, tf.Tensor],
                       alpha: float = 0.5,
                       write: bool = True):

        canvas = self.visualizer.overlay_boxes(image=image.numpy(),
                                               boxes=model_output["boxes"].numpy(),
                                               types=model_output["types"].numpy(),
                                               alpha=alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def draw_raw_heatmap(self, image, model_output, alpha=0.5, write: bool = True):
        canvas = self.visualizer.overlay_heatmap(image=image, hmap=model_output["heatmap"], alpha=alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def draw_raw_box(self, image, model_output, alpha=1., write: bool = True):
        canvas = self.visualizer.overlay_box_tensor(image, model_output["box_wh"], alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def draw_semantic_segmentation(self, image, model_output, alpha=0.3, write: bool = True):
        canvas = self.visualizer.overlay_segmentation_mask(image, model_output[3], alpha=alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def draw_instance_segmentation(self, image, model_output, alpha=0.3, write: bool = True):
        canvas = self.visualizer.overlay_instance_mask(image, model_output[2], alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def draw_panoptic_segmentation(self, image, model_output, alpha=0.3, write: bool = True):
        canvas = self.visualizer.overlay_panoptic(image, coords=model_output[0], affils=model_output[1], alpha=alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def __enter__(self):
        self.device.__enter__()
        return self

    def __exit__(self, *args):
        self.device.__exit__(*args)
