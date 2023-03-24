from typing import Type

import tensorflow as tf
import numpy as np
import cv2

import verres as V
from verres.utils import colors as c
from verres.operation import numeric
from . import device


def _as_tuple(x: np.ndarray, cast_type: Type = int) -> tuple:
    assert x.ndim == 1
    return tuple(map(cast_type, x))


class DataVisualizer:
    BG = np.zeros((200, 320, 2), dtype="uint8")

    def __init__(self, config: V.Config):
        self.cfg = config
        self.colors_bgr = np.array(list(config.class_mapping.class_colors_rgb.values()))[:, ::-1]

    @staticmethod
    def deprocess_image(image):
        image = numeric.untensorize(image)
        return image

    def _colorify_dense_mask(self, y):
        return self._colorify_sparse_mask(y)

    def _colorify_sparse_mask(self, y):
        segmentation = np.zeros(y.shape[:2] + (3,), dtype="uint8")
        for i in range(1, self.cfg.class_mapping.num_classes):
            indices = np.where(y == i)
            segmentation[indices[0], indices[1]] = self.colors_bgr[i]
        return segmentation

    def overlay_segmentation_mask(self, image, semantic_mask, alpha=0.3):
        semantic_mask = numeric.untensorize(semantic_mask)
        if semantic_mask.shape[-1] > 1:
            semantic_mask = np.argmax(semantic_mask, axis=-1)

        image = self.deprocess_image(image)

        for i, color in enumerate(self.colors_bgr):
            x, y = np.where(semantic_mask == i)
            image[x, y] = color * alpha + image[x, y] * (1 - alpha)

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

    def overlay_heatmap(self, image, hmap, alpha=0.3):
        image = self.deprocess_image(image)
        hmap = numeric.untensorize(hmap)
        hmap = cv2.resize(hmap, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        for class_idx in range(hmap.shape[-1]):
            color = self.colors_bgr[class_idx]
            mask = hmap[..., class_idx] > 0.1
            image[mask] = alpha * image[mask] + (1. - alpha) * hmap[mask][:, None] * color[None, :]
        return image

    def overlay_box(self, image: np.ndarray, box_corners: np.ndarray, type_id: int, alpha: float):
        scale = np.array(image.shape[:2])[::-1]
        pt1 = _as_tuple(box_corners[:2] * scale)
        pt2 = _as_tuple(box_corners[2:] * scale)
        color = _as_tuple(self.colors_bgr[type_id])
        image = np.copy(image)
        canvas = cv2.rectangle(image, pt1, pt2, color=color, thickness=2)
        canvas = cv2.addWeighted(image, 1. - alpha, canvas, alpha, 1.)
        return canvas

    def overlay_boxes(self, image, all_box_corners: np.ndarray, types: np.ndarray, alpha=0.4):
        image = self.deprocess_image(image)
        for corners, type_id in zip(all_box_corners, types):
            image = self.overlay_box(image, corners, type_id, alpha=alpha)
        return image


class PredictionVisualizer:

    def __init__(self, config: V.Config):

        self.cfg = config
        self.device = device.output_device_factory(
            fps=config.inference.fps,
            scale=config.inference.output_upscale_factor,
            to_screen=config.inference.to_screen,
            output_file=config.inference.output_video_path,
        )
        self.visualizer = DataVisualizer(config)

    def draw_detection(self,
                       image: np.ndarray,
                       all_box_corners: np.ndarray,
                       all_box_types: np.ndarray,
                       alpha: float = 0.5,
                       write: bool = True):

        canvas = self.visualizer.overlay_boxes(
            image=image,
            all_box_corners=all_box_corners,
            types=all_box_types,
            alpha=alpha,
        )
        if write:
            self.device.write(canvas)
        return canvas

    def draw_raw_heatmap(self, image, model_output, alpha=0.5, write: bool = True):
        canvas = self.visualizer.overlay_heatmap(image=image, hmap=model_output["heatmap"], alpha=alpha)
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

    def draw_from_sample(
        self,
        sample: V.data.Sample,
        for_gt: bool = False,
        alpha: float = 0.7,
        write: bool = True,
    ) -> np.ndarray:
        label_obj = sample.label if for_gt else sample.detection
        assert label_obj is not None
        return self.draw_detection(
            image=sample.encoded_tensors["image"],
            all_box_corners=label_obj.object_keypoint_coords,
            all_box_types=label_obj.object_types,
            alpha=alpha,
            write=write,
        )

    def __enter__(self):
        self.device.__enter__()
        return self

    def __exit__(self, *args):
        self.device.__exit__(*args)
