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
        self.class_names = list(config.class_mapping.class_order)

    @staticmethod
    def deprocess_image(image):
        image = numeric.untensorize(image)
        return image

    @staticmethod
    def put_text(image, text, origin_xy, color=(255, 255, 255), scale=0.4, thickness=1):
        """Draws a single line of antialiased text with its bottom-left corner at `origin_xy`."""
        origin = (int(origin_xy[0]), int(origin_xy[1]))
        cv2.putText(image, str(text), origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return image

    def overlay_lines(self, image, lines, origin_xy=(6, 6), line_height=15, scale=0.4):
        """Draws a top-left HUD block on a darkened backdrop so the text stays readable."""
        lines = [str(line) for line in lines]
        if not lines:
            return image
        width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for line in lines) + 10
        height = line_height * len(lines) + 6
        x, y = origin_xy
        width = min(width, image.shape[1] - x)
        height = min(height, image.shape[0] - y)
        backdrop = image[y:y + height, x:x + width]
        backdrop[:] = (backdrop * 0.35).astype(backdrop.dtype)
        for index, line in enumerate(lines):
            self.put_text(image, line, (x + 5, y + line_height * (index + 1) - 3), scale=scale)
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

    def overlay_heatmap(self, image, hmap, alpha=0.3):
        image = self.deprocess_image(image)
        hmap = numeric.untensorize(hmap)
        hmap = 1. / (1. + np.exp(-hmap))  # heatmaps are raw logits
        hmap = cv2.resize(hmap, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        for class_idx in range(hmap.shape[-1]):
            color = self.colors_bgr[class_idx]
            mask = hmap[..., class_idx] > 0.1
            image[mask] = alpha * image[mask] + (1. - alpha) * hmap[mask][:, None] * color[None, :]
        return image

    def overlay_box(self, image: np.ndarray, box_corners: np.ndarray, type_id: int, alpha: float, label=None):
        scale = np.array(image.shape[:2])[::-1]
        pt1 = _as_tuple(box_corners[:2] * scale)
        pt2 = _as_tuple(box_corners[2:] * scale)
        color = _as_tuple(self.colors_bgr[type_id])
        image = np.copy(image)
        canvas = cv2.rectangle(image, pt1, pt2, color=color, thickness=2)
        canvas = cv2.addWeighted(image, 1. - alpha, canvas, alpha, 1.)
        if label is not None:
            self.put_text(canvas, label, (pt1[0], max(pt1[1] - 4, 10)), color=color)
        return canvas

    def overlay_boxes(self,
                      image,
                      all_box_corners: np.ndarray,
                      types: np.ndarray,
                      scores: np.ndarray = None,
                      score_threshold: float = 0.0,
                      alpha=0.4):
        """Draws every box above `score_threshold`, colour- and label-coded per class."""
        image = self.deprocess_image(image)
        for index, (corners, type_id) in enumerate(zip(all_box_corners, types)):
            score = None if scores is None else float(scores[index])
            if score is not None and score < score_threshold:
                continue
            if not 0 <= int(type_id) < len(self.class_names):
                raise RuntimeError(f"Detected class index {type_id} is out of range for the class mapping.")
            label = None if score is None else f"{self.class_names[int(type_id)]} {score:.2f}"
            image = self.overlay_box(image, corners, int(type_id), alpha=alpha, label=label)
        return image


class PredictionVisualizer:

    def __init__(self,
                 config: V.Config,
                 score_threshold: float = 0.3,
                 realtime_playback: bool = False):

        self.cfg = config
        self.score_threshold = score_threshold
        self.device = device.output_device_factory(
            fps=config.inference.fps,
            scale=config.inference.output_upscale_factor,
            to_screen=config.inference.to_screen,
            output_file=config.inference.output_video_path,
            realtime_playback=realtime_playback,
        )
        self.visualizer = DataVisualizer(config)

    def draw_detection(self,
                       image: np.ndarray,
                       all_box_corners: np.ndarray,
                       all_box_types: np.ndarray,
                       all_box_scores: np.ndarray = None,
                       score_threshold: float = None,
                       hud: list = None,
                       alpha: float = 0.5,
                       write: bool = True):

        canvas = self.visualizer.overlay_boxes(
            image=image,
            all_box_corners=all_box_corners,
            types=all_box_types,
            scores=all_box_scores,
            score_threshold=self.score_threshold if score_threshold is None else score_threshold,
            alpha=alpha,
        )
        return self._emit(canvas, hud, write)

    def draw_raw_heatmap(self, image, model_output, alpha=0.5, hud: list = None, write: bool = True):
        canvas = self.visualizer.overlay_heatmap(image=image, hmap=model_output["heatmap"], alpha=alpha)
        return self._emit(canvas, hud, write)

    def draw_semantic_segmentation(self, image, model_output, alpha=0.3, hud: list = None, write: bool = True):
        canvas = self.visualizer.overlay_segmentation_mask(image, model_output[3], alpha=alpha)
        return self._emit(canvas, hud, write)

    def draw_instance_segmentation(self, image, model_output, alpha=0.3, hud: list = None, write: bool = True):
        canvas = self.visualizer.overlay_instance_mask(image, model_output[2], alpha)
        return self._emit(canvas, hud, write)

    def draw_panoptic_segmentation(self, image, model_output, alpha=0.3, hud: list = None, write: bool = True):
        canvas = self.visualizer.overlay_panoptic(image, coords=model_output[0], affils=model_output[1], alpha=alpha)
        return self._emit(canvas, hud, write)

    def _emit(self, canvas, hud: list, write: bool):
        """Stamps the HUD onto a finished canvas and forwards it to every output device."""
        if hud:
            self.visualizer.overlay_lines(canvas, hud)
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
            all_box_scores=label_obj.object_scores,
            score_threshold=0.,
            alpha=alpha,
            write=write,
        )

    def __enter__(self):
        self.device.__enter__()
        return self

    def __exit__(self, *args):
        self.device.__exit__(*args)
