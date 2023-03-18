import cv2
import numpy as np

import verres as V


class SampleVisualizer:

    def __init__(self):
        ...

    def visualize_heatmaps(
        self,
        config: V.Config,
        sample: V.data.Sample,
        alpha: float = 0.5,
    ):
        canvas = sample.encoded_tensors["image_tensor"].copy()
        heatmap = sample.encoded_tensors["heatmap_tensor"]
        for i, class_name in enumerate(config.class_mapping.class_order):
            class_color_bgr = np.array(config.class_mapping.class_colors_rgb[class_name])[::-1]
            class_heatmap = cv2.resize(heatmap[..., i], canvas.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            relevant_pixels = class_heatmap > 0.1

            canvas[relevant_pixels] = (
                    canvas[relevant_pixels] * (1. - alpha) + class_color_bgr[None, :] * class_heatmap[relevant_pixels][:, None] * alpha
            )
        return canvas

    def visualize_semseg(
        self,
        config: V.Config,
        sample: V.data.Sample,
        alpha: float = 0.5
    ):
        canvas = sample.encoded_tensors["image_tensor"].copy()
        segmentation = sample.encoded_tensors["semantic_segmentation_tensor"]
        for i, class_name in enumerate(config.class_mapping.class_order, start=1):
            class_color_bgr = np.array(config.class_mapping.class_colors_rgb[class_name])[::-1]
            bool_mask = np.squeeze(segmentation) == i
            class_segmentation = cv2.resize(bool_mask.astype("uint8"), canvas.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            canvas[bool_mask] = (
                    canvas[bool_mask] * (1. - alpha) + class_color_bgr[None, :] * class_segmentation[bool_mask][:, None] * alpha
            )
        return canvas

    def visualize_instance_seg(
        self,
        config: V.Config,
        sample: V.data.Sample,
        alpha: float = 0.5,
    ):
        canvas = sample.encoded_tensors["image_tensor"].copy()
        segmentation = sample.encoded_tensors["instance_segmentation_tensor"]

        norms = np.linalg.norm(segmentation, axis=-1, ord=2)
        norms /= max(np.max(norms), 1)
        norms *= 255
        norms = np.clip(norms, 0, 255).astype("uint8")

        bool_mask = np.any(segmentation != 0, axis=-1)
        canvas[..., 2][bool_mask] = (1 - alpha) * canvas[..., 2][bool_mask] + alpha * norms[bool_mask]
        return canvas

    def visualize_panoptic_seg(
        self,
        config: V.Config,
        sample: V.data.Sample,
        alpha: float = 0.5,
    ):
        canvas = sample.encoded_tensors["image_tensor"].copy()
        segmentation = sample.encoded_tensors["semantic_segmentation_tensor"]
        for i, class_name in enumerate(config.class_mapping.class_order, start=1):
            class_color_bgr = np.array(config.class_mapping.class_colors_rgb[class_name])[::-1]
            bool_mask = np.squeeze(segmentation) == i
            class_segmentation = cv2.resize(bool_mask.astype("uint8"), canvas.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            canvas[bool_mask] = (
                    canvas[bool_mask] * (1. - alpha) + class_color_bgr[None, :] * class_segmentation[bool_mask][:, None] * alpha
            )
        return canvas
