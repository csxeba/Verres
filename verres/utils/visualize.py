import os
import pathlib
from typing import Tuple, Union, List

import tensorflow as tf
import numpy as np
import cv2

from . import colors as c
from . import box as boxutil
from ..operation import tensor_ops


class Visualizer:

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
        image = tensor_ops.untensorize(image)
        image = image * 255.
        image = np.clip(image, 0, 255).astype("uint8")
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
        semantic_mask = tensor_ops.untensorize(semantic_mask)
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

    def overlay_panoptic(self, image, iseg, sseg, alpha=0.3):
        iseg, sseg = map(tensor_ops.untensorize, [iseg, sseg])
        if sseg.shape[-1] > 1:
            sseg = np.argmax(sseg, axis=-1)
        if sseg.ndim == 2:
            sseg = sseg[..., None]
        iseg_masked = iseg * sseg > 0
        return self.overlay_instance_mask(image, iseg_masked, alpha)

    @staticmethod
    def overlay_vector_field(image, field, alpha=0.3):
        result = image.copy()
        for x, y in np.argwhere(np.linalg.norm(field, ord=1, axis=-1)):
            dx, dy = field[x, y].astype(int)
            canvas = cv2.arrowedLine(image, (y, x), (y+dy, x+dx), color=(0, 0, 255), thickness=1)
            result = result * alpha + canvas * (1 - alpha)
        return result.astype("uint8")

    def overlay_box_tensor(self, image, bbox, alpha=0.3):
        bbox = tensor_ops.untensorize(bbox)

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
        hmap = tensor_ops.untensorize(hmap)

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

    def overlay_box(self, image: np.ndarray, box: np.ndarray, stride: int, alpha: float):
        half_wh = (box[2:4] / 2) * stride
        pt1 = tuple(map(int, box[:2] * stride - half_wh))
        pt2 = tuple(map(int, box[:2] * stride + half_wh))
        color = int(box[-1])
        image = np.copy(image)
        canvas = cv2.rectangle(image, pt1, pt2, self.COLORS[color], thickness=2)
        return canvas

    def overlay_boxes(self, image, boxes: np.ndarray, stride: int = 1, alpha=0.4):
        image = self.deprocess_image(image)
        for box in boxes:
            image = self.overlay_box(image, box, stride, alpha=alpha)
        return image


class OutputDevice:

    def __init__(self, scale: float, fps: int):
        self.scale = scale
        self.fps = fps

    def write(self, frame):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class CV2Screen(OutputDevice):

    def __init__(self, window_name="CV2Screen", fps=None, scale=1.):
        super().__init__(scale, fps)
        self.name = window_name
        if self.fps is None:
            self.fps = 1000
        self.spf = 1000 // fps
        self.online = False

    def write(self, frame):
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


class CV2VideoWriter(OutputDevice):

    def __init__(self, file_name: str, fps: int, scale: float):
        super().__init__(scale, fps)
        self.file_name = file_name
        self.size = None
        self.device: Union[cv2.VideoWriter, None] = None
        self._in_context = False

    def __enter__(self):
        self._in_context = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.device.release()
        self.device = None
        self._in_context = False

    def write(self, frame):
        if not self._in_context:
            raise RuntimeError("Please run in a `with` context!")
        if self.device is None:
            fourcc = cv2.VideoWriter_fourcc(*"DIVX")
            self.device = cv2.VideoWriter(self.file_name, fourcc, float(self.fps), frame.shape[:2][::-1])
        if self.scale != 1:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        self.device.write(frame)


class OutputDeviceList:

    def __init__(self,
                 output_device_list: List[OutputDevice],
                 scale: float = None):

        self.devices: List[OutputDevice] = output_device_list
        self.in_context = False
        if scale is not None:
            for device in self.devices:
                if device.scale not in [scale, 1.]:
                    raise RuntimeError("Ambiguous definitions for scale!")
                device.scale = 1.
        self.scale = scale
        if not self.devices:
            raise RuntimeError("No devices!")

    def __enter__(self):
        self.in_context = True
        for device in self.devices:
            device.__enter__()

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        for device in self.devices:
            device.__exit__(exc_type, exc_val, exc_tb)

    def write(self, frame):
        if not self.in_context:
            raise RuntimeError("Please execute write() in a context manager!")
        if self.scale != 1:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        for device in self.devices:
            device.write(frame)


def output_device_factory(fps: int,
                          scale: float = 1.,
                          to_screen: bool = True,
                          output_file: str = None):

    devices = []
    if to_screen:
        devices.append(CV2Screen(fps=fps, scale=1.))
    if output_file:
        output_file = pathlib.Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        devices.append(CV2VideoWriter(str(output_file), fps, scale=1.))

    return OutputDeviceList(devices, scale)
