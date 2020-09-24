import json
import os
from collections import defaultdict
from typing import Union

import numpy as np
import cv2

from verres.utils import masking, transform
from .config import COCODoomLoaderConfig

ENEMY_TYPES = [
    "POSSESSED", "SHOTGUY", "VILE", "UNDEAD", "FATSO", "CHAINGUY", "TROOP", "SERGEANT", "HEAD", "BRUISER",
    "KNIGHT", "SKULL", "SPIDER", "BABY", "CYBORG", "PAIN", "WOLFSS"
]


class COCODoomLoader:

    IMAGE_SHAPE = (200, 320, 3)

    def __init__(self, config: COCODoomLoaderConfig):

        self.cfg = config

        data = json.load(open(config.data_json))

        self.categories = {cat["id"]: cat for cat in data["categories"]}
        self.image_meta = {meta["id"]: meta for meta in data["images"]}
        self.index = defaultdict(list)
        for anno in data["annotations"]:
            category = self.categories[anno["category_id"]]
            if category["name"] not in ENEMY_TYPES:
                continue
            self.index[anno["image_id"]].append(anno)
        self.num_classes = len(ENEMY_TYPES)
        self.cache = {}
        self.cache_id = -1
        self.model_input_shape = config.input_shape or self.IMAGE_SHAPE
        self.warper: Union[transform.Warper, None] = None
        if config.input_shape is not None:
            self.warper = transform.Warper(original_shape=np.array(self.IMAGE_SHAPE),
                                           target_shape=config.input_shape)
        self.model_output_shape = (self.model_input_shape[0] // config.stride,
                                   self.model_input_shape[1] // config.stride)

        print(f" [Verres.COCODoomLoader] - Num images :", len(data["images"]))
        print(f" [Verres.COCODoomLoader] - Num annos  :", len(data["annotations"]))
        print(f" [Verres.COCODoomLoader] - Num classes:", self.num_classes)

    @classmethod
    def default_train(cls):
        cfg = COCODoomLoaderConfig("/data/Datasets/cocodoom/map-train.json",
                                   "/data/Datasets/cocodoom")
        return cls(cfg)

    @classmethod
    def default_val(cls):
        cfg = COCODoomLoaderConfig("/data/Datasets/cocodoom/map-val.json",
                                   "/data/Datasets/cocodoom")
        return cls(cfg)

    @classmethod
    def default_test(cls):
        cfg = COCODoomLoaderConfig("/data/Datasets/cocodoom/map-full-test.json",
                                   "/data/Datasets/cocodoom")
        return cls(cfg)

    @property
    def N(self):
        return len(self.index)

    def empty_image(self):
        return np.zeros(self.model_input_shape, dtype="uint8")

    def empty_tensor(self, depth: int, stride: int = None, dtype="float32"):
        if stride is None:
            s = self.cfg.stride
        else:
            s = stride
        sh = self.model_input_shape
        return np.zeros((sh[0] // s, sh[1] // s, depth), dtype=dtype)

    def empty_sparse_tensor(self, dims: int, dtype):
        return np.zeros((0, dims), dtype=dtype)

    def get_image(self, image_id, preprocess=False):
        if image_id is None:
            return self.empty_image()
        meta = self.image_meta[image_id]
        image_path = os.path.join(self.cfg.images_root, meta["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"No image found @ {image_path}")
        if self.warper is not None:
            image = self.warper.warp_image(image)
        if preprocess:
            image = np.float32(image / 255.)
        return image

    def get_panoptic_masks(self, image_id):
        segmentation_mask = self.empty_tensor(depth=1, stride=1, dtype="int64")
        instance_canvas = self.empty_tensor(depth=2, stride=1, dtype="float32")
        if image_id is None:
            return [instance_canvas, segmentation_mask]
        coord_template = np.stack(
            np.meshgrid(
                np.arange(self.model_input_shape[1]),
                np.arange(self.model_input_shape[0])),
            axis=-1)

        for anno in self.index[image_id]:
            category = self.categories[anno["category_id"]]
            if category["name"] not in ENEMY_TYPES:
                continue

            class_idx = ENEMY_TYPES.index(category["name"])

            instance_mask = np.squeeze(masking.get_mask(anno, self.IMAGE_SHAPE))
            if self.warper is not None:
                instance_mask = self.warper.warp_image(instance_mask)

            coords = np.argwhere(instance_mask)  # type: np.ndarray

            segmentation_mask[instance_mask] = class_idx+1
            instance_canvas[instance_mask] = coords.mean(axis=0, keepdims=True) - coords

        return [instance_canvas, segmentation_mask]

    def get_depth_image(self, image_id):
        meta = self.image_meta[image_id]
        depth_image_path = meta["file_path"].replace("/rgb/", "/depth/")
        depth_image = cv2.imread(depth_image_path)
        return depth_image

    def get_object_heatmap(self, image_id):
        annos = self.index[image_id]
        heatmap = self.empty_tensor(depth=self.num_classes, stride=self.cfg.stride, dtype="float32")
        if image_id is None or len(annos) == 0:
            return heatmap

        tensor_shape = heatmap.shape[:2]
        hit = 0
        for anno in annos:
            category = self.categories[anno["category_id"]]
            if category["name"] not in ENEMY_TYPES:
                continue

            hit = 1
            class_idx = ENEMY_TYPES.index(category["name"])
            box = np.array(anno["bbox"]) / self.cfg.stride
            if self.warper is not None:
                box = self.warper.warp_box(box)
            centroid = box[:2] + box[2:] / 2
            centroid_rounded = np.clip(np.round(centroid).astype(int), [0, 0], tensor_shape[:2][::-1]-1)
            heatmap[centroid_rounded[1], centroid_rounded[0], class_idx] = 1

        if hit:
            kernel_size = 5
            heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0, borderType=cv2.BORDER_CONSTANT)
            heatmap /= heatmap.max()

        return heatmap

    def _get_regression_base(self, image_id, batch_idx):

        if image_id == self.cache_id:
            return

        self.cache_id = image_id

        if image_id is None or len(self.index[image_id]) == 0:
            self.cache["locations"] = np.zeros((0, 4), dtype=int)
            self.cache["values"] = np.zeros((0, 2), dtype="float32")
            self.cache["bbox"] = np.zeros((0, 2), dtype="float32")
            return

        tensor_shape = np.array(self.model_output_shape)
        _01 = [0, 1]
        _10 = [1, 0]
        locations = []
        values = []
        bbox = []
        for anno in self.index[image_id]:
            category = self.categories[anno["category_id"]]
            if category["name"] not in ENEMY_TYPES:
                continue

            class_idx = ENEMY_TYPES.index(category["name"])
            box = np.array(anno["bbox"]) / self.cfg.stride
            if self.warper is not None:
                box = self.warper.warp_box(box)
            centroid = box[:2] + box[2:] / 2

            centroid_floored = np.floor(centroid).astype(int)
            augmented_coords = np.stack([
                centroid_floored, centroid_floored + _01, centroid_floored + _10, centroid_floored + 1
            ], axis=0)

            in_frame = np.all([augmented_coords >= 0, augmented_coords < tensor_shape[::-1][None, :]], axis=(0, 2))
            augmented_coords = augmented_coords[in_frame]
            augmented_locations = np.concatenate([
                np.full((len(augmented_coords), 1), batch_idx, dtype=augmented_coords.dtype),
                augmented_coords,
                np.full((len(augmented_coords), 1), class_idx, dtype=augmented_coords.dtype)
            ], axis=1)
            augmented_values = centroid[None, :] - augmented_coords
            augmented_boxes = np.stack([box[2:]]*4, axis=0)[in_frame]

            locations.append(augmented_locations)
            values.append(augmented_values)
            bbox.append(augmented_boxes)

        self.cache["locations"] = np.concatenate(locations).astype(int)
        self.cache["values"] = np.concatenate(values).astype("float32")
        self.cache["bbox"] = np.concatenate(bbox).astype("float32")

    def get_refinements(self, image_id, batch_idx):
        self._get_regression_base(image_id, batch_idx)
        return self.cache["locations"], self.cache["values"]

    def get_bbox(self, image_id, batch_idx):
        self._get_regression_base(image_id, batch_idx)
        return self.cache["locations"], self.cache["bbox"]
