import json
import os
from collections import defaultdict

import numpy as np
import cv2

from verres.utils import masking
from .config import COCODoomLoaderConfig

ENEMY_TYPES = [
    "POSSESSED", "SHOTGUY", "VILE", "UNDEAD", "FATSO", "CHAINGUY", "TROOP", "SERGEANT", "HEAD", "BRUISER",
    "KNIGHT", "SKULL", "SPIDER", "BABY", "CYBORG", "PAIN", "WOLFSS"
]


class COCODoomLoader:

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

    def get_image(self, image_id):
        meta = self.image_meta[image_id]
        image_path = os.path.join(self.cfg.images_root, meta["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"No image found @ {image_path}")
        return image

    def get_panoptic_masks(self, image_id):
        meta = self.image_meta[image_id]
        image_shape = [meta["height"], meta["width"]]
        segmentation_mask = np.zeros(image_shape + [1])
        coord_template = np.stack(
            np.meshgrid(
                np.arange(image_shape[1]),
                np.arange(image_shape[0])),
            axis=-1)
        instance_canvas = np.zeros_like(coord_template, dtype="float32")

        for anno in self.index[image_id]:
            category = self.categories[anno["category_id"]]
            if category["name"] not in ENEMY_TYPES:
                continue

            class_idx = ENEMY_TYPES.index(category["name"])

            instance_mask = masking.get_mask(anno, image_shape)
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
        meta = self.image_meta[image_id]
        tensor_shape = np.array([meta["height"], meta["width"]]) // self.cfg.stride
        heatmap = np.zeros(list(tensor_shape) + [self.num_classes], dtype="float32")

        hit = 0
        for anno in self.index[image_id]:
            category = self.categories[anno["category_id"]]
            if category["name"] not in ENEMY_TYPES:
                continue

            hit = 1
            class_idx = ENEMY_TYPES.index(category["name"])
            box = np.array(anno["bbox"]) / self.cfg.stride
            centroid = box[:2] + box[2:] / 2
            centroid_rounded = np.clip(np.round(centroid).astype(int), [0, 0], tensor_shape[::-1]-1)
            heatmap[centroid_rounded[1], centroid_rounded[0], class_idx] = 1

        if hit:
            kernel_size = 5
            heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0, borderType=cv2.BORDER_CONSTANT)
            heatmap /= heatmap.max()

        return heatmap

    def _get_regression_base(self, image_id, batch_idx):
        if image_id == self.cache_id:
            return

        meta = self.image_meta[image_id]
        tensor_shape = np.array([meta["height"], meta["width"]]) // self.cfg.stride
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
            centroid = box[:2] + box[2:] / 2

            centroid_floored = np.floor(centroid).astype(int)
            augmented_coords = np.stack([
                centroid_floored, centroid_floored + _01, centroid_floored + _10, centroid_floored + 1
            ], axis=0)

            in_frame = np.all([augmented_coords >= 0, augmented_coords < tensor_shape[::-1][None, :]], axis=(0, 2))
            augmented_coords = augmented_coords[in_frame]
            augmented_locations = np.concatenate([
                np.full((len(augmented_coords), 1), batch_idx, dtype=augmented_coords.dtype),
                augmented_coords[:, ::-1],
                np.full((len(augmented_coords), 1), class_idx, dtype=augmented_coords.dtype)
            ], axis=1)
            augmented_values = centroid[None, :] - augmented_coords
            augmented_boxes = np.stack([box[2:]]*4, axis=0)[in_frame]

            locations.append(augmented_locations)
            values.append(augmented_values)
            bbox.append(augmented_boxes)

        self.cache_id = image_id
        self.cache["locations"] = np.concatenate(locations).astype(int)
        self.cache["values"] = np.concatenate(values).astype("float32")
        self.cache["bbox"] = np.concatenate(bbox).astype("float32")

    def get_refinements(self, image_id, batch_idx):
        self._get_regression_base(image_id, batch_idx)
        return self.cache["locations"], self.cache["values"]

    def get_bbox(self, image_id, batch_idx):
        self._get_regression_base(image_id, batch_idx)
        return self.cache["locations"], self.cache["bbox"]
