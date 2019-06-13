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

        print(f"Num images :", len(data["images"]))
        print(f"Num annos  :", len(data["annotations"]))
        print(f"Num classes:", self.num_classes+1)

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

    def get_segmentation_mask(self, image_id):
        meta = self.image_meta[image_id]
        image_shape = [meta["height"], meta["width"]]
        mask = np.zeros(image_shape + [1])
        for anno in self.index[image_id]:
            category = self.categories[anno["category_id"]]
            if category["name"] not in ENEMY_TYPES:
                continue

            class_idx = ENEMY_TYPES.index(category["name"])
            instance_mask = masking.get_mask(anno, image_shape)
            mask[instance_mask] = class_idx+1
        return mask

    def get_depth_image(self, image_id):
        meta = self.image_meta[image_id]
        depth_image_path = meta["file_path"].replace("/rgb/", "/depth/")
        depth_image = cv2.imread(depth_image_path)
        return depth_image

    def get_box_ground_truth(self, image_id):
        meta = self.image_meta[image_id]
        tensor_shape = [meta["height"] // self.cfg.stride, meta["width"] // self.cfg.stride]
        heatmap = np.zeros(tensor_shape + [len(ENEMY_TYPES)])
        refinements = np.zeros(tensor_shape + [2])
        wh = np.zeros(tensor_shape + [2])
        mask = np.zeros(tensor_shape + [1])

        hit = 0
        for anno in self.index[image_id]:
            category = self.categories[anno["category_id"]]
            if category["name"] not in ENEMY_TYPES:
                continue

            hit = 1
            class_idx = ENEMY_TYPES.index(category["name"])
            box = np.array(anno["bbox"]) / self.cfg.stride
            centroid = box[:2] + box[2:] / 2
            centroid_rounded = np.floor(centroid).astype(int)
            refinement = centroid - centroid_rounded

            heatmap[centroid_rounded[1], centroid_rounded[0], class_idx] = 1
            refinements[centroid_rounded[1], centroid_rounded[0]] = refinement
            wh[centroid_rounded[1], centroid_rounded[0]] = box[2:] / 2
            mask[centroid_rounded[1], centroid_rounded[0]] = 1

        mask = np.concatenate([mask]*2, axis=-1)

        if hit:
            kernel_size = 3
            heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0, borderType=cv2.BORDER_CONSTANT)
            heatmap /= heatmap.max()
            # mask = filters.gaussian(mask, mode="constant", cval=0, multichannel=True)

        return heatmap, refinements, wh, mask
