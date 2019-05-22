import json
import os
from collections import defaultdict

import cv2
import numpy as np

from verres.utils import masking
from .config import COCODoomLoaderConfig, COCODoomStreamConfig

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
            self.index[anno["image_id"]].append(anno)
        self.num_classes = len(ENEMY_TYPES)

        print(f"Num images :", len(data["images"]))
        print(f"Num annos  :", len(data["annotations"]))
        print(f"Num classes:", self.num_classes+1)

    @classmethod
    def default_train(cls):
        cfg = COCODoomLoaderConfig("/data/Datasets/cocodoom/map-train.json",
                                   "/data/Datasets/cocodoom",
                                   stream_batch_size=32)
        return cls(cfg)

    @classmethod
    def default_val(cls):
        cfg = COCODoomLoaderConfig("/data/Datasets/cocodoom/map-val.json",
                                   "/data/Datasets/cocodoom",
                                   stream_batch_size=32)
        return cls(cfg)

    @classmethod
    def default_test(cls):
        cfg = COCODoomLoaderConfig("/data/Datasets/cocodoom/map-full-test.json",
                                   "/data/Datasets/cocodoom",
                                   stream_batch_size=32)
        return cls(cfg)

    @property
    def steps_per_epoch(self):
        return len(self.index) // self.cfg.stream_batch_size

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
