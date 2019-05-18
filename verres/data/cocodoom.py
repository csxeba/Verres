import os
import json
from collections import defaultdict

import numpy as np
import cv2

from ..utils import masking, colors as c


class COCODoomLoader:

    ENEMY_TYPES = [
        "POSSESSED", "SHOTGUY", "VILE", "UNDEAD", "FATSO", "CHAINGUY", "TROOP", "SERGEANT", "HEAD", "BRUISER",
        "KNIGHT", "SKULL", "SPIDER", "BABY", "CYBORG", "PAIN", "WOLFSS"
    ]

    def __init__(self, data, root, batch_size=16):

        self.root = root
        self.batch_size = batch_size

        if not isinstance(data, dict):
            data = json.load(open(data))

        self.categories = {cat["id"]: cat for cat in data["categories"]}
        self.image_meta = {meta["id"]: meta for meta in data["images"]}
        self.index = defaultdict(list)
        for anno in data["annotations"]:
            self.index[anno["image_id"]].append(anno)
        self.num_classes = len(self.ENEMY_TYPES)

        print(f"Num images :", len(data["images"]))
        print(f"Num annos  :", len(data["annotations"]))
        print(f"Num classes:", self.num_classes+1)

    @property
    def steps_per_epoch(self):
        return len(self.index) // self.batch_size

    def make_sample(self, image_id, sparse_y=True):
        meta = self.image_meta[image_id]
        image_path = os.path.join(self.root, meta["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"No image found @ {image_path}")
        if sparse_y:
            mask = self._mask_sparse(image.shape, image_id)
        else:
            mask = self._mask_dense(image.shape, image_id)
        return image, mask

    def _mask_sparse(self, image_shape, image_id):
        mask = np.zeros(image_shape[:2] + (self.num_classes + 1,))
        for anno in self.index[image_id]:
            category = self.categories[anno["category_id"]]
            if category["name"] not in self.ENEMY_TYPES:
                continue

            class_idx = self.ENEMY_TYPES.index(category["name"])
            instance_mask = masking.get_mask(anno, image_shape[:2])
            mask[..., class_idx][instance_mask] = 1

        overlaps = mask.sum(axis=2)[..., None]
        overlaps[overlaps == 0] = 1
        mask /= overlaps
        mask[..., 0] = 1 - mask[..., 1:].sum(axis=2)
        return mask

    def _mask_dense(self, image_shape, image_id):
        mask = np.zeros(image_shape[:2] + (1,))
        for anno in self.index[image_id]:
            category = self.categories[anno["category_id"]]
            if category["name"] not in self.ENEMY_TYPES:
                continue

            class_idx = self.ENEMY_TYPES.index(category["name"])
            instance_mask = masking.get_mask(anno, image_shape[:2])
            mask[instance_mask] = class_idx+1
        return mask

    def stream(self, shuffle=True, use_onehot_y=False, run_number=None, level_number=None):
        meta_iterator = self.image_meta.values()
        if run_number is not None:
            criterion = "run{}".format(run_number)
            meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)
        if level_number is not None:
            criterion = "map{}".format(level_number)
            meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)

        ids = sorted(meta["id"] for meta in meta_iterator)
        N = len(ids)
        if N == 0:
            raise RuntimeError("No IDs left. Relax your filters!")

        while 1:
            if shuffle:
                np.random.shuffle(ids)
            for batch in (ids[start:start + self.batch_size] for start in range(0, N, self.batch_size)):
                X, Y = [], []
                for ID in batch:
                    x, y = self.make_sample(ID, use_onehot_y)
                    X.append(x)
                    Y.append(y)

                yield np.array(X) / 255, np.array(Y)
