import json
import os
import sys

import numpy as np
import tensorflow as tf
import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .loader import COCODoomLoader, ENEMY_TYPES
import verres as vrs


def run(loader: COCODoomLoader, model, detection_file="default", verbose=1):
    detections = []
    category_index = {cat["name"]: cat for cat in loader.categories.values()}

    if verbose:
        print()
        iter_me = tqdm.tqdm(loader.image_meta.items(),
                            desc=" [Verres] - COCO eval progress",
                            total=len(loader.image_meta),
                            initial=1,
                            file=sys.stdout)
    else:
        iter_me = loader.image_meta.items()

    for i, (ID, meta) in enumerate(iter_me, start=1):
        image = loader.get_image(ID)
        image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.
        detection = model.detect(image[None, ...])
        for centroid, wh, type, score in zip(*detection):
            x0y0 = centroid.numpy() - wh.numpy() / 2.
            bbox = list(map(float, list(x0y0) + list(wh)))
            type_name = ENEMY_TYPES[int(type)]
            category = category_index.get(type_name, None)
            if category is None:
                continue
            detections.append({"bbox": bbox,
                               "category_id": category["id"],
                               "image_id": meta["id"],
                               "score": float(score)})

    if len(detections) == 0:
        print(" [Verres] - No detections were generated.")
        return np.zeros(12, dtype=float)

    if detection_file == "default":
        artifactory = vrs.artifactory.Artifactory.get_default()
        detection_file = os.path.join(artifactory.root, "OD-detections.json")

    with open(detection_file, "w") as file:
        json.dump(detections, file)

    coco = COCO(loader.cfg.data_json)
    det_coco = coco.loadRes(detection_file)

    cocoeval = COCOeval(coco, det_coco, iouType="bbox")
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()
    return cocoeval.stats
