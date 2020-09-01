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
from verres.utils import profiling


def run_detection(loader: COCODoomLoader, model, detection_file="default"):
    detections = []
    category_index = {cat["name"]: cat for cat in loader.categories.values()}

    data_time = []
    model_time = []
    postproc_time = []
    timer = profiling.Timer()
    for i, (ID, meta) in enumerate(loader.image_meta.items(), start=1):

        with timer:
            image = loader.get_image(ID)
            image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.
        data_time.append(timer.result)
        with timer:
            detection = model.detect(image[None, ...])
            detection = tuple(map(lambda t: t[-128:].numpy(), detection))
        model_time.append(timer.result)

        with timer:
            for centroid, wh, type, score in zip(*detection):
                x0y0 = centroid - wh / 2.
                bbox = list(map(float, list(x0y0) + list(wh)))
                type_name = ENEMY_TYPES[int(type)]
                category = category_index.get(type_name, None)
                if category is None:
                    continue
                detections.append({"bbox": bbox,
                                   "category_id": category["id"],
                                   "image_id": meta["id"],
                                   "score": float(score)})
        postproc_time.append(timer.result)
        print("\r [Verres] - COCO eval "
              f"P: {i/len(loader.image_meta):>7.2%} "
              f"DTime: {np.mean(data_time[-10:]):.4f} "
              f"MTime: {np.mean(model_time[-10:]):.4f} "
              f"PTime: {np.mean(postproc_time[-10:]):.4f}", end="")
    print()

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
