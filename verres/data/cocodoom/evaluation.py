import json
import os
import sys
from typing import List

import numpy as np
import tensorflow as tf
import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .loader import COCODoomLoader, ENEMY_TYPES
import verres as vrs
from verres.utils import profiling


def _convert_to_coco_format(detection, category_index, ID):
    detections = []
    for centroid, wh, type, score in zip(*detection):
        x0y0 = centroid - wh / 2.
        bbox = list(map(float, list(x0y0) + list(wh)))
        type_name = ENEMY_TYPES[int(type)]
        category = category_index.get(type_name, None)
        if category is None:
            continue
        detections.append({"bbox": bbox,
                           "category_id": category["id"],
                           "image_id": ID,
                           "score": float(score)})
    return detections


def _evaluate(detections: List[dict],
              loader: COCODoomLoader,
              detection_file: str = "default"):

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
            detections.extend(_convert_to_coco_format(detection, category_index, meta["id"]))

        postproc_time.append(timer.result)
        print("\r [Verres] - COCO eval "
              f"P: {i/len(loader.image_meta):>7.2%} "
              f"DTime: {np.mean(data_time[-10:]):.4f} "
              f"MTime: {np.mean(model_time[-10:]):.4f} "
              f"PTime: {np.mean(postproc_time[-10:]):.4f}", end="")
    print()

    return _evaluate(detections, loader, detection_file)


def run_time_priorized_detection(time_data_loader: COCODoomLoader,
                                 model: tf.keras.Model,
                                 detection_file="default"):

    def process(file_names_list):
        file_name_dataset = tf.data.Dataset.from_tensor_slices(file_names_list)
        file_name_dataset = file_name_dataset.map(tf.io.read_file)
        file_name_dataset = file_name_dataset.map(tf.io.decode_image)
        file_name_dataset = file_name_dataset.map(lambda t: tf.image.convert_image_dtype(t, tf.float32))
        return file_name_dataset

    img_root = time_data_loader.cfg.images_root
    file_names = [os.path.join(img_root, meta["file_name"])
                  for meta in time_data_loader.image_meta.values()]
    category_index = {cat["name"]: cat for cat in time_data_loader.categories.values()}

    prev_file_names = []
    for meta in time_data_loader.image_meta.values():
        prev_meta = time_data_loader.image_meta[meta["prev_image_id"]]
        prev_file_names.append(os.path.join(img_root, prev_meta["file_name"]))

    present: tf.data.Dataset = process(file_names)
    past: tf.data.Dataset = process(prev_file_names)

    IDs = tf.data.Dataset.from_tensor_slices([meta["id"] for meta in time_data_loader.image_meta.values()])

    dataset = tf.data.Dataset.zip((present, past, IDs))

    detections = []

    print()
    for i, (past, present, ID) in enumerate(dataset, start=1):
        result = model.detect((past, present))
        detections.extend(_convert_to_coco_format(result, category_index, int(ID)))
        print("\r [Verres] - COCO eval "
              f"P: {i / len(time_data_loader.image_meta):>7.2%} ")

    print()

    return _evaluate(detections, time_data_loader, detection_file)
