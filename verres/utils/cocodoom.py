import json
import os
from typing import Tuple

import numpy as np

from verres.config import ClassMapping


def generate_class_mapped_dataset(
    orig_annotation_path: str,
    mapped_annotation_path: str,
    class_mapping: ClassMapping
):
    if os.path.exists(mapped_annotation_path):
        print(f"[Verres] - Attempted class mapping dataset, but it already exists at {mapped_annotation_path}")
        return
    data = json.load(open(orig_annotation_path, "r"))
    old_category_index = {cat["id"]: cat["name"] for cat in data["categories"]}
    new_category_stuff = [{"id": idx, "name": name} for idx, name in enumerate(class_mapping.class_order)]
    new_annotations = []
    for anno in data["annotations"]:
        old_category_name = old_category_index[anno["category_id"]]
        if old_category_name not in class_mapping.class_mapping:
            continue
        anno["category_id"] = class_mapping.coco_name_to_verres_index(old_category_name)
        new_annotations.append(anno.copy())
    data["annotations"] = new_annotations
    data["categories"] = new_category_stuff
    with open(mapped_annotation_path, "w") as handle:
        json.dump(data, handle)
    print(f"[Verres] - Class-mapped dataset dumped to {mapped_annotation_path}")


def subsample_dataset(
    orig_annotation_path: str,
    subsampled_annotation_path: str,
    num_samples_to_keep: int,
):
    data = json.load(open(orig_annotation_path, "r"))
    unique_image_ids = np.array(list({anno["image_id"] for anno in data["annotations"]}))
    if len(unique_image_ids) < num_samples_to_keep:
        raise ValueError("Supplied num_samples_to_keep is smaller than the total number of samples:"
                         f" {len(unique_image_ids)} < {num_samples_to_keep}")
    np.random.shuffle(unique_image_ids)
    selected_image_ids = unique_image_ids[:num_samples_to_keep]
    data["annotations"] = [anno for anno in data["annotations"] if anno["category_id"] in selected_image_ids]
    with open(subsampled_annotation_path, "w") as handle:
        json.dump(data, handle)
    print(f"[Verres] - Dumped subsampled ({num_samples_to_keep/len(unique_image_ids):.2%}) "
          f"dataset to {subsampled_annotation_path}")


def generate_coco_detections(
    boxes_xy_01: np.ndarray,
    types: np.ndarray,
    scores: np.ndarray,
    image_shape_xy: Tuple[int, int],
    image_id: int,
):

    detections = []

    scales = np.array(image_shape_xy)  # format: image

    xy0 = boxes_xy_01[:, :2] * scales
    xy1 = boxes_xy_01[:, 2:] * scales
    wh = xy1 - xy0
    coco_bboxes = np.concatenate([xy0, wh], axis=1)

    for bbox, category_id, score in zip(coco_bboxes, types, scores):
        detections.append({
            "image_id": int(image_id),
            "bbox": list(map(float, bbox)),
            "category_id": int(category_id),
            "score": float(score),
        })

    return detections
