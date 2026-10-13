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
    os.makedirs(os.path.dirname(mapped_annotation_path), exist_ok=True)
    with open(mapped_annotation_path, "w") as handle:
        json.dump(data, handle)
    print(f"[Verres] - Class-mapped dataset dumped to {mapped_annotation_path}")


def generate_mini_dataset(
    orig_annotation_path: str,
    mini_annotation_path: str,
    num_images: int = 500,
):
    """Deterministic evenly-strided subset of a COCO annotation file, written next to the source."""
    if os.path.exists(mini_annotation_path):
        return
    data = json.load(open(orig_annotation_path, "r"))
    images = data["images"]
    stride = max(1, len(images) // num_images)
    kept_ids = {meta["id"] for meta in images[::stride][:num_images]}
    data["images"] = [meta for meta in images if meta["id"] in kept_ids]
    data["annotations"] = [a for a in data["annotations"] if a["image_id"] in kept_ids]
    # Written through a temp file: a crash mid-dump must not leave a truncated subset behind.
    temporary_path = mini_annotation_path + ".tmp"
    with open(temporary_path, "w") as handle:
        json.dump(data, handle)
    os.replace(temporary_path, mini_annotation_path)
    print(f"[Verres] - Mini subset ({len(data['images'])} images) dumped to {mini_annotation_path}")


def generate_coco_detections(
    boxes_xy_01: np.ndarray,
    types: np.ndarray,
    scores: np.ndarray,
    image_shape_xy: Tuple[int, int],
    image_id: int,
    top_k: int = 100,
):

    detections = []
    boxes_xy_01, types, scores = np.asarray(boxes_xy_01), np.asarray(types), np.asarray(scores)

    # COCOeval is quadratic in detections per image; keep only the highest scoring ones.
    if top_k is not None and len(scores) > top_k:
        keep = np.argsort(scores)[::-1][:top_k]
        boxes_xy_01, types, scores = boxes_xy_01[keep], types[keep], scores[keep]

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
