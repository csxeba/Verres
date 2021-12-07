from typing import Tuple

import numpy as np
import tensorflow as tf

import verres as V

enemy_type_ids = [1, 2, 3, 5, 8, 10, 11, 12, 14, 15, 17, 18, 19, 20, 21, 22, 23]


def convert_type_id(type_id):
    return enemy_type_ids[type_id]


def generate_coco_detections(boxes: tf.Tensor,
                             types: tf.Tensor,
                             scores: tf.Tensor,
                             image_shape: Tuple[int, int],
                             image_id: int):

    detections = []

    scales = np.array(image_shape)  # format: image

    xy = boxes[:, :2] * scales
    wh = boxes[:, 2:] * scales
    half_wh = wh / 2.  # format: image
    x0y0 = xy - half_wh  # format: image
    coco_bboxes = np.concatenate([x0y0[:, ::-1], wh[:, ::-1]], axis=1)

    for bbox, category_id, score in zip(coco_bboxes, types.numpy(), scores.numpy()):

        category_id = V.utils.cocodoom.enemy_type_ids[category_id]

        detections.append({"image_id": int(image_id),
                           "bbox": list(map(float, bbox)),
                           "category_id": int(category_id),
                           "score": float(score)})

    return detections
