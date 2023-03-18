import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

import verres as V

from .abstract import Dataset
from ..sample import Input, Label, Sample
from .. import filters


def _generate_annotation_path(split: str, full: bool, subset: str):
    if split not in ["map", "run"]:
        raise RuntimeError('COCODoom split must either be one of "map", "run".')
    if subset not in ["train", "val", "test", "val-mini"]:
        raise RuntimeError('COCODoom subset must either be on of "train", "val", "test".')
    elements = [split]
    if full:
        elements.append("full")
    elements.append(subset)
    return "-".join(elements) + ".json"


def _get_map_number(image_meta):
    image_path = image_meta["file_name"]
    map_no = [int(mapno[3:]) for mapno in image_path.split("/") if "map" in mapno][0]
    return map_no


def _image_meta_to_sample_object(
    image_meta: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    spec: V.config.DatasetSpec,
) -> Sample:

    image_path = os.path.join(spec.root, image_meta["file_name"])
    object_centers = []
    object_keypoints = []
    object_types = []
    object_segmentations = []
    map_no = [int(mapno[3:]) for mapno in image_path.split("/") if "map" in mapno][0]

    for anno in annotations:
        object_centers.append([
            anno["bbox"][0] + anno["bbox"][2] / 2,
            anno["bbox"][1] + anno["bbox"][3] / 2,
        ])
        object_keypoints.append([
            anno["bbox"][0],
            anno["bbox"][1],
            anno["bbox"][0] + anno["bbox"][2],
            anno["bbox"][1] + anno["bbox"][3],
        ])
        object_types.append(anno["verres_type_id"])
        object_segmentations.append(anno["segmentation"])

    object_centers_np = np.array(object_centers) if object_centers else np.zeros((0, 2), dtype=float)
    object_keypoints_np = np.array(object_keypoints) if object_centers else np.zeros((0, 4), dtype=float)
    object_types_np = np.array(object_types) if object_centers else np.zeros((0,), dtype=int)

    inputs = Input(image_path=image_path, shape_whc=(image_meta["width"], image_meta["height"], 3))

    image_shape_np = np.array([image_meta["width"], image_meta["height"]])
    label = Label(
        object_centers=object_centers_np / image_shape_np[None, :],
        object_types=object_types_np,
        object_scores=None,
        object_keypoint_coords=object_keypoints_np / np.concatenate([image_shape_np]*2)[None, :],
        segmentation_repr=object_segmentations,
    )

    sample = Sample.create(
        ID=image_meta["id"],
        input_=inputs,
        label=label,
        metadata={"map_no": map_no, "num_objects": len(object_centers_np)}
    )

    return sample


class COCODoomDataset(Dataset):

    def __init__(self, config: V.Config, spec: V.config.DatasetSpec):
        self.annotation_file_path = os.path.join(
            spec.root,
            _generate_annotation_path(
                spec.kwargs["split"],
                spec.kwargs["full"],
                spec.subset,
        ))
        data = json.load(open(self.annotation_file_path))

        category_index = {cat["id"]: cat for cat in data["categories"]}

        image_id_to_anno_list = defaultdict(list)
        object_level_filters = filters.factory(spec.object_level_filters)
        num_valid_annos = 0
        for anno in data["annotations"]:
            category_meta = category_index[anno["category_id"]]
            if category_meta["name"] not in config.class_mapping:
                continue  # Skip unmapped classes
            if not all(filt(anno) for filt in object_level_filters):
                continue
            anno["verres_type_id"] = config.class_mapping.map_name_to_index(category_meta['name'])
            image_id_to_anno_list[anno["image_id"]].append(anno)
            num_valid_annos += 1

        image_level_filters = filters.factory(spec.image_level_filters)
        self.index: Dict[int, Sample] = {}
        valid_ids = []
        for meta in data["images"]:
            annos = image_id_to_anno_list[meta["id"]]
            sample = _image_meta_to_sample_object(meta, annos, spec)
            self.index[meta["id"]] = sample
            if all(filt(sample) for filt in image_level_filters):
                valid_ids.append(meta["id"])

        super().__init__(config,
                         dataset_spec=spec,
                         IDs=valid_ids)

        self.class_mapping = config.class_mapping
        print(f" [Verres.COCODoomLoader] - Loaded", self.annotation_file_path)
        print(f" [Verres.COCODoomLoader] - Num images :", len(valid_ids))
        print(f" [Verres.COCODoomLoader] - Num annos  :", num_valid_annos)
        print(f" [Verres.COCODoomLoader] - Num classes:", config.class_mapping.num_classes)

    def unpack(self, ID: int) -> Sample:
        return self.index[ID]
