import dataclasses
import json
import os
from typing import List, Callable, Optional

import numpy as np
import pydantic

from .abstract import Dataset, DatasetDescriptor

import verres as V


class MSCOCOAnnotation(pydantic.BaseModel):
    bbox: List[float]
    iscrowd: bool
    category_id: int


class MSCOCODataElement(pydantic.BaseModel):
    image_id: int
    file_name: str
    width: int
    height: int
    annotations: List[MSCOCOAnnotation] = []


class MSCOCODataDescriptor(DatasetDescriptor):

    def __init__(self, data_spec: V.config.DatasetSpec):
        super().__init__()
        self.class_mapping = {"car": 0, "truck": 1, "bus": 1, "person": 2, "bicycle": 3, "motorcycle": 4}
        self.train_image_shape = None, None, 3
        self.num_classes = len(self.class_mapping)
        self.root = data_spec.root
        self.annotation_file_path = os.path.join(
            self.root,
            "annotations",
            f"instances_{data_spec.subset}2017.json",
        )


class MSCOCODataset(Dataset):

    def __init__(
        self,
        config: V.Config,
        dataset_spec: V.config.DatasetSpec,
        filters: List[Callable[[MSCOCODataElement], bool]] = None,
    ):

        descriptor = MSCOCODataDescriptor(dataset_spec)
        data = json.load(open(descriptor.annotation_file_path))
        image_index = {}
        for meta in data["images"]:
            meta["annotations"] = []
            image_index[meta["id"]] = meta
        for anno in data["annotations"]:
            image_index[anno["image_id"]]["annotations"].append(anno)
        self.index: List[MSCOCODataElement] = []
        for meta in image_index:
            data_element = MSCOCODataElement(**meta)
            if all(flt(data_element) for flt in filters):
                self.index.append(data_element)

        super().__init__(config, dataset_spec, list(range(len(self.index))), descriptor)
        self._coco_category_index = {cat["id"]: cat for cat in data["categories"]}

    def unpack(self, ID: int) -> V.data.sample.Sample:
        element = self.index[ID]

        N = len(element.annotations)
        image_shape_xy = np.array([element.height, element.width]).astype(np.float32)

        object_centers = np.empty((N, 2), dtype=np.float3)  # [N, 2], float32: [0 .. 1[
        object_types = np.empty((M,), dtype=np.int64)
        object_keypoints = np.empty((0, 2), dtype=np.float32)
        object_correspondance = np.empty((0,), dtype=np.int64)
        for annotation in element.annotations:
            coco_bbox = np.array(annotation.bbox)
            object_center

        sample = V.data.sample.Sample(
            image_id=element.image_id,
            image_path=os.path.join(self.dataset_spec.root, element.file_name),
            image_shape_wh=(element.width, element.height),
            object_centers=object_centers,
            object_keypoint_coords=object_keypoint_coords,
            object_keypoint_ids=object_keypoint_ids,
        )

        return sample

    def __len__(self):
        pass
