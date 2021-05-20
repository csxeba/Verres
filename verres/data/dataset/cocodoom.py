import json
import os
from collections import defaultdict

import verres as V
from .abstract import Dataset, DatasetDescriptor


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


class COCODoomDataDescriptor(DatasetDescriptor):

    def __init__(self, data_spec: V.config.DatasetSpec):
        super().__init__()
        self.enemy_types = [
            "POSSESSED", "SHOTGUY", "VILE", "UNDEAD", "FATSO", "CHAINGUY", "TROOP", "SERGEANT", "HEAD", "BRUISER",
            "KNIGHT", "SKULL", "SPIDER", "BABY", "CYBORG", "PAIN", "WOLFSS"]
        self.enemy_type_ids = [1, 2, 3, 5, 8, 10, 11, 12, 14, 15, 17, 18, 19, 20, 21, 22, 23]
        self.image_shape = (200, 320, 3)
        self.num_classes = len(self.enemy_type_ids)
        self.root = data_spec.root
        self.annotation_file_path = _generate_annotation_path(data_spec.kwargs["split"],
                                                              data_spec.kwargs["full"],
                                                              data_spec.subset)
        self.annotation_file_path = os.path.join(data_spec.root, self.annotation_file_path)

    def __getitem__(self, item):
        if item not in self.__dict__ or item == "self":
            raise KeyError(f"No such item in COCODoom data descriptor: {item}")
        return getattr(self, item)


class COCODoomDataset(Dataset):

    def __init__(self, config: V.Config, spec: V.config.DatasetSpec):

        descriptor = COCODoomDataDescriptor(spec)

        data = json.load(open(descriptor.annotation_file_path))

        self.image_meta = {meta["id"]: meta for meta in data["images"]}
        self.index = defaultdict(list)
        for anno in data["annotations"]:
            if anno["category_id"] not in descriptor.enemy_type_ids:
                continue
            self.index[anno["image_id"]].append(anno)

        super().__init__(config,
                         dataset_spec=spec,
                         IDs=sorted(list(self.index)),
                         descriptor=descriptor)

        print(f" [Verres.COCODoomLoader] - Num images :", len(data["images"]))
        print(f" [Verres.COCODoomLoader] - Num annos  :", len(data["annotations"]))
        print(f" [Verres.COCODoomLoader] - Num classes:", descriptor.num_classes)

    def unpack(self, ID: int) -> dict:
        image_meta = self.image_meta[ID]
        annotations = self.index[ID]

        image_path = os.path.join(self.dataset_spec.root, image_meta["file_name"])
        meta = {"bboxes": [], "segmentations": [], "types": [], "image_path": image_path, "image_id": image_meta["id"]}

        for anno in annotations:
            types = [self.descriptor["enemy_type_ids"].index(anno["category_id"])]
            meta["bboxes"].append(anno["bbox"])
            meta["segmentations"].append(anno["segmentation"])
            meta["types"].append(types)

        return meta

    def __len__(self):
        return len(self.index)
