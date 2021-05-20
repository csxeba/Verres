import random
from typing import Any, Dict, List

import verres as V


class DatasetDescriptor:

    def __init__(self):
        self.annotation_file_path: str = ""

    def __getitem__(self, item):
        raise NotImplementedError


class Dataset:

    def __init__(self,
                 config: V.Config,
                 dataset_spec: V.config.DatasetSpec,
                 IDs: List[Any],
                 descriptor: DatasetDescriptor):

        self.cfg = config
        self.dataset_spec = dataset_spec
        self.IDs = IDs
        self.descriptor = descriptor

    def unpack(self, ID):
        raise NotImplementedError

    def meta_stream(self, shuffle: bool, infinite: bool) -> dict:
        IDs = self.IDs.copy()
        while 1:
            if shuffle:
                random.shuffle(IDs)
            for ID in IDs:
                yield self.unpack(ID)
            if not infinite:
                break

    def __len__(self):
        raise NotImplementedError
