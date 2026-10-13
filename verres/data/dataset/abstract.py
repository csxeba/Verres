import random
from typing import Any, List, Iterable

import verres as V
from ..sample import Sample


class Dataset:

    def __init__(self,
                 config: V.Config,
                 dataset_spec: V.config.DatasetSpec,
                 IDs: List[Any]):

        self.cfg = config
        self.dataset_spec = dataset_spec
        self.IDs = IDs

    def unpack(self, ID):
        raise NotImplementedError

    def meta_stream(self, shuffle: bool, infinite: bool) -> Iterable[Sample]:
        IDs = self.IDs.copy()
        assert len(IDs) > 0
        while 1:
            if shuffle:
                random.shuffle(IDs)
            for ID in IDs:
                yield self.unpack(ID)
            if not infinite:
                break

    def __len__(self) -> int:
        return len(self.IDs)
