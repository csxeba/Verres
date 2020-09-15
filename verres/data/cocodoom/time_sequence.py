import numpy as np

from .config import COCODoomStreamConfig
from .loader import COCODoomLoader
from .sequence import COCODoomSequence

from verres.utils import cocodoom_utils


class COCODoomTimeSequence(COCODoomSequence):

    def __init__(self, stream_config: COCODoomStreamConfig, data_loader: COCODoomLoader,
                 full_data_loader: COCODoomLoader):

        super().__init__(stream_config, full_data_loader)
        self.small_loader = data_loader
        self.mapping = {}
        self._prepare()

    def _prepare(self):
        new_ids = []
        meta_stream = cocodoom_utils.apply_filters(
            self.small_loader.image_meta.values(), self.cfg, self.small_loader)
        for meta in meta_stream:
            ID = meta["id"]
            prev_meta = self.loader.image_meta.get(ID-1, None)
            if prev_meta is None:
                continue
            self.mapping[ID] = prev_meta["id"], meta["id"]
            new_ids.append(ID)
        self.ids = np.array(new_ids)

    def make_sample(self, ID, batch_index: int):

        features = []

        for actual_id in self.mapping[ID]:
            local_features = super().make_sample(actual_id, batch_index)
            features.extend(local_features)

        return features
