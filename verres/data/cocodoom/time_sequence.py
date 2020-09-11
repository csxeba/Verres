import numpy as np

from .config import COCODoomStreamConfig
from .loader import COCODoomLoader
from .sequence import COCODoomSequence

from verres.utils import cocodoom_utils


class COCODoomTimeSequence(COCODoomSequence):

    def __init__(self, stream_config: COCODoomStreamConfig, data_loader: COCODoomLoader,
                 full_data_loader: COCODoomLoader):

        super().__init__(stream_config, data_loader)
        self.full_loader = full_data_loader
        self.mapping = {}
        self._prepare()

    def _prepare(self):
        new_ids = []
        for ID in self.ids:
            meta = self.loader.index[ID]
            run_no, map_no, frame_no = cocodoom_utils.deconstruct_path(meta["file_name"])
            prev_meta = self.full_loader.index[ID-1]
            prev_run_no, prev_map_no, prev_frame_no = cocodoom_utils.deconstruct_path(prev_meta["file_name"])
            if prev_map_no != map_no:
                continue

            self.mapping[ID] = prev_meta["id"], meta["id"]
            new_ids.append(ID)

    def make_sample(self, ID, batch_index: int):

        features = []

        for actual_id in self.mapping[ID]:
            local_features = super().make_sample(actual_id, batch_index)
            features.extend(local_features)

        return features
