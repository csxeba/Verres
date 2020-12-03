from .config import COCODoomStreamConfig
from .loader import COCODoomLoader
from .sequence import COCODoomSequence


class COCODoomTimeSequence(COCODoomSequence):

    def __init__(self,
                 stream_config: COCODoomStreamConfig,
                 time_data_loader: COCODoomLoader):

        super().__init__(stream_config, time_data_loader)

    def make_sample(self, ID, batch_index: int):

        features = []

        meta = self.loader.image_meta[ID]
        ID = meta["id"]
        prev_ID = meta.get("prev_image_id", None)

        for actual_id in [prev_ID, ID]:
            local_features = super().make_sample(actual_id, batch_index)
            features.extend(local_features)

        return features
