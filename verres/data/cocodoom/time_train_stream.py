from .config import StreamConfig
from verres.data.dataset.cocodoom import COCODoomDataset
from .train_stream import Stream


class COCODoomTimeSequence(Stream):

    def __init__(self,
                 stream_config: StreamConfig,
                 time_data_loader: COCODoomDataset):

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
