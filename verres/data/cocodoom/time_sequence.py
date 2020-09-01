import numpy as np

from .config import COCODoomStreamConfig, TASK
from .loader import COCODoomLoader
from .sequence import COCODoomSequence

from verres.utils import cocodoom_utils


class COCODoomTimeSequence(COCODoomSequence):

    def __init__(self, stream_config: COCODoomStreamConfig, data_loader: COCODoomLoader,
                 full_data_loader: COCODoomLoader):

        super().__init__(stream_config, data_loader)
        self.full_loader = full_data_loader
        self.mapping = {}

    def _prepare(self):
        new_ids = []
        for meta in cocodoom_utils.apply_filters(self.base_loader.index.values(), self.cfg, self.loader):
            ID = meta["id"]
            run_no, map_no, frame_no = cocodoom_utils.deconstruct_path(meta["file_name"])
            prev_meta = self.full_loader.index[ID-1]
            prev_run_no, prev_map_no, prev_frame_no = cocodoom_utils.deconstruct_path(prev_meta["file_name"])
            if prev_map_no != map_no:
                continue

            self.mapping[ID] = prev_meta["id"], meta["id"]
            new_ids.append(ID)
        self.ids = np.array(sorted(new_ids))

    def make_batch(self, IDs=None):

        if self.cfg.task != TASK.DETECTION:
            raise NotImplementedError

        if IDs is None:
            IDs = np.random.choice(self.ids, size=self.cfg.batch_size)

        batch = []

        for i, ids in enumerate(IDs):

            features = []

            for ID in ids:
                image = self.loader.get_image(ID).astype("float32") / 255.
                heatmap = self.loader.get_object_heatmap(ID)
                locations, rreg_values = self.loader.get_refinements(ID, i)
                _, boxx_values = self.loader.get_bbox(ID, i)
                features += [image, heatmap, locations, rreg_values, boxx_values]

            batch.append(features)

        return self._reconfigure_batch(batch)
