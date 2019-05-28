import numpy as np

from .config import COCODoomStreamConfig, TASK
from .loader import COCODoomLoader

from verres.utils import cocodoom_utils


class COCODoomStream:

    def __init__(self,
                 stream_config: COCODoomStreamConfig,
                 data_loader: COCODoomLoader):

        self.cfg = stream_config
        self.loader = data_loader
        self._internal_iterator = None
        self.shape = (self.steps_per_epoch, 200, 320, 3)

    @property
    def steps_per_epoch(self):
        return self.loader.N // self.cfg.batch_size

    def stream(self):
        meta_iterator = cocodoom_utils.apply_image_filters(self.loader.image_meta.values(), self.cfg)
        ids = sorted(meta["id"] for meta in meta_iterator)
        N = len(ids)
        if N == 0:
            raise RuntimeError("No IDs left. Relax your filters!")

        while 1:
            if self.cfg.shuffle:
                np.random.shuffle(ids)
            for batch in (ids[start:start+self.cfg.batch_size]
                          for start in range(0, N, self.cfg.batch_size)):

                X, Y = [], []
                for ID in batch:
                    x = self.loader.get_image(ID)

                    if self.cfg.task == TASK.SEGMENTATION:
                        y = self.loader.get_segmentation_mask(ID)
                    elif self.cfg.task == TASK.DEPTH:
                        y = self.loader.get_depth_image(ID)
                    else:
                        assert False

                    X.append(x)
                    Y.append(y)

                yield np.array(X) / 255, np.array(Y)

    def __iter__(self):
        return self

    def __next__(self):
        if self._internal_iterator is None:
            self._internal_iterator = self.stream()
        return next(self._internal_iterator)
