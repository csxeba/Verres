import numpy as np

from .config import COCODoomStreamConfig, TASK
from .loader import COCODoomLoader


class COCODoomStream:

    def __init__(self,
                 stream_config: COCODoomStreamConfig,
                 data_loader: COCODoomLoader):

        self.cfg = stream_config
        self.loader = data_loader
        self._internal_iterator = None

    def steps_per_epoch(self):
        return self.loader.N // self.cfg.batch_size

    def stream(self):
        meta_iterator = self.loader.image_meta.values()
        if self.cfg.run_number is not None:
            criterion = "run{}".format(self.cfg.run_number)
            meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)
        if self.cfg.level_number is not None:
            criterion = "map{}".format(self.cfg.level_number)
            meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)

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
