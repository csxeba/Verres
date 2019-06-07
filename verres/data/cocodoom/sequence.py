import numpy as np
import tensorflow as tf

from .config import COCODoomStreamConfig, TASK
from .loader import COCODoomLoader


class COCODoomSequence(tf.keras.utils.Sequence):

    def __init__(self,
                 stream_config: COCODoomStreamConfig,
                 data_loader: COCODoomLoader):

        self.cfg = stream_config
        self.loader = data_loader
        meta_iterator = self.loader.image_meta.values()
        if self.cfg.run_number is not None:
            criterion = "run{}".format(self.cfg.run_number)
            meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)
        if self.cfg.level_number is not None:
            criterion = "map{}".format(self.cfg.level_number)
            meta_iterator = filter(lambda meta: criterion in meta["file_name"], meta_iterator)

        self.ids = sorted(meta["id"] for meta in meta_iterator)
        self.N = len(self.ids)
        if self.N == 0:
            raise RuntimeError("No IDs left. Relax your filters!")
        self._internal_interator = self.stream()

    def __len__(self):
        return self.N // self.cfg.batch_size

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
                masks = []
                for ID in batch:
                    x = self.loader.get_image(ID)

                    if self.cfg.task == TASK.SEGMENTATION:
                        y = self.loader.get_segmentation_mask(ID)
                    elif self.cfg.task == TASK.DEPTH:
                        y = self.loader.get_depth_image(ID)
                    elif self.cfg.task in (TASK.DETECTION_TRAINING, TASK.DETECTION_INFERENCE):
                        y, mask = self.loader.get_box_ground_truth(ID)
                        masks.append(mask)
                    else:
                        assert False

                    X.append(x)
                    Y.append(y)

                X = np.array(X) / 255.
                Y = np.array(Y)

                if self.cfg.task in (TASK.DETECTION_TRAINING, TASK.DETECTION_INFERENCE):
                    masks = np.array(masks)
                    X = [X, masks]

                yield X, Y

    def __getitem__(self, item=None):
        return next(self._internal_interator)

    def on_epoch_end(self):
        np.random.shuffle(self.ids)
