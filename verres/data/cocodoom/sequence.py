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
        np.random.shuffle(self.ids)

    def __len__(self):
        return self.N // self.cfg.batch_size

    def __getitem__(self, item):
        ID = self.ids[item]
        x = self.loader.get_image(ID)

        if self.cfg.task == TASK.SEGMENTATION:
            y = self.loader.get_segmentation_mask(ID)
        elif self.cfg.task == TASK.DEPTH:
            y = self.loader.get_depth_image(ID)
        else:
            assert False

    def on_epoch_end(self):
        np.random.shuffle(self.ids)
