import numpy as np
import tensorflow as tf

from .config import COCODoomStreamConfig, TASK
from .loader import COCODoomLoader
from verres.utils import cocodoom_utils


class COCODoomSequence(tf.keras.utils.Sequence):

    def __init__(self,
                 stream_config: COCODoomStreamConfig,
                 data_loader: COCODoomLoader):

        self.cfg = stream_config
        self.loader = data_loader
        meta_iterator = cocodoom_utils.apply_filters(self.loader.image_meta.values(), stream_config, data_loader)

        self.ids = sorted(meta["id"] for meta in meta_iterator)
        self.N = len(self.ids)
        if self.N == 0:
            raise RuntimeError("No IDs left. Relax your filters!")
        self._internal_interator = self.stream()

    def steps_per_epoch(self):
        return self.N // self.cfg.batch_size

    @staticmethod
    def _configure_batch(xs, ys):
        if isinstance(xs[0], list):
            X = [np.array([x[i] for x in xs]) for i in range(len(xs[0]))]
        else:
            X = np.array(xs)
        if isinstance(ys[0], list):
            Y = [np.array([y[i] for y in ys]) for i in range(len(ys[0]))]
        else:
            Y = np.array(ys)
        return X, Y

    def make_batch(self, IDs=None):
        xs, ys = [], []

        if IDs is None:
            IDs = np.random.choice(self.ids, size=self.cfg.batch_size)

        for ID in IDs:
            x = self.loader.get_image(ID) / 255.

            if self.cfg.task == TASK.SEGMENTATION:
                y = self.loader.get_segmentation_mask(ID)
            elif self.cfg.task == TASK.DEPTH:
                y = self.loader.get_depth_image(ID)
            elif self.cfg.task == TASK.DETECTION_TRAINING:
                heatmap, refinement, wh, mask = self.loader.get_box_ground_truth(ID)
                x = [x, mask]
                y = [heatmap, refinement*mask, wh*mask]
            elif self.cfg.task == TASK.DETECTION_INFERENCE:
                heatmap, refinement, wh, mask = self.loader.get_box_ground_truth(ID)
                y = [heatmap, refinement, wh]
            else:
                assert False

            xs.append(x)
            ys.append(y)

        batch = self._configure_batch(xs, ys)

        return batch

    def stream(self):
        while 1:
            if self.cfg.shuffle:
                np.random.shuffle(self.ids)
            for batch in (self.ids[start:start+self.cfg.batch_size] for start in range(0, self.N, self.cfg.batch_size)):
                yield self.make_batch(batch)

    # Keras Sequence interface
    def __getitem__(self, item=None):
        return next(self._internal_interator)

    def __len__(self):
        return self.steps_per_epoch()

    # Python generator interface
    def __iter__(self):
        return self

    def __next__(self):
        if self._internal_interator is None:
            self._internal_interator = self.stream()
        return next(self._internal_interator)
