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
    def _reconfigure_batch(batch: list):
        elements = []
        for stack in zip(*batch):
            if stack[0].ndim > 2:
                elements.append(tf.convert_to_tensor(stack))
            else:
                elements.append(tf.concat(stack, axis=0))

        return tuple(elements),

    def make_batch(self, IDs=None):

        if IDs is None:
            IDs = np.random.choice(self.ids, size=self.cfg.batch_size)

        batch = []

        for i, ID in enumerate(IDs):

            features = [self.loader.get_image(ID).astype("float32") / 255.]

            if self.cfg.task == TASK.SEMSEG:
                y = self.loader.get_panoptic_masks(ID)
                features.append(y[0])
            elif self.cfg.task == TASK.PANSEG:
                iseg, sseg = self.loader.get_panoptic_masks(ID)
                heatmap = self.loader.get_object_heatmap(ID)
                locations, refinements = self.loader.get_refinements(ID, i)
                features += [heatmap, locations, refinements, iseg, sseg]
            elif self.cfg.task == TASK.DEPTH:
                y = self.loader.get_depth_image(ID)
                features.append(y)
            elif self.cfg.task == TASK.DETECTION:
                heatmap = self.loader.get_object_heatmap(ID)
                locations, rreg_values = self.loader.get_refinements(ID, i)
                _, boxx_values = self.loader.get_bbox(ID, i)
                features += [heatmap, locations, rreg_values, boxx_values]
            elif self.cfg.task == TASK.INFERENCE:
                pass
            else:
                assert False

            batch.append(features)

        return self._reconfigure_batch(batch)

    def stream(self):
        while 1:
            if self.cfg.shuffle:
                np.random.shuffle(self.ids)
            for batch in (self.ids[start:start+self.cfg.batch_size] for start in range(0, self.N, self.cfg.batch_size)):
                if len(batch) < self.cfg.batch_size:
                    break
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
