from typing import Dict

import numpy as np
import tensorflow as tf

from verres.artifactory import Artifactory
from verres.data.cocodoom import evaluation, COCODoomLoader


class ResetOptimizerState(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        self.model.optimizer = self.model.optimizer.from_config(self.model.optimizer.get_config())


class LinearLRSchedule(tf.keras.callbacks.Callback):

    def __init__(self,
                 cycle_length: int,
                 steps_per_epoch: int,
                 lr_map: Dict[int, float],
                 initial_lr: float = None):

        super().__init__()
        self.schedule = None
        self.pointer = 0
        self.cycle_length = None
        self.make_schedule(cycle_length, steps_per_epoch, lr_map, initial_lr)

    def make_schedule(self,
                      cycle_length: int,
                      steps_per_epoch: int,
                      lr_map: Dict[int, float],
                      initial_lr: float = None):

        self.cycle_length = cycle_length

        schedule = np.empty(self.cycle_length * steps_per_epoch, dtype="float32")
        if 0 not in lr_map:
            if initial_lr is None:
                raise RuntimeError("Either pass the initial learning rate in the lr_map or as a dedicated parameter!")
        else:
            lr_map = lr_map.copy()
            initial_lr = lr_map.pop(0)

        start_step = 0
        current_lr = initial_lr
        for end_epoch, next_lr in sorted(lr_map.items(), key=lambda it: it[0]):
            steps = end_epoch * steps_per_epoch - start_step
            schedule[start_step:start_step+steps] = np.linspace(
                current_lr, next_lr, num=steps, endpoint=False, dtype="float32")
            start_step += steps
            current_lr = next_lr
        schedule[start_step:] = current_lr
        self.schedule = schedule

    def on_batch_end(self, batch, logs=None):
        self.model.optimizer.lr = self.schedule[self.pointer]
        self.pointer += 1
        self.pointer %= self.cycle_length

    def on_epoch_end(self, epoch, logs=None):
        logs["lr"] = self.schedule[self.pointer]


class ObjectMAP(tf.keras.callbacks.Callback):

    def __init__(self, loader: COCODoomLoader, artifactory: Artifactory = None, checkpoint_best: bool = True):
        super().__init__()
        self.loader = loader
        if artifactory is None:
            artifactory = Artifactory.get_default()
        self.artifactory = artifactory
        self.detection_tmp = str((artifactory.detections / "detections_epoch_{epoch}.json"))
        self.checkpoint_tmp = str((artifactory.checkpoints / "chkp_epoch_{epoch}_map_{map:.4f}.h5"))
        self.last_map = -1.
        self.last_chkp = ""
        self.checkpoint_best = checkpoint_best

    def on_epoch_end(self, epoch, logs=None):
        result = evaluation.run(loader=self.loader, model=self.model, detection_file=self.detection_tmp.format(epoch=epoch+1))
        mAP = result[0]
        if self.checkpoint_best:
            if mAP > self.last_map:
                checkpoint_path = self.checkpoint_tmp.format(epoch=epoch+1, map=mAP)
                self.model.save_weights(checkpoint_path, overwrite=True)
                self.last_map = mAP
                self.last_chkp = checkpoint_path
        logs = logs or {}
        logs["mAP"] = mAP
        logs["mAR"] = result[6]
        return logs
