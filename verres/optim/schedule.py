from typing import Dict

import numpy as np
import tensorflow as tf

import verres as V


class ConstantSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = float(learning_rate)

    def __call__(self, step):
        return self.learning_rate

    def get_config(self):
        return dict(learning_rate=self.learning_rate)


def factory(config: V.Config) -> tf.optimizers.schedules.LearningRateSchedule:

    spec = config.training.lr_schedule_spec.copy()
    name = spec.pop("name", "default")

    if name.lower() in {"default", "constant"}:
        scheduler = ConstantSchedule(float(spec["learning_rate"]))
    else:
        scheduler_type = getattr(tf.optimizers.schedules, name)
        scheduler = scheduler_type(**spec)
    if config.context.verbose > 1:
        print(f" [Verres.schedule] - Factory built: {name}")

    return scheduler


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
