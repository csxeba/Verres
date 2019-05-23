import numpy as np

import tensorflow as tf


class LearningRateSwipeCallback(tf.keras.callbacks.Callback):

    def __init__(self, schedule, loss_container):
        super().__init__()
        self.schedule = schedule
        self.loss_container = loss_container

    def on_batch_begin(self, batch, logs=None):
        return self.schedule[batch]

    def on_batch_end(self, batch, logs=None):
        self.loss_container.append(logs["loss"])


class LearningRateSwipeConfig:

    def __init__(self,
                 minimum_lrate,
                 maximum_lrate,
                 increment,
                 verbose=0):

        self.minimum_lrate = minimum_lrate
        self.maximum_lrate = maximum_lrate
        self.increment = increment
        self.verbose = verbose


class LearningRateSwipe:

    def __init__(self, config: LearningRateSwipeConfig, model, training_iterator):
        self.cfg = config
        self.model = model
        self.training_iterator = training_iterator
        self.schedule = None
        self.losses = None

    def _create_schedule(self):
        if type(self.cfg.increment) in (float, int):
            self.schedule = np.arange(self.cfg.minimum_lrate, self.cfg.maximum_lrate + self.cfg.increment, self.cfg.increment)
        elif callable(self.cfg.increment):
            self.schedule = [self.cfg.minimum_lrate]
            i = 1
            while self.schedule[-1] < self.cfg.maximum_lrate:
                self.schedule.append(self.cfg.increment(self.schedule[-1]), i)
                i += 1
                if i > 5000:
                    raise RuntimeError("Too many iterations! Check your scheduler!")

    def run(self, plot=True):
        swipe_callback = LearningRateSwipeCallback(
            schedule=self.schedule,
            loss_container=self.losses
        )
        self.model.fit_generator(self.training_iterator,
                                 steps_per_epoch=self.num_iterations,
                                 callbacks=[swipe_callback],
                                 verbose=self.cfg.verbose)
        if plot:
            self.plot()
        return self.losses

    def plot(self):
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.plot(self.schedule, self.losses, "b-")
        plt.plot(self.schedule, self.losses, "rx")
        plt.title("Learning rate swipe epxeriment")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.grid()
        plt.show()
