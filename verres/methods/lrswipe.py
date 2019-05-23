import numpy as np

from verres.keras_engine import get_engine


def make_callback(schedule, ann_engine=None):

    engine = get_engine(ann_engine)

    class _LearningRateSwipeCallback(engine.callbacks.Callback):

        def __init__(self):
            super().__init__()
            self.current_learning_rate = schedule[0]

        def on_batch_begin(self, batch, logs=None):
            if callable(schedule):
                return schedule(batch, self.current_learning_rate)
            return schedule[batch]

    return _LearningRateSwipeCallback()


class LearningRateSwipeConfig:

    def __init__(self,
                 minimum_lrate,
                 maximum_lrate,
                 increment,
                 num_steps=None,
                 verbose=0):

        self.minimum_lrate = minimum_lrate
        self.maximum_lrate = maximum_lrate
        self.increment = increment
        self.num_steps = num_steps
        self.verbose = verbose


class LearningRateSwipe:

    def __init__(self, config: LearningRateSwipeConfig, model, training_iterator, ann_engine=None):
        self.engine = get_engine(ann_engine)
        self.cfg = config
        self.model = model
        self.training_iterator = training_iterator
        self.schedule = None
        self.num_iterations = None
        self.losses = None

    def _create_schedule(self):
        if type(self.cfg.increment) in (float, int):
            self.schedule = np.arange(self.cfg.minimum_lrate, self.cfg.maximum_lrate + self.cfg.increment, self.cfg.increment)
            self.num_iterations = len(self.schedule)
        elif callable(self.cfg.increment):
            self.schedule = self.cfg.increment
            self.num_iterations = self.cfg.num_steps
        if self.num_iterations is None:
            raise RuntimeError("Either pass a function as config.increment and config.num_steps or a scalar value!")

    def run(self, plot=True):
        swipe_callback = make_callback(
            schedule=self.schedule,
            ann_engine=self.engine
        )
        history = self.model.fit_generator(self.training_iterator,
                                           steps_per_epoch=self.num_iterations,
                                           callbacks=[swipe_callback],
                                           verbose=self.cfg.verbose)
        self.losses = history.history["loss"]
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
