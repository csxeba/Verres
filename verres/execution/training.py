import tensorflow as tf

import verres as V
from verres.data import streaming
from verres.architecture import VRSArchitecture
from verres.optim.criteria import VRSCriteria
from verres.optim.callbacks import factory as callback_factory


class TrainingExecutor:

    def __init__(self,
                 config: V.Config,
                 model: VRSArchitecture,
                 criteria: VRSCriteria,
                 optimizer: tf.keras.optimizers.Optimizer,
                 scheduler: tf.keras.optimizers.schedules.LearningRateSchedule):

        self.cfg = config
        self.model = model
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_tracker = V.optim.losses.Tracker(criteria.OUTPUT_KEYS)
        self.model.train_step = self.train_step

    @classmethod
    def factory(cls, config: V.Config):
        model = V.architecture.VRSArchitecture.factory(config)
        criteria = V.optim.criteria.factory(config)
        scheduler = V.optim.schedule.factory(config)
        optimizer = V.optim.optimizers.factory(config, scheduler)
        return cls(config, model, criteria, optimizer, scheduler)

    def _train_loop_custom(self, stream):
        epochs = self.cfg.training.epochs
        steps = self.cfg.training.steps_per_epoch
        if self.cfg.context.verbose:
            print(" [Verres.training] - Executing in DEBUG mode.")
        for epoch in range(1, epochs + 1):
            for i, data in enumerate(stream, start=1):
                logs = self.train_step(data)
                if self.cfg.context.verbose:
                    print("-" * 100)
                    print(f" [Verres.training] - Epoch {epoch}/{epochs} - Step {i}/{steps}")
                    for key, value in logs.items():
                        print(f" {key}: {value:.6f}")

    def execute(self, stream=None):

        if stream is None:
            pipes = V.data.factory(self.cfg, specs=self.cfg.training.data)

            stream = streaming.get_tf_dataset(
                self.cfg,
                pipes,
                batch_size=self.cfg.training.batch_size,
                shuffle=True,
                collate="default")

        if not self.cfg.context.debug:
            self.model.compile(optimizer=self.optimizer)
            self.model.fit(stream,
                           steps_per_epoch=self.cfg.training.steps_per_epoch,
                           epochs=self.cfg.training.epochs,
                           callbacks=callback_factory(self.cfg),
                           initial_epoch=self.cfg.training.initial_epoch)
        else:
            self._train_loop_custom(stream)

    # @tf.function
    def train_step(self, batch):
        image = batch["image"]
        image = self.model.preprocess_input(image)

        with tf.GradientTape() as tape:
            prediction = self.model(image, training=True)
            losses = self.criteria(batch, prediction)

        grads = tape.gradient(losses["loss"], self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return losses
