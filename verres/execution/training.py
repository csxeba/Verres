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
        self.loss_tracker = V.operation.losses.Tracker(criteria.get_output_keys())
        if config.context.debug:
            self.train_step = self._train_step_function
        else:
            self.train_step = tf.function(self._train_step_function, experimental_relax_shapes=True)
        self.model.train_step = self.train_step

    @classmethod
    def factory(cls, config: V.Config):
        model = V.architecture.VRSArchitecture.factory(config)
        criteria = V.optim.criteria.VRSCriteria(config, config.training.criteria_spec)
        scheduler = V.optim.schedule.factory(config.training.lr_schedule_spec)
        optimizer = V.optim.optimizers.factory(config, scheduler)
        return cls(config, model, criteria, optimizer, scheduler)

    def execute(self, stream=None):

        if stream is None:
            pipes = V.data.factory(self.cfg, specs=self.cfg.training.data)

            stream = streaming.get_tf_dataset(
                self.cfg,
                pipes,
                batch_size=self.cfg.training.batch_size,
                shuffle=True,
                collate="default")

        self.model.compile(optimizer=self.optimizer)
        self.model.fit(stream,
                       steps_per_epoch=self.cfg.training.steps_per_epoch,
                       epochs=self.cfg.training.epochs,
                       callbacks=callback_factory(self.cfg))

    def _train_step_function(self, batch):
        image = batch["image"]
        image = self.model.preprocess_input(image)

        with tf.GradientTape() as tape:
            prediction = self.model(image, training=True)
            losses = self.criteria(batch, prediction)

        grads = tape.gradient(losses["loss"], self.model.trainable_weights)
        grad_norm = tf.reduce_sum([tf.math.reduce_euclidean_norm(grad) for grad in grads])
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        losses["grad_norm"] = grad_norm

        return losses
