import time
from typing import List

import tensorflow as tf
import yaml

import verres as V
from verres.data import streaming
from verres.architecture import VRSArchitecture
from verres.optim.criteria import VRSCriteria
from verres.optim.callbacks import factory as callback_factory
from verres.optim import checkpoint


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
        self.model.train_step = self.train_step

    @classmethod
    def factory(cls, config: V.Config):
        model = V.architecture.VRSArchitecture.factory(config)
        criteria = V.optim.criteria.VRSCriteria(config, config.training.criteria_spec)
        scheduler = V.optim.schedule.factory(config.training.lr_schedule_spec)
        optimizer = V.optim.optimizers.factory(config, scheduler)
        return cls(config, model, criteria, optimizer, scheduler)

    def _train_loop_custom(self, stream, callbacks: List[tf.keras.callbacks.Callback]):
        epochs = self.cfg.training.epochs
        steps = self.cfg.training.steps_per_epoch
        if self.cfg.context.verbose:
            print(" [Verres.training] - Executing in DEBUG mode.")
        cbs = tf.keras.callbacks.CallbackList(callbacks, model=self.model)
        cbs.on_train_begin()
        logs = {}
        for epoch in range(1, epochs + 1):
            for i, data in enumerate(stream, start=1):
                logs = self.train_step(data)
                if self.cfg.context.verbose:
                    print("-" * 100)
                    print(f" [Verres.training] - Epoch {epoch}/{epochs} - Step {i}/{steps}")
                    for key, value in logs.items():
                        print(f" {key}: {value:.6f}")
                if i >= steps:
                    break
            cbs.on_epoch_end(epoch, logs)
        cbs.on_train_end()

    def execute(self, stream=None):

        artifactory = V.Artifactory.get_default(self.cfg)
        with open(artifactory.root / "config.yml", "w") as handle:
            yaml.dump(self.cfg.dict(), handle)

        if stream is None:
            pipes = V.data.factory(self.cfg, specs=self.cfg.training.data)

            if self.cfg.context.debug:
                stream_provider = streaming.stream
            else:
                stream_provider = streaming.get_tf_dataset

            stream = stream_provider(
                self.cfg,
                pipes,
                batch_size=self.cfg.training.batch_size,
                shuffle=True,
                collate="default")
        callbacks = callback_factory(self.cfg)
        saver = checkpoint.CheckpointSaver(artifactory.root, self.model, self.optimizer)
        callbacks.append(saver)
        if not self.cfg.context.debug:
            initial_epoch = self._resume(saver, callbacks)
            self.model.compile(optimizer=self.optimizer)
            self.model.fit(
                stream,
                steps_per_epoch=self.cfg.training.steps_per_epoch,
                epochs=self.cfg.training.epochs,
                callbacks=callbacks,
                initial_epoch=initial_epoch,
            )
        else:
            self._train_loop_custom(stream, callbacks)

    def _resume(self, saver: checkpoint.CheckpointSaver,
                callbacks: List[tf.keras.callbacks.Callback]) -> int:
        """Restores an interrupted run in place; returns the epoch to pass to `model.fit`."""
        initial_epoch = self.cfg.training.initial_epoch
        if saver.latest is None:
            return initial_epoch
        if not self.model.built:
            self.model.build((None, *self.cfg.model.input_shape_hw, 3))
        self.optimizer.build(self.model.trainable_variables)
        epoch = saver.restore()
        for callback in callbacks:
            if isinstance(callback, tf.keras.callbacks.CSVLogger):
                callback.append = True  # the interrupted run already logged earlier epochs
        print(f" [Verres.training] - Resuming from checkpoint, epoch {epoch}")
        # The log can be one epoch ahead of the checkpoint if the process died between the two.
        return max(epoch, initial_epoch)

    # @tf.function
    def train_step(self, batch):
        dbg = self.cfg.context.debug
        verbose = self.cfg.context.verbose

        times = {"prep": 0.,
                 "forw": 0.,
                 "crit": 0.,
                 "back": 0.,
                 "updt": 0.}

        if dbg:
            times["prep"] = time.time()
        image = batch["image"]
        image = self.model.preprocess_input(image)
        if dbg:
            times["prep"] = time.time() - times["prep"]
            if verbose > 1:
                print(f" [Verres.train_step] - prep: {times['prep']:.4f}")

        with tf.GradientTape() as tape:
            if dbg:
                times["forw"] = time.time()
            prediction = self.model(image, training=True)
            if dbg:
                times["forw"] = time.time() - times["forw"]
                if verbose > 1:
                    print(f" [Verres.train_step] - forw: {times['forw']:.4f}")

            if dbg:
                times["crit"] = time.time()
            losses = self.criteria(batch, prediction)
            if dbg:
                times["crit"] = time.time() - times["crit"]
                if verbose > 1:
                    print(f" [Verres.train_step] - crit: {times['crit']:.4f}")

        if dbg:
            times["back"] = time.time()
        grads = tape.gradient(losses["loss"], self.model.trainable_weights)
        if dbg:
            times["back"] = time.time() - times["back"]
            if verbose > 1:
                print(f" [Verres.train_step] - back: {times['back']:.4f}")

        if dbg:
            times["updt"] = time.time()
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        if dbg:
            times["updt"] = time.time() - times["updt"]
            if verbose > 1:
                print(f" [Verres.train_step] - updt: {times['updt']:.4f}")

        return losses
