import os

import tensorflow as tf


def checkpoint_dir(run_dir) -> str:
    """Directory holding the tensorflow checkpoints of a single run."""
    return os.path.join(str(run_dir), "checkpoints", "tf_ckpt")


class CheckpointSaver(tf.keras.callbacks.Callback):
    """Saves model, optimizer state and epoch so an interrupted run can be resumed exactly."""

    def __init__(self,
                 run_dir,
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 save_every: int = 1):

        super().__init__()
        self.save_every = save_every
        self.optimizer = optimizer
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        # Keras 3 hides layers kept in a plain list from `tf.train.Checkpoint`, which would then save
        # no weights at all, so the model's variables are tracked by position instead.
        self.model_variables = list(model.weights)
        if not self.model_variables:
            raise RuntimeError("CheckpointSaver needs a built model: call it once before saving.")
        self.checkpoint = tf.train.Checkpoint(
            model_variables=self.model_variables, optimizer=optimizer, epoch=self.epoch)
        # CheckpointManager renames atomically, so a crash cannot leave a half-written checkpoint.
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, checkpoint_dir(run_dir), max_to_keep=2)

    @property
    def latest(self):
        """Newest checkpoint prefix in the run directory, or None."""
        return self.manager.latest_checkpoint

    def restore(self) -> int:
        """Restores model, optimizer state and epoch; returns the epoch to continue from (0 if none)."""
        prefix = self.manager.latest_checkpoint
        if prefix is None:
            return 0
        tensors = tf.train.list_variables(prefix)
        saved_model_variables = [name for name, _ in tensors if name.startswith("model_variables/")]
        if len(saved_model_variables) != len(self.model_variables):
            raise RuntimeError(
                f"Checkpoint {prefix} holds {len(saved_model_variables)} model variables while the "
                f"model has {len(self.model_variables)}. A checkpoint written before the weights were "
                "tracked cannot be resumed exactly: delete `checkpoints/tf_ckpt` to restart from "
                "`latest.weights.h5`, or resume with the config the run was started from.")
        # A Keras 3 optimizer also tracks the model's own variables, which are not part of the state
        # it restores positionally; only its slots and step counter have to line up with the run's.
        saved_variables = sum(
            1 for name, _ in tensors
            if name.startswith("optimizer/") and not name.startswith("optimizer/_trainable_variables/"))
        if saved_variables != len(self.optimizer.variables):
            raise RuntimeError(
                f"Checkpoint {prefix} holds {saved_variables} optimizer variables while the optimizer "
                f"has {len(self.optimizer.variables)}. Build the optimizer with "
                "`optimizer.build(model.trainable_variables)` on the model being resumed, otherwise its "
                "slots and step counter are restored into the wrong variables.")
        # Restoring through the manager also restores its save counter, keeping numbering monotonic.
        self.checkpoint.restore(prefix).expect_partial()
        return int(self.epoch.numpy())

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.assign_add(1)  # counts completed epochs, i.e. initial_epoch when resuming
        if int(self.epoch) % self.save_every == 0:
            self.manager.save()


def restore_latest(run_dir, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> int:
    """Restores the newest checkpoint of a run, returns the epoch to continue from (0 if none)."""
    return CheckpointSaver(run_dir, model, optimizer).restore()
