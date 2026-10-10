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
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=self.epoch)
        # CheckpointManager renames atomically, so a crash cannot leave a half-written checkpoint.
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, checkpoint_dir(run_dir), max_to_keep=2)

    @property
    def latest(self):
        """Newest checkpoint prefix in the run directory, or None."""
        return self.manager.latest_checkpoint

    def restore(self) -> int:
        """Restores model, optimizer state and epoch; returns the epoch to continue from (0 if none)."""
        if self.manager.latest_checkpoint is None:
            return 0
        # Restoring through the manager also restores its save counter, keeping numbering monotonic.
        self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()
        return int(self.epoch.numpy())

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.assign_add(1)  # counts completed epochs, i.e. initial_epoch when resuming
        if int(self.epoch) % self.save_every == 0:
            self.manager.save()


def restore_latest(run_dir, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> int:
    """Restores the newest checkpoint of a run, returns the epoch to continue from (0 if none)."""
    return CheckpointSaver(run_dir, model, optimizer).restore()
