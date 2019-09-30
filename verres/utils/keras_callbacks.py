import tensorflow as tf


class ResetOptimizerState(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        self.model.optimizer = self.model.optimizer.from_config(self.model.optimizer.get_config())
