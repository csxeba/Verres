import tensorflow as tf


class UniformRangeSearch(tf.keras.optimizers.Optimizer):

    def __init__(self, minimium=-5., maximum=5., **kwargs):
        super().__init__(**kwargs)
        self.minimum = minimium
        self.maximum = maximum
        self.best = None
        self.best_loss = 0.

    def get_updates(self, loss, params):
        loss_value = tf.keras.backend.get_value(loss)
        updates = []
        if loss_value > self.best_loss:
            self.best_loss = loss_value
            self.best = tf.keras.backend.batch_get_value(params)
        for param in params:
            updates.append(
                tf.keras.backend.batch_set_value(
                    [(param, tf.keras.backend.random_uniform(param.shape, self.minimum, self.maximum, param.dtype))]
                )
            )
        return updates

    def set_best(self, model: tf.keras.Model):
        model.set_weights(self.best)
        return model
