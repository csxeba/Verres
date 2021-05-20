import tensorflow as tf


class LineSearch:

    def __init__(self, model: tf.keras.Model, loss):
        self.model = model
        self.loss = loss
        self.last_update = None
        self.last_loss = None

    def step(self, updates):
        for weight, update in zip(self.model.trainable_weights, updates):
            weight.assign_sub(update)

    def step_grad(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = tf.reduce_mean(self.loss(y, predictions))
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.last_update = gradients
        self.step(gradients)
        return loss

    def make_return(self, backprops):
        train_metrics = {"loss": self.last_loss.numpy(), "backprops": backprops}
        return train_metrics

    def learn_batch(self, X, Y):

        W = self.model.get_weights()

        if self.last_loss is None:
            self.last_loss = self.step_grad(X, Y)
            return self.make_return(backprops=1)

        self.step(self.last_update)
        new_loss = tf.reduce_mean(self.loss(Y, self.model.predict(X)))
        if new_loss < self.last_loss:
            self.last_loss = new_loss
            return self.make_return(backprops=0)

        self.model.set_weights(W)
        self.last_loss = self.step_grad(X, Y)

        return self.make_return(backprops=1)
