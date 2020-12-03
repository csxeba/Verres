import numpy as np
import tensorflow.keras as keras


class ReservoirInit(keras.initializers.Initializer):

    def __init__(self, r=0.01):
        self.r = r

    def __call__(self, shape, dtype=None):
        mask = keras.backend.cast(
            keras.backend.random_uniform(shape, dtype=dtype) < self.r, dtype)
        weight = keras.backend.random_normal(shape) * mask
        return weight


class Reservoir:

    def __init__(self, num_units=(128, 256), batch_size=16, wordsize=16, r=0.01):

        self.batch_size = batch_size
        self.wordsize = wordsize

        self.layers = []
        inputs = keras.Input(batch_shape=(batch_size, None, self.wordsize))
        echo = inputs
        for units in num_units:
            self.layers.append(keras.layers.LSTM(
                units,
                kernel_initializer=ReservoirInit(r),
                recurrent_initializer=ReservoirInit(r),
                activation=keras.backend.tanh,
                trainable=False,
                stateful=True,
                return_sequences=True))
            echo = self.layers[-1](echo)

        self.echo = keras.Model(inputs, echo)
        self.last_states_np = None

    def _store_states(self, states):
        if self.last_states_np is None:
            self.last_states_np = states
        else:
            self.last_states_np = np.concatenate([self.last_states_np, states], axis=1)

    def run_reservoir(self, activation_time=10, echo_time=90, reset_before=True):
        if reset_before:
            self.echo.reset_states()
        if activation_time:
            self.drive_reservoir(activation_time=activation_time)
        if echo_time:
            self.echo_reservoir(echo_time=echo_time)

    def reset_reservoir(self):
        self.echo.reset_states()
        self.last_states_np = None

    def drive_reservoir(self, x=None, activation_time=10):
        if x is None:
            x = np.random.randn(self.batch_size, activation_time, self.wordsize)
        states = self.echo.predict(x)
        self._store_states(states)

    def echo_reservoir(self, echo_time=90):
        x = np.zeros((self.batch_size, echo_time, self.wordsize))
        echo_states = self.echo.predict(x)
        self._store_states(echo_states)


def xperiment():
    from matplotlib import pyplot as plt

    TOTAL_XP_TIME = 10000
    BATCH_SIZE = 8
    WORD_SIZE = 1

    inputs = np.random.normal(scale=0.01, size=(BATCH_SIZE, TOTAL_XP_TIME, WORD_SIZE)).astype("float32")
    # inputs[:, 100, :] += 10
    # inputs[:, 300:400, :] += 1.
    # inputs[:, 500:600, :] -= 1.
    # inputs[:, 600:700, :] -= np.linspace(10, 0, 100, dtype="float32")[None, :, None]
    inputs[:, 0, :] += 1
    inputs[:, 400:, :] -= 1

    res = Reservoir(num_units=[1024, 256, 1024], batch_size=BATCH_SIZE, wordsize=WORD_SIZE, r=0.0019)
    res.drive_reservoir(x=inputs)
    means = np.mean(res.last_states_np, axis=-1)

    plt.figure(figsize=(16, 9))
    x = np.arange(TOTAL_XP_TIME)
    for i in range(BATCH_SIZE):
        plt.plot(x, means[i], "r-", alpha=0.1)
    plt.plot(x, (inputs[0, ..., 0] / inputs.max()) * means.max(), "b-", alpha=1)
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    xperiment()
