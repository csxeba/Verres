import tensorflow as tf


def get_activation(activation):
    return {"leakyrelu": tf.keras.layers.LeakyReLU,
            "prelu": lambda: tf.keras.layers.PReLU(shared_axes=(1, 2))
            }.get(activation, lambda: tf.keras.layers.Activation(activation))()
