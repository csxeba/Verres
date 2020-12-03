import tensorflow as tf


def get_activation(activation, as_layer=True):
    if as_layer:
        return {"leakyrelu": tf.keras.layers.LeakyReLU,
                "prelu": lambda: tf.keras.layers.PReLU(shared_axes=(1, 2))
                }.get(activation, lambda: tf.keras.layers.Activation(activation))()

    else:
        act_fn = getattr(tf.nn, activation, None)
        if act_fn is None:
            if activation == "leakyrelu":
                return tf.nn.leaky_relu
            else:
                raise NotImplementedError("Activation function not found for {}".format(activation))
