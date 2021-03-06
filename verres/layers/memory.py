import tensorflow as tf


class Memory(tf.keras.layers.Layer):

    def __init__(self, memsize=32, wordsize=48, **kwargs):
        super().__init__(**kwargs)
        self.wordsize = wordsize
        self.memsize = memsize
        self.memory = None
        self._do_reseted = True

    def call(self, weights, **kwargs):
        if self._do_reseted:
            self.reset(tf.keras.backend.shape(weights)[0])
        operation = kwargs.get("operation")
        final = kwargs.get("final", False)
        result = tf.keras.backend.zeros([tf.keras.backend.shape(weights)[0], self.wordsize])
        if operation == "read":
            result = self.read(weights)
        elif operation == "write":
            self.write(weights, content=kwargs.get("content"))
        elif operation is None:
            raise ValueError("Must pass <operation> kwarg! ('read'|'write')")
        if final:
            self._do_reseted = True
        return result

    def reset(self, batch_size):
        self.memory = tf.keras.backend.zeros((batch_size, self.memsize, self.wordsize))

    def read(self, weights):
        weights = tf.keras.backend.expand_dims(weights, 1)
        readout = tf.keras.backend.batch_dot(weights, self.memory)
        return tf.keras.backend.squeeze(readout, 1)

    def write(self, weights, content):
        weights = tf.keras.backend.expand_dims(weights, 2)
        content = tf.keras.backend.expand_dims(content, 1)
        new_memories = tf.keras.backend.batch_dot(weights, content)
        self.memory += new_memories

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.wordsize
