import tensorflow as tf
from . import block


class PonderConvolution(block.VRSConvolution):

    def __init__(self,
                 width: int,
                 recurrent_width: int,
                 activation: str = None,
                 recurrent_activation: str = None,
                 batch_normalize: bool = True,
                 kernel_size: int = 3,
                 stride: int = 1,
                 initializer: str = "he_uniform",
                 maximum_ponder_steps: int = 10):

        super().__init__(width=width,
                         activation=activation,
                         batch_normalize=batch_normalize,
                         kernel_size=kernel_size,
                         stride=stride,
                         initializer=initializer)

        self.ponder_unit = block.VRSConvolution(width=1, activation="linear", batch_normalize=False, kernel_size=1)
        self.maximum_ponder_steps = maximum_ponder_steps

        gate_kwargs = dict(width=width,
                           batch_normalize=batch_normalize,
                           kernel_size=kernel_size,
                           stride=stride)

        self.encode_fn = block.VRSConvolution(**gate_kwargs, activation=activation)
        self.output_fn = block.VRSConvolution(**gate_kwargs, activation=activation)
        self.state_transition_fn = block.VRSConvolution(**gate_kwargs, activation=activation)

    def _step_transition_net(self, state, training=None, mask=None):
        """
        INPUTS:
        x: [m, ix, iy, input_c]
        state_t-1: [m, ox, oy, state_c]

        OUTPUTS:
        y_t: [m, ox, oy, output_c]
        state_t: [m, ox, oy, state_c]
        lambda_t: [m]
        """

        state = self.state_transition_fn(state)
        y_t = self.output_fn(state)
        lambda_t = tf.reduce_mean(self.ponder_unit(state), axis=(1, 2, 3))

        return y_t, state, lambda_t

    def call(self, x, training=None, mask=None):
        state_t = self.encode_fn(x, training=training, mask=mask)
        y_t, state_t, lambda_t = tf.scan(self._step_transition_net,
                                         (state_t,))
