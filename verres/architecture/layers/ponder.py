"""
PonderNet base idea according to https://arxiv.org/pdf/2107.05407.pdf
Adapted to convolutional layers for Computer Vision.
"""
from typing import Optional

import tensorflow as tf
from . import block
from .. import types


class PonderConvolution(block.VRSConvolution):

    def __init__(self,
                 width: int,
                 activation: str = None,
                 batch_normalize: bool = True,
                 kernel_size: int = 3,
                 stride: int = 1,
                 initializer: str = "he_uniform",
                 maximum_ponder_steps: int = 10,
                 epsilon: float = 0.05,
                 kernel_regularizer: Optional[str] = None,
                 **kwargs):

        super().__init__(width=width,
                         activation=activation,
                         batch_normalize=batch_normalize,
                         kernel_size=kernel_size,
                         stride=stride,
                         initializer=initializer,
                         **kwargs)

        self.ponder_unit = block.VRSConvolution(width=1, activation="sigmoid", batch_normalize=False, kernel_size=1,
                                                kernel_regularizer=kwargs.get("kernel_regularizer", None))
        self.maximum_ponder_steps = maximum_ponder_steps
        self.epsilon = epsilon

        gate_kwargs = dict(width=width,
                           batch_normalize=batch_normalize,
                           kernel_size=kernel_size,
                           stride=stride,
                           kernel_regularizer=kernel_regularizer)

        self.encode_fn = block.VRSConvolution(**gate_kwargs, activation=activation)
        self.output_fn = super().call
        self.state_transition_fn = block.VRSConvolution(**gate_kwargs, activation=activation)

    def call(self, x, **kwargs):
        state_t = self.encode_fn(x, **kwargs)
        lambda_accum = tf.ones(tf.shape(state_t)[0])
        y = []
        lambda_ = []
        probs = []

        for step in range(self.maximum_ponder_steps):
            state_t = self.state_transition_fn(state_t, **kwargs)
            y_t = self.output_fn(state_t, **kwargs)
            lambda_t = tf.reduce_mean(self.ponder_unit(state_t, **kwargs), axis=(1, 2, 3))
            prob_t = lambda_t * lambda_accum

            lambda_accum = lambda_accum * (1. - lambda_t)

            y.append(y_t)
            lambda_.append(lambda_t)
            probs.append(prob_t)

        y, lambda_, probs = map(tf.convert_to_tensor, [y, lambda_, probs])

        return types.IntermediateResult(outputs={"output": y, "lambda": lambda_, "probs": probs},
                                        metrics={"avg_steps": 1. / tf.reduce_mean(lambda_)},
                                        losses={})

    def forward_if_inference(self, x, **kwargs):
        state_t = self.encode_fn(x, **kwargs)
        lambda_accum = tf.ones(tf.shape(state_t)[0])

        y_t = None
        step = 0

        for step in range(self.maximum_ponder_steps):
            state_t = self.state_transition_fn(state_t, **kwargs)
            y_t = self.output_fn(state_t, **kwargs)
            lambda_t = tf.reduce_mean(self.ponder_unit(state_t, **kwargs), axis=(1, 2, 3))
            prob_t = lambda_t * lambda_accum

            lambda_accum = lambda_accum * (1. - lambda_t)

            if prob_t > tf.random.uniform(tf.shape(prob_t)):
                break

        return types.IntermediateResult(outputs={"output": y_t[None, ...]},
                                        metrics={"steps": tf.convert_to_tensor(step)},
                                        losses={})
