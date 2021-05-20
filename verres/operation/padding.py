from typing import Tuple

import numpy as np
import tensorflow as tf


def calculate_paddings(shape: Tuple[int, int], stride: int):
    pad_x = -shape[0] % stride
    pad_y = -shape[1] % stride
    return pad_x, pad_y


def calculate_padded_input_shape(input_shape: Tuple[int, int], model_stride: int):
    pad_x, pad_y = calculate_paddings(input_shape, model_stride)
    return input_shape[0] + pad_x, input_shape[1] + pad_y


def calculate_padded_output_shape(input_shape: Tuple[int, int], tensor_stride: int, model_stride: int):
    input_pad_x, input_pad_y = calculate_padded_input_shape(input_shape, model_stride)
    assert input_pad_x % tensor_stride == 0 and input_pad_y % tensor_stride == 0
    pad_x, pad_y = input_pad_x // tensor_stride, input_pad_y // tensor_stride
    return pad_x, pad_y


def pad_to_stride(tensor: tf.Tensor, model_stride: int):
    """
    :param tensor: tf.Tensor[float32]
        Batch of input image tensors, ideally already normalized.
    :param model_stride: int
        The maximum stride inside the model (eg. 8 with 3 x MaxPool2D layers).
    :returns: tf.Tensor[float32]
        The images, padded with 0s to be compatible with the model stride.
    """
    shape = tf.keras.backend.int_shape(tensor)[1:3]
    pad_x, pad_y = calculate_paddings(shape, model_stride)
    result = tf.pad(tensor, ((0, 0), (0, pad_x), (0, pad_y), (0, 0)))
    return result


def pad_output_tensor_to_stride(tensor: np.ndarray, model_stride: int, tensor_stride: int):
    """
    :param tensor:
        A ground truth tensor on output stride.
    :param model_stride:
        The maximum stride of the model. Input images are padded to be compatible with this stride.
    :param tensor_stride:
        The stride between the unpadded input and ground truth shapes.
    :return:
        The tensor, padded with the scaled difference between the model and tensor strides
    """
    shape = tf.keras.backend.int_shape(tensor)[1:3]
    input_shape_deduced = shape[0] * tensor_stride, shape[1] * tensor_stride
    input_padding_x, input_padding_y = calculate_paddings(input_shape_deduced, model_stride)
    assert input_padding_x % tensor_stride == 0 and input_padding_y % tensor_stride == 0
    pad_x, pad_y = input_padding_x // tensor_stride, input_padding_y // tensor_stride
    result = tf.pad(tensor, ((0, 0), (0, pad_x), (0, pad_y), (0, 0)))
    return result
