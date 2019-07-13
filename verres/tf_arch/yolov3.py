import tensorflow as tf


_convlayer_counter = 1
_batchnorm_counter = 1
_leakyrelu_counter = 1
FORMAT = "channels_last"


def _conv(x, filter_num, kernel=3, stride=1, activate=True):
    global _convlayer_counter, _batchnorm_counter, _leakyrelu_counter
    batchnorm_axis = -1 if FORMAT == "channels_last" else 1
    x = tf.keras.layers.Conv2D(filter_num, kernel, use_bias=False, strides=stride, padding="same",
                               data_format=FORMAT, name="conv2d_{}".format(_convlayer_counter))(x)
    _convlayer_counter += 1

    x = tf.keras.layers.BatchNormalization(axis=batchnorm_axis,
                                           name="batch_normalization_{}".format(_batchnorm_counter))(x)
    _batchnorm_counter += 1
    if activate:
        x = tf.keras.layers.LeakyReLU(alpha=0.1, name="leaky_re_lu_{}".format(_leakyrelu_counter))(x)
        _leakyrelu_counter += 1
    return x


def _conv_output(x):
    global _convlayer_counter
    x = tf.keras.layers.Conv2D(255, 1, data_format=FORMAT, name="conv2d_{}".format(_convlayer_counter))(x)
    _convlayer_counter += 1
    return x


def _skippy_block(x, filters1, filters2):
    x1 = _conv(x, filters1, kernel=1)
    x1 = _conv(x1, filters2, kernel=3)
    return tf.keras.layers.add([x, x1])


class YOLOv3:

    def __init__(self, input_shape=(None, None, 3), batch_size=None, weights=None):

        if any([input_shape[0] is not None and input_shape[0] % 32,
                input_shape[1] is not None and input_shape[1] % 32]):
            raise ValueError("Input shapes must be divisible by 32!")

        self.inputs = tf.keras.Input(input_shape, batch_size=batch_size)
        outputs = self.build()
        self.model = tf.keras.Model(self.inputs, outputs)
        if weights is not None:
            self.model.load_weights(weights, by_name=True)

    def build(self):
        concat_axis = -1 if FORMAT == "channels_last" else 1
        outputs = []

        scale_1 = _conv(self.inputs, 32)  # 1

        x = _conv(scale_1, 64, stride=2)  # 2

        scale_2 = _skippy_block(x, 32, 64)

        x = _conv(scale_2, 128, stride=2)  # 4

        x = _skippy_block(x, 64, 128)
        scale_3 = _skippy_block(x, 64, 128)

        x = _conv(scale_3, 256, stride=2)  # 8

        x = _skippy_block(x, 128, 256)
        x = _skippy_block(x, 128, 256)
        x = _skippy_block(x, 128, 256)
        x = _skippy_block(x, 128, 256)
        x = _skippy_block(x, 128, 256)
        x = _skippy_block(x, 128, 256)
        x = _skippy_block(x, 128, 256)
        scale_4 = _skippy_block(x, 128, 256)

        x = _conv(scale_4, 512, stride=2)  # 16

        x = _skippy_block(x, 256, 512)
        x = _skippy_block(x, 256, 512)
        x = _skippy_block(x, 256, 512)
        x = _skippy_block(x, 256, 512)
        x = _skippy_block(x, 256, 512)
        x = _skippy_block(x, 256, 512)
        x = _skippy_block(x, 256, 512)
        scale_5 = _skippy_block(x, 256, 512)

        x = _conv(scale_5, 1024, stride=2)  # 32

        x = _skippy_block(x, 512, 1024)
        x = _skippy_block(x, 512, 1024)
        x = _skippy_block(x, 512, 1024)
        x = _skippy_block(x, 512, 1024)

        # END Darknet-53
        # START YOLOv3

        x = _conv(x, 512, kernel=1)
        x = _conv(x, 1024, kernel=3)
        x = _conv(x, 512, kernel=1)
        x = _conv(x, 1024, kernel=3)
        x1 = _conv(x, 512, kernel=1)

        x_o = _conv(x1, 1024, kernel=3)
        outputs.append(_conv_output(x_o))

        x = _conv(x1, 256, kernel=1)
        x = tf.keras.layers.UpSampling2D()(x)

        x = tf.keras.layers.concatenate([x, scale_5], axis=concat_axis, name="concatenate_1")

        x = _conv(x, 256, kernel=1)
        x = _conv(x, 512, kernel=3)
        x = _conv(x, 256, kernel=1)
        x = _conv(x, 512, kernel=3)
        x1 = _conv(x, 256, kernel=1)

        x = _conv(x1, 512, kernel=3)
        outputs.append(_conv_output(x))

        x = _conv(x1, 128, kernel=1)
        x = tf.keras.layers.UpSampling2D()(x)

        x = tf.keras.layers.concatenate([x, scale_4], axis=concat_axis, name="concatenate_2")

        x = _conv(x, 128, kernel=1)
        x = _conv(x, 256, kernel=3)
        x = _conv(x, 128, kernel=1)
        x = _conv(x, 256, kernel=3)
        x1 = _conv(x, 128, kernel=1)

        x = _conv(x1, 256, kernel=3)
        outputs.append(_conv_output(x))

        return outputs


if __name__ == '__main__':
    import numpy as np
    yolo = YOLOv3((352, 352, 3))
    print(tf.keras.__version__)
    print(sorted(layer.name for layer in yolo.model.layers))
    yolo.model.predict(np.random.uniform(size=(1, 352, 352, 3)))
