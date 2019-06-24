import tensorflow as tf


_convlayer_counter = 1


def _conv(x, filter_num, kernel=3, stride=1, activate=True):
    global _convlayer_counter

    x = tf.keras.layers.Conv2D(filter_num, kernel, use_bias=False, strides=stride, padding="same",
                               data_format="channels_first", name="conv_{}".format(_convlayer_counter))(x)
    x = tf.keras.layers.BatchNormalization(axis=1, name="batch_normalization_v1_{}".format(_convlayer_counter))(x)
    if activate:
        x = tf.keras.layers.LeakyReLU(name="leaky_re_lu_{}".format(_convlayer_counter))(x)
    _convlayer_counter += 1
    return x


def _conv_output(x):
    global _convlayer_counter
    x = tf.keras.layers.Conv2D(255, 1, data_format="channels_first", name="conv_{}".format(_convlayer_counter))(x)
    return


def _skippy_block(x, filters1, filters2):
    x1 = _conv(x, filters1, kernel=1)
    x1 = _conv(x1, filters2, kernel=3)
    return tf.keras.layers.add([x, x1])


class YOLOv3:

    def __init__(self, input_shape=(3, None, None), batch_size=None, weights=None):
        self.inputs = tf.keras.Input(input_shape, batch_size=batch_size)
        outputs = self.build()
        self.model = tf.keras.Model(self.inputs, outputs)
        if weights is not None:
            self.model.load_weights(weights, by_name=True)

    def build(self):
        outputs = []

        scale_1 = _conv(self.inputs, 32)  # 1

        x = _conv(scale_1, 64, stride=2)  # 2

        scale_2 = _skippy_block(x, 32, 64)  # 4

        x = _conv(scale_2, 128, stride=2)  # 5

        x = _skippy_block(x, 64, 128)  # 7
        scale_3 = _skippy_block(x, 64, 128)  # 9

        x = _conv(scale_3, 256, stride=2)  # 10

        x = _skippy_block(x, 128, 256)  # 12
        x = _skippy_block(x, 128, 256)
        x = _skippy_block(x, 128, 256)
        x = _skippy_block(x, 128, 256)
        x = _skippy_block(x, 128, 256)
        x = _skippy_block(x, 128, 256)
        x = _skippy_block(x, 128, 256)
        scale_4 = _skippy_block(x, 128, 256)  # 26

        x = _conv(scale_4, 512, stride=2)  # 27

        x = _skippy_block(x, 256, 512)  # 29
        x = _skippy_block(x, 256, 512)
        x = _skippy_block(x, 256, 512)
        x = _skippy_block(x, 256, 512)
        x = _skippy_block(x, 256, 512)
        x = _skippy_block(x, 256, 512)
        x = _skippy_block(x, 256, 512)
        scale_5 = _skippy_block(x, 256, 512)  # 43

        x = _conv(scale_5, 1024, stride=2)  # 44

        x = _skippy_block(x, 512, 1024)  # 46
        x = _skippy_block(x, 512, 1024)
        x = _skippy_block(x, 512, 1024)
        x = _skippy_block(x, 512, 1024)  # 52

        # END Darknet-53
        # START YOLOv3

        x = _conv(x, 512, kernel=1)
        x = _conv(x, 1024, kernel=3)
        x = _conv(x, 512, kernel=1)
        x = _conv(x, 1024, kernel=3)
        x1 = _conv(x, 512, kernel=1)

        x = _conv(x, 1024, kernel=1)
        outputs.append(_conv_output(x))

        x = _conv(x1, 256, kernel=1)
        x = tf.keras.layers.UpSampling2D()(x)

        x = tf.keras.layers.concatenate([x, scale_4], axis=1)

        x = _conv(x, 256, kernel=1)
        x = _conv(x, 512, kernel=3)
        x = _conv(x, 256, kernel=1)
        x = _conv(x, 512, kernel=3)
        x1 = _conv(x, 256, kernel=1)

        x = _conv(x1, 512, kernel=3)
        outputs.append(_conv_output(x))

        x = _conv(x1, 128, kernel=1)
        x = tf.keras.layers.UpSampling2D()(x)

        x = tf.keras.layers.concatenate([x, scale_3], axis=1)

        x = _conv(x, 128, kernel=1)
        x = _conv(x, 256, kernel=3)
        x = _conv(x, 128, kernel=1)
        x = _conv(x, 256, kernel=3)
        x1 = _conv(x, 128, kernel=1)

        x = _conv(x1, 256, kernel=3)
        outputs.append(_conv_output(x))

        return outputs


if __name__ == '__main__':
    yolo = YOLOv3()
    print(tf.keras.__version__)
    print(sorted(layer.name for layer in yolo.model.layers))

