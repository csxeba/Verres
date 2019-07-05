import tensorflow as tf

from ..data.cocodoom import TASK


def build(task=TASK.SEGMENTATION, input_shape=(None, 320, 200, 3), num_output_classes=None, weights=None):
    vgg = tf.keras.applications.VGG19(input_shape=input_shape[1:], weights=weights)  # type: tf.keras.Model
    features = vgg.get_layer("flatten").output
    x = tf.keras.layers.Dense(num_output_classes, activation="softmax")(features)
    return tf.keras.Model(vgg.input, x, name="VGGClassifier")
