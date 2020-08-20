import tensorflow as tf
from tensorflow.keras import layers as tfl

from . import FeatureSpec


# noinspection PyPep8Naming
def SmallFCNN(width_base: int = 8):
    model = tf.keras.models.Sequential([
        tfl.Conv2D(width_base, 3, padding="same", activation="relu", input_shape=(None, None, 3)),
        tfl.MaxPool2D(),
        tfl.Conv2D(width_base*2, 3, padding="same", activation="relu"),
        tfl.MaxPool2D(),
        tfl.Conv2D(width_base*4, 3, padding="same", activation="relu"),
        tfl.Conv2D(width_base*4, 3, padding="same", activation="relu"),
        tfl.MaxPool2D(),
        tfl.Conv2D(width_base*8, 3, padding="same", activation="relu"),
        tfl.Lambda(lambda x: [x], name="small_fcnn_features")])

    ftr_spec = FeatureSpec("small_fcnn_features", working_stride=8)
    ftr_spec.width = width_base * 8

    model.feature_specs = [ftr_spec]
    return model
