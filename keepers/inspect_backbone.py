import tensorflow as tf

from verres.utils import keras_utils

MODEL = "MobileNet"
INPUT_SHAPE = (200, 320, 3)

catalogue = keras_utils.ApplicationCatalogue()

model = catalogue.make_model(
    MODEL,
    include_top=False,
    input_shape=INPUT_SHAPE,
    build_model=True)

tf.keras.utils.plot_model(model, MODEL + ".png", show_shapes=True, show_layer_names=True)

keras_utils.measure_backwards_fps(
    model,
    input_tensor_shape=INPUT_SHAPE,
    output_tensor_shape=tf.keras.backend.int_shape(model.output)[1:])
