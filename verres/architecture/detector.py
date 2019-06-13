import tensorflow as tf

from ..utils import losses


def _block(x0, num_filters, depth, pool):
    if pool:
        x0 = tf.keras.layers.MaxPool2D()(x0)
    x0 = tf.keras.layers.Conv2D(num_filters, 1, padding="valid")(x0)
    x = tf.keras.layers.BatchNormalization()(x0)
    x = tf.keras.layers.ReLU()(x)
    for d in range(depth):
        x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x0)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    if depth > 1:
        x = tf.keras.layers.Add()([x0, x])
    return x


def _pool(x):
    return tf.keras.layers.MaxPool2D()(x)


def _head(x, num_output_classes, postfix=""):
    heatmap_output = tf.keras.layers.Conv2D(num_output_classes, kernel_size=5, padding="same", activation="sigmoid",
                                            name=f"Heatmaps{postfix}")(x)
    refinement_output = tf.keras.layers.Conv2D(2, kernel_size=5, padding="same",
                                               name=f"Refinements{postfix}")(x)
    boxparam_output = tf.keras.layers.Conv2D(2, kernel_size=5, padding="same",
                                             name=f"BoxParams{postfix}")(x)
    return [heatmap_output, refinement_output, boxparam_output]


def _mask(refinement_output, boxparam_output, mask, postfix=""):
    masked_refinement = tf.keras.layers.Lambda(lambda xx: xx[0] * xx[1],
                                               name=f"M_refine{postfix}")([refinement_output, mask])
    masked_boxparam = tf.keras.layers.Lambda(lambda xx: xx[0] * xx[1],
                                             name=f"M_boxparm{postfix}")([boxparam_output, mask])
    return [masked_refinement, masked_boxparam]


class COCODoomDetector:

    def __init__(self, input_shape: tuple="cocodoom", num_output_classes=18, block_depth=1, block_widening=1):

        if input_shape == "cocodoom":
            input_shape = (None, 200, 320, 3)
        if len(input_shape) == 3:
            input_shape = (None,) + input_shape

        inputs = tf.keras.layers.Input(batch_shape=input_shape, name="images")

        x = _block(inputs, 8*block_widening, block_depth, 0)  # 200 320
        x = _block(x, 16*block_widening, block_depth, 1)  # 100 160
        x = _block(x, 32*block_widening, block_depth, 1)  # 50 80
        x = _block(x, 64*block_widening, block_depth, 1)  # 25 40

        mask = tf.keras.Input(batch_shape=tf.keras.backend.int_shape(x)[:3] + (2,))

        heatmap_output, *other_outputs = _head(x, num_output_classes)

        masked_outputs = _mask(*other_outputs, mask)

        self.learner = tf.keras.Model(inputs=[inputs, mask],
                                      outputs=[heatmap_output] + masked_outputs, name="Learner")
        self.predictor = tf.keras.Model(inputs=inputs,
                                        outputs=[heatmap_output] + other_outputs, name="Predictor")

        self.stride = 8

    def compile_default(self, lrate=2e-5):
        self.learner.compile(optimizer=tf.keras.optimizers.Adam(lrate),
                             loss=[losses.sse] + [losses.sae]*2,
                             loss_weights=[1, 0.1, 1])
