import tensorflow as tf

from ..utils import losses


class COCODoomDetector:

    def __init__(self, input_shape="cocodoom", num_output_classes=18):

        if len(input_shape) == 3:
            input_shape = (None,) + input_shape
        if input_shape == "cocodoom":
            input_shape = (None, 200, 320, 3)

        inputs = tf.keras.layers.Input(batch_shape=input_shape, name="images")

        down_stage1 = tf.keras.layers.Conv2D(8, 3, padding="same")(inputs)
        down_stage1 = tf.keras.layers.BatchNormalization()(down_stage1)
        down_stage1 = tf.keras.layers.ReLU()(down_stage1)

        down_stage2 = tf.keras.layers.MaxPool2D()(down_stage1)  # 100
        down_stage2 = tf.keras.layers.Conv2D(16, 3, padding="same")(down_stage2)
        down_stage2 = tf.keras.layers.BatchNormalization()(down_stage2)
        down_stage2 = tf.keras.layers.ReLU()(down_stage2)

        down_stage3 = tf.keras.layers.MaxPool2D()(down_stage2)  # 50
        down_stage3 = tf.keras.layers.Conv2D(16, 3, padding="same")(down_stage3)
        down_stage3 = tf.keras.layers.BatchNormalization()(down_stage3)
        down_stage3 = tf.keras.layers.ReLU()(down_stage3)

        down_stage4 = tf.keras.layers.MaxPool2D()(down_stage3)  # 25
        down_stage4 = tf.keras.layers.Conv2D(16, 3, padding="same")(down_stage4)
        down_stage4 = tf.keras.layers.BatchNormalization()(down_stage4)
        down_stage4 = tf.keras.layers.ReLU()(down_stage4)

        down_stage4 = tf.keras.layers.Conv2D(16, 3, padding="same")(down_stage4)
        down_stage4 = tf.keras.layers.BatchNormalization()(down_stage4)
        down_stage4 = tf.keras.layers.ReLU()(down_stage4)

        mask = tf.keras.Input(batch_shape=tf.keras.backend.int_shape(down_stage4)[:3] + (2,))

        heatmap_output = tf.keras.layers.Conv2D(num_output_classes, kernel_size=5, padding="same",
                                                name="Heatmaps")(down_stage4)
        refinement_output = tf.keras.layers.Conv2D(2, kernel_size=5, padding="same",
                                                   name="Refinements")(down_stage4)
        boxparam_output = tf.keras.layers.Conv2D(2, kernel_size=5, padding="same",
                                                 name="BoxParams")(down_stage4)

        masked_refinement = tf.keras.layers.Lambda(lambda xx: xx[0] * xx[1],
                                                   name="Masked_refinements")([refinement_output, mask])
        masked_boxparam = tf.keras.layers.Lambda(lambda xx: xx[0] * xx[1],
                                                 name="Masked_boxparams")([boxparam_output, mask])

        self.learner = tf.keras.Model(inputs=[inputs, mask],
                                      outputs=[heatmap_output, masked_refinement, masked_boxparam], name="Learner")
        self.predictor = tf.keras.Model(inputs=inputs,
                                        outputs=[heatmap_output, refinement_output, boxparam_output], name="Predictor")

        self.stride = 8

    def compile_default(self, lrate=2e-5):
        self.learner.compile(optimizer=tf.keras.optimizers.Adam(lrate),
                             loss=losses.sse)
