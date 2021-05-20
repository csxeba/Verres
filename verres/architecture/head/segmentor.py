import tensorflow as tf

import verres as V
from ..layers import block
from ..layers.stage import StageBody


class Panoptic(tf.keras.layers.Layer):

    def __init__(self, config: V.Config):
        super().__init__()
        num_classes = config.model.head_spec["num_classes"]
        self.body8 = StageBody(width=64, num_blocks=5, skip_connect=True)
        self.body4 = StageBody(width=32, num_blocks=3, skip_connect=True)
        self.body2 = StageBody(width=16, num_blocks=1, skip_connect=True)
        self.upsc8_4 = block.VRSUpscale(stride=2, base_width=64, kernel_size=3)
        self.upsc4_2 = block.VRSUpscale(stride=2, base_width=32, kernel_size=3)
        self.upsc2_1 = block.VRSUpscale(stride=2, base_width=16, kernel_size=3)
        self.hmap = block.VRSHead(pre_width=64, output_width=num_classes)
        self.rreg = block.VRSHead(pre_width=64, output_width=num_classes * 2)
        self.sseg = block.VRSHead(pre_width=8, output_width=num_classes + 1)
        self.iseg = block.VRSHead(pre_width=8, output_width=2)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        ftr1, ftr2, ftr4, ftr8 = inputs

        ftr8 = self.body8(ftr8, training=training, mask=mask)

        ftr4 = tf.concat([self.upsc8_4(ftr8, training=training, mask=mask), ftr4], axis=-1)
        ftr4 = self.body4(ftr4, training=training, mask=mask)

        ftr2 = tf.concat([self.upsc4_2(ftr4, training=training, mask=mask), ftr2], axis=-1)
        ftr2 = self.body2(ftr2, training=training, mask=mask)

        ftr1 = tf.concat([self.upsc2_1(ftr2, training=training, mask=mask), ftr1], axis=-1)

        hmap = self.hmap(ftr8, training=training, mask=mask)
        rreg = self.rreg(ftr8, training=training, mask=mask)
        iseg = self.iseg(ftr1, training=training, mask=mask)
        sseg = self.sseg(ftr1, training=training, mask=mask)

        return {"heatmap": hmap, "refinement": rreg, "instance_seg": iseg, "semantic_seg": sseg}
