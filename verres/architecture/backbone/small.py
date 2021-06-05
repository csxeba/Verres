import tensorflow as tf

import verres as V
from . import FeatureSpec, VRSBackbone
from ..layers import block


class SmallFCNN(VRSBackbone):

    def __init__(self, config: V.Config):

        self.cfg = config

        self.spec = spec = config.model.backbone_spec.copy()

        width_base = spec["width_base"]
        convolution_kwargs = {"batch_normalize": spec.get("batch_normalize", True),
                              "activation": spec.get("activation", "leakyrelu")}
        bottleneck_kwargs = convolution_kwargs.copy()
        bottleneck_kwargs["channel_expansion"] = spec.get("channel_expansion", 4)

        super().__init__([FeatureSpec("small_fcnn_output", working_stride=8, width=width_base*8)])
        self.layer_objects = [
            block.VRSConvolution(width=width_base, kernel_size=7, **convolution_kwargs)]
        if config.context.verbose > 1:
            print(f" [Verres.SmallFCNN] - Starting architecture with Convolution of width {width_base} and ksize 7")
        self.layer_objects.extend(self._make_stage(depth=1, input_width=width_base, final_width=width_base*2))
        self.layer_objects.extend(self._make_stage(depth=2, input_width=width_base*2, final_width=width_base*4))
        self.layer_objects.extend(self._make_stage(depth=4, input_width=width_base*4, final_width=width_base*8))

    def _make_stage(self, depth: int, input_width: int, final_width: int):
        layers = [block.VRSResidualBottleneck(width=final_width, input_width=input_width, stride=2)]
        if self.cfg.context.verbose > 1:
            print(f" [Verres.SmallFCNN] - Starting stage with residual block: {input_width} -> {final_width}")
        if depth > 1:
            processing_layers = []
            for _ in range(depth - 1):
                processing_layers.append(
                    block.VRSResidualBottleneck(width=final_width, input_width=final_width, stride=1))
                if self.cfg.context.verbose > 1:
                    print(f" [Verres.SmallFCNN] --- Added processing block: {final_width} -> {final_width}")
            layers.extend(processing_layers)
        return layers

    def preprocess_input(self, inputs):
        return tf.cast(inputs, tf.float32) / 255.

    def call(self, x, training=None, mask=None):
        for layer in self.layer_objects:
            x = layer(x, training=training)
        return [x]
