from typing import List, Tuple

from tensorflow.keras import layers as tfl

from . import FeatureSpec, VRSBackbone


class SmallFCNN(VRSBackbone):

    default_specs = {1: FeatureSpec("block1_conv1", 1),
                     2: FeatureSpec("block2_conv1", 2),
                     4: FeatureSpec("block3_conv2", 4),
                     8: FeatureSpec("block4_conv1", 8)}

    def __init__(self,
                 strides=(8,),
                 width_base: int = 16):

        feature_specs = []
        for stride in strides:
            if stride not in self.default_specs:
                raise NotImplementedError(f"No SmallBackbone available for stride {stride}")
            feature_specs.append(self.default_specs[stride])

        super().__init__(feature_specs)
        stride = max(spec.working_stride for spec in feature_specs)
        self.layer_objects = [
            tfl.Conv2D(width_base, 3, padding="same", activation="relu", name="block1_conv1")]
        if stride > 1:
            self.layer_objects.extend([
                tfl.MaxPool2D(name="block1_pool"),
                tfl.Conv2D(width_base*2, 3, padding="same", activation="relu", name="block2_conv1")])
        if stride > 2:
            self.layer_objects.extend([
                tfl.MaxPool2D(name="block2_pool"),
                tfl.Conv2D(width_base*4, 3, padding="same", activation="relu", name="block3_conv1"),
                tfl.Conv2D(width_base*4, 3, padding="same", activation="relu", name="block3_conv2")])
        if stride > 4:
            self.layer_objects.extend([
                tfl.MaxPool2D(name="block3_pool"),
                tfl.Conv2D(width_base*8, 3, padding="same", activation="relu", name="block4_conv1")])
        self.output_layer_names = [spec.layer_name for spec in self.feature_specs]

    def call(self, x, training=None, mask=None):
        result = []
        for layer in self.layer_objects:
            x = layer(x)
            if layer.name in self.output_layer_names:
                result.append(x)
        return result
