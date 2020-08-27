from tensorflow.keras import layers as tfl

from . import FeatureSpec, VRSBackbone


class SmallFCNN(VRSBackbone):

    def __init__(self,
                 width_base: int = 16,
                 stride: int = 8):

        feature_specs = [{4: FeatureSpec("block3_conv2", working_stride=4),
                          8: FeatureSpec("block4_conv1", working_stride=8)}[stride]]

        super().__init__(feature_specs)
        if stride not in {4, 8}:
            raise RuntimeError("Only stride 4 and 8 are supported.")
        self.layer_objects = [
            tfl.Conv2D(width_base, 3, padding="same", activation="relu", name="block1_conv1"),
            tfl.MaxPool2D(name="block1_pool"),
            tfl.Conv2D(width_base*2, 3, padding="same", activation="relu", name="block2_conv1"),
            tfl.MaxPool2D(name="block2_pool"),
            tfl.Conv2D(width_base*4, 3, padding="same", activation="relu", name="block3_conv1"),
            tfl.Conv2D(width_base*4, 3, padding="same", activation="relu", name="block3_conv2")]
        if stride == 8:
            self.layer_objects.extend([
                tfl.MaxPool2D(name="block3_pool"),
                tfl.Conv2D(width_base*8, 3, padding="same", activation="relu", name="block4_conv1")])

    def call(self, x, training=None, mask=None):
        for layer in self.layer_objects:
            x = layer(x)
        return [x]
