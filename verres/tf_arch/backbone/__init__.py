class FeatureSpec:

    def __init__(self, layer_name: str, working_stride: int = None):
        self.layer_name = layer_name
        self.working_stride = working_stride
        self.width = -1


from .application import ApplicationBackbone
from .side_tune import SideTunedBackbone
from .small import SmallFCNN
