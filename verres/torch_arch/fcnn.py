import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SkippyBackbone(nn.ModuleList):

    def __init__(self, input_shape, width_multiplier=1, batch_norm=True):
        widths = (np.array([8, 16, 32, 64, 64, 64, 64, 64]) * width_multiplier).tolist()
        inputs = [input_shape[0]] + widths[:-1]
        layers = []
        for input_dim, output_dim in zip(inputs, widths):
            layers.append(nn.Conv2d(input_dim, output_dim, 3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_features=output_dim))
            layers.append(F.relu)

        super().__init__(layers)

    def forward(self, x):
        output_cache = None
        for i, layer in self:
            x = layer(x)
            if i in (5, 7):
                x = torch.add(output_cache, x)
            if i in (3, 5):
                output_cache = x
        return x


class FCNN(nn.Module):

    def __init__(self, input_shape, width_multiplier=1, batch_normalize=True):
        super().__init__()
        self.input_shape = input_shape
        self.width_multiplier = width_multiplier
        self.backbone = None  # type: nn.ModuleList

    def build_skippy_backbone(self):
        pass

    def skippy_backbone_forward(self):
        pass
