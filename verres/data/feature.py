import warnings
from typing import List, Tuple

import verres as V


class Feature:

    def __init__(self,
                 name: str,
                 stride: int,
                 sparse: bool,
                 dtype: str,
                 meta_field: str = None,
                 depth: int = -1,
                 shape: tuple = ()):

        self.name: str = name
        self.stride = stride
        self.sparse = sparse
        self.dtype = dtype
        self.meta_field = meta_field
        if self.meta_field is None:
            self.meta_field = self.name + "_tensor"
        self.parent = None
        self.depth: int = depth
        self.shape: Tuple[int] = shape
        if not self.sparse and self.shape[1] < self.shape[0]:
            warnings.warn(f"Non-sparse feature {name} might have erroneous height-first shape definition: {self.shape}")


class MultiFeature:

    def __init__(self,
                 name: str,
                 feature_list: List[Feature]):

        self.name: str = name
        self.feature_list = feature_list
        for ftr in self.feature_list:
            ftr.parent = self
