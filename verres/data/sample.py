import dataclasses
from typing import Optional, Tuple, Dict, Any, List, NamedTuple

import numpy as np
import tensorflow as tf


@dataclasses.dataclass
class Input:
    image_path: str
    shape_whc: Tuple[int, int, int]


@dataclasses.dataclass
class Label:
    object_centers: Optional[np.ndarray] = None
    object_types: Optional[np.ndarray] = None
    object_scores: Optional[np.ndarray] = None
    object_keypoint_coords: Optional[np.ndarray] = None
    segmentation_repr: Optional[list] = None
    semantic_segmentation: Optional[np.ndarray] = None
    instance_pixel_coords: Optional[np.ndarray] = None
    instance_pixel_affiliations: Optional[np.ndarray] = None


@dataclasses.dataclass
class Sample:
    ID: int
    input: Input
    label: Optional[Label] = None
    detection: Optional[Label] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    encoded_tensors: Dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    @classmethod
    def create(
        cls,
        ID: int,
        input_: Input,
        label: Optional[Label] = None,
        detection: Optional[Label] = None,
        metadata: Optional[Dict[str, Any]] = None,
        encoded_tensors: Optional[Dict[str, np.ndarray]] = None,
    ) -> "Sample":
        return cls(
            ID, input_, label or Label(), detection or Label(), metadata or {}, encoded_tensors or {}
        )
