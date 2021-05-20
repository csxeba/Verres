from typing import List, Dict

import verres as V
from ..dataset import DatasetDescriptor
from .abstract import Transformation, TransformationList
from .image import ImageProcessor
from .heatmap import HeatmapProcessor
from .regression import RegressionTensor
from .segmentation import SemanticSegmentationTensor, PanopticSegmentationTensor
from .collate import CollateBatch

# noinspection PyTypeChecker
_feature_transformation_map: Dict[str, Transformation] = {
    "image": ImageProcessor,
    "heatmap": HeatmapProcessor,
    "regression": RegressionTensor,
    "panoptic_seg": PanopticSegmentationTensor,
    "semantic_seg": SemanticSegmentationTensor
}


def factory(cfg: V.Config,
            data_descriptor: DatasetDescriptor,
            transformation_specs: List[dict]) -> List[Transformation]:

    transformations = []
    for transformation_params in transformation_specs:
        transformation_params = transformation_params.copy()
        transformation_type = _feature_transformation_map[transformation_params.pop("name")]
        transformation = transformation_type.from_descriptors(cfg, data_descriptor, transformation_params)
        transformations.append(transformation)
        print(" [Verres.transformation] - Factory built:", transformation.__class__.__name__)
    return transformations
