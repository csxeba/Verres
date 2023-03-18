from typing import List

import verres as V
from .abstract import Transformation
from .image import ImageProcessor
from .heatmap import UniformSigmaHeatmapProcessor, VariableSigmaHeatmapProcessor
from .regression import RefinementTensor, BoxWHTensor, BoxCornerTensor
from .segmentation import SemanticSegmentationTensor, InstanceSegmentationTensor


_feature_transformation_map = {
    "image": ImageProcessor,
    "uniform_heatmap": UniformSigmaHeatmapProcessor,
    "variable_heatmap": VariableSigmaHeatmapProcessor,
    "refinement": RefinementTensor,
    "box_wh": BoxWHTensor,
    "box_corner": BoxCornerTensor,
    "instance_seg": InstanceSegmentationTensor,
    "semantic_seg": SemanticSegmentationTensor,
    "heatmap": UniformSigmaHeatmapProcessor
}

_sparse_transformations = {
    "refinement", "box_wh", "box_corner"
}


def factory(cfg: V.Config, transformation_specs: List[dict]) -> List[Transformation]:

    transformations = []
    for transformation_params in transformation_specs:
        transformation_params = transformation_params.copy()
        transformation_name = transformation_params.pop("name")
        transformation_type = _feature_transformation_map[transformation_name]
        transformation = transformation_type(cfg, transformation_params)
        transformations.append(transformation)
    print(" [Verres.transformation] - Factory built:", ', '.join(t.__class__.__name__ for t in transformations))
    return transformations
