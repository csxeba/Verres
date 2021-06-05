from typing import List

import verres as V
from .abstract import Transformation
from ..dataset import DatasetDescriptor
from .image import ImageProcessor
from .heatmap import UniformSigmaHeatmapProcessor, VariableSigmaHeatmapProcessor
from .regression import RegressionTensor
from .segmentation import PanopticSegmentationTensor, SemanticSegmentationTensor
from .filters import FilterNumObjects


_feature_transformation_map = {
    "image": ImageProcessor,
    "uniform_heatmap": UniformSigmaHeatmapProcessor,
    "variable_heatmap": VariableSigmaHeatmapProcessor,
    "regression": RegressionTensor,
    "panoptic_seg": PanopticSegmentationTensor,
    "semantic_seg": SemanticSegmentationTensor,
    "filter_num_objects": FilterNumObjects
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
        if cfg.context.verbose:
            print(" [Verres.transformation] - Factory built:", transformation.__class__.__name__)
    return transformations
