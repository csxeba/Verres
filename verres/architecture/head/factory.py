import verres as V

from .base import VRSHead
from . import object_detector
from . import segmentor


_mapping = {
    "od": object_detector.OD,
    "ctdet": object_detector.CTDet,
    "semseg": segmentor.SemSeg,
    "panoptic": segmentor.Panoptic,
}


def factory(config: V.Config, input_feature_specs) -> VRSHead:
    spec = config.model.head_spec.copy()
    name = spec.pop("name")
    if name is None:
        raise RuntimeError(f"name must be set under config/model/head_spec")
    name = name.lower()
    if name not in _mapping:
        raise RuntimeError(f"No such head: {name}. Available heads: {', '.join(_mapping.keys())}")
    head_type = _mapping[name]
    head_obj = head_type(config, input_feature_specs)
    if config.context.verbose > 1:
        print(f" [Verres.head] - Factory built: {name}")
    return head_obj
