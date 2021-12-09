import verres as V

from . import object_detector
from . import segmentor


def factory(config: V.Config, input_feature_specs):
    spec = config.model.head_spec.copy()
    name = spec.pop("name").lower()
    if name == "od":
        head = object_detector.OD(config, input_feature_specs)
    elif name == "panoptic":
        head = segmentor.Panoptic(config)
    else:
        assert False
    if config.context.verbose > 1:
        print(f" [Verres.head] - Factory built: {name}")

    return head
