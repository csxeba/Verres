import verres as V

from . import object_detector
from . import segmentor


def factory(config: V.Config):
    spec = config.model.head_spec.copy()
    name = spec.pop("name").lower()
    if name == "od":
        head = object_detector.OD(config)
    elif name == "panoptic":
        head = segmentor.Panoptic(config)
    else:
        assert False
    if config.context.verbose > 1:
        print(f" [Verres.neck] - Factory built: {name}")

    return head
