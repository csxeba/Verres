from typing import Union

import verres as V

from .base import VRSNeck
from .fusion import AutoFusion
from .rescaler import Rescaler
from ..backbone import VRSBackbone


_mapping = {"autofusion": AutoFusion,
            "rescaler": Rescaler}


def factory(config: V.Config, backbone: VRSBackbone) -> Union[None, VRSNeck]:

    neck = None
    if config.model.neck_spec:
        spec = config.model.neck_spec.copy()
        name = spec.pop("name")
        neck = _mapping[name.lower()](config, backbone)
        if config.context.verbose > 1:
            print(f" [Verres.neck] - Factory built: {name}")

    return neck
