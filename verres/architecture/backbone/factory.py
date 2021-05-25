import verres as V

from .small import SmallFCNN
from .application import ApplicationBackbone


def factory(config: V.Config):
    spec = config.model.backbone_spec.copy()
    name = spec.pop("name")
    if name.lower() in ["small", "smallfcnn", "small_fcnn"]:
        backbone = SmallFCNN(config)
    else:
        backbone = ApplicationBackbone(config)
    if config.context.verbose > 1:
        print(f" [Verres.backbone] - Factory built: {name}")
    return backbone
