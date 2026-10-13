import verres as V

from .small import SmallFCNN
from .application import ApplicationBackbone


def factory(config: V.Config):
    spec = dict(config.model.backbone_spec)
    name = spec.pop("name", None)
    if name is None:
        raise RuntimeError("name must be set under config/model/backbone_spec")
    if name.lower() in ["small", "smallfcnn", "small_fcnn"]:
        backbone = SmallFCNN(config)
        if config.context.verbose > 1:
            print(" [Verres.backbone] - Factory built: SmallFCNN")
    else:
        backbone = ApplicationBackbone(config)
    if config.context.verbose > 1:
        print(f" [Verres.backbone] - Factory built: {name}")
    return backbone
