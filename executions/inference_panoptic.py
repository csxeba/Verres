import verres.architecture.head.segmentation
from verres.data import cocodoom
from verres.architecture import backbone as vrsbackbone
from verres.architecture.head import vision

WEIGHTS = "artifactory/panseg/xp_20200908.152323/checkpoints/latest.h5"

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        # data_json="/data/Datasets/cocodoom/enemy-map-val.json",
        data_json="/data/Datasets/cocodoom/map-full-val.json",
        images_root="/data/Datasets/cocodoom",
        stride=8
    )
)

backbone = vrsbackbone.SmallFCNN(strides=(1, 2, 4, 8), width_base=16)
model = verres.architecture.head.segmentation.PanopticSegmentor(
    num_classes=loader.num_classes,
    backbone=backbone,
    weights=WEIGHTS)

cocodoom.inference.run(loader,
                       model,
                       mode=cocodoom.inference.Mode.PANOPTIC,
                       to_screen=True,
                       alpha=1)
