from typing import List

import tqdm

from verres.data import cocodoom
from verres.tf_arch.backbone import ApplicationBackbone, FeatureSpec
from verres.tf_arch.vision import ObjectDetector
from verres.data.cocodoom import evaluation

BATCH_SIZE = 1
BACKBONE = "MobileNet"
FEATURE_LAYER_NAMES: List[str] = ["conv_pw_5_relu"]
FEATURE_STRIDES: List[int] = [8]
feature_specs = [FeatureSpec(name, stride) for name, stride in zip(FEATURE_LAYER_NAMES, FEATURE_STRIDES)]

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/enemy-map-val.json",
        images_root="/data/Datasets/cocodoom",
        stride=FEATURE_STRIDES[-1]
    )
)

backbone = ApplicationBackbone(BACKBONE,
                               feature_specs=feature_specs,
                               input_shape=(None, None, 3),
                               fixed_batch_size=BATCH_SIZE)
model = ObjectDetector(num_classes=loader.num_classes,
                       backbone=backbone,
                       stride=FEATURE_STRIDES[-1],
                       weights="models/MobileNet-OD.h5")

evaluation.run(loader, model, detection_file="MobileNet-OD-detections.json")
