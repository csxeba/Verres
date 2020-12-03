import pathlib

import verres.architecture.head.detection
from verres.data import cocodoom
from verres.architecture import backbone as vrsbackbone
from verres.architecture.head import vision
from verres.data.cocodoom import inference

p: pathlib.Path
WEIGHTS = str(sorted(pathlib.Path("artifactory/od_small_mr/checkpoints").glob("*"),
                     key=lambda p: int(p.parts[-1].split("_")[2]))[-1])
print(" [Verres] - Weights will be loaded from", WEIGHTS)

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        # data_json="/data/Datasets/cocodoom/enemy-map-val.json",
        data_json="/data/Datasets/cocodoom/map-full-val.json",
        images_root="/data/Datasets/cocodoom",
        stride=8
    )
)

backbone = vrsbackbone.SmallFCNN(width_base=16, strides=(2, 4, 8))
fusion = vrsbackbone.FeatureFuser(backbone, final_stride=8, base_width=8, final_width=32)

model = verres.architecture.head.detection.ObjectDetector(num_classes=loader.num_classes,
                                                          backbone=fusion,
                                                          stride=8,
                                                          peak_nms=0.1,
                                                          weights=WEIGHTS)

inference.run(loader, model, mode=inference.Mode.DETECTION, to_screen=False,
              output_file="MicroNet-COCODoom-OD-mr.avi", stop_after=30*120)

# inference.run_od(loader, model, mode=inference.Mode.RAW_HEATMAP, to_screen=True)

# evaluation.run(loader, model, detection_file="MicroNet-COCODoom-OD-sched.json")
