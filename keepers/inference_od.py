import pathlib

from verres.data import cocodoom
from verres.tf_arch import backbone as vrsbackbone, vision
from verres.data.cocodoom import inference, evaluation

p: pathlib.Path
WEIGHTS = str(sorted(pathlib.Path("artifactory/od_sched/checkpoints").glob("*"),
                     key=lambda p: int(p.parts[-1].split("_")[2]))[-1])
print(" [Verres] - Weights will be loaded from", WEIGHTS)

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/enemy-map-val.json",
        images_root="/data/Datasets/cocodoom",
        stride=8
    )
)

backbone = vrsbackbone.SmallFCNN(width_base=16)

model = vision.ObjectDetector(num_classes=loader.num_classes,
                              backbone=backbone,
                              stride=8,
                              weights=WEIGHTS,
                              peak_nms=0.3)

# inference.run_od(loader, model, mode=inference.Mode.DETECTION, to_screen=False,
#                  output_file="MicroNet-COCODoom-OD-sched.avi", stop_after=30*120)

# inference.run_od(loader, model, mode=inference.Mode.DETECTION, to_screen=True)

evaluation.run(loader, model, detection_file="MicroNet-COCODoom-OD-sched.json")
