import time

import numpy as np
from matplotlib import pyplot as plt, patches

from verres.data import cocodoom
from verres.tf_arch import backbone as vrsbackbone, vision
from verres.data.cocodoom import inference
from verres.utils import profiling
from verres.operation import numpy_ops

WEIGHTS = "artifactory/panseg/xp_20200907.195808/checkpoints/latest.h5"

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        # data_json="/data/Datasets/cocodoom/enemy-map-val.json",
        data_json="/data/Datasets/cocodoom/map-val.json",
        images_root="/data/Datasets/cocodoom",
        stride=8
    )
)

backbone = vrsbackbone.SmallFCNN(strides=(1, 2, 4, 8), width_base=16)
model = vision.PanopticSegmentor(
    num_classes=loader.num_classes,
    backbone=backbone,
    weights=WEIGHTS)

# inference.run(loader, model, mode=inference.Mode.PANOPTIC, to_screen=True, alpha=1)

stream = cocodoom.COCODoomSequence(
    cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.INFERENCE, batch_size=1, shuffle=True),
    data_loader=loader
)

iterator = iter(stream)
timer = profiling.MultiTimer()

for i in range(100):

    timer.reset()

    with timer.time("dtime"):
        [[image]] = next(stream)

    with timer.time("mtime"):
        hmap, rreg, iseg, sseg = model(image)

    with timer.time("ptime"):
        centroids, types, scores = model.get_centroids(hmap, rreg)
        if centroids.shape[0] == 0:
            continue
        coords_non_bg, iseg_offset = model.get_offsetted_coords(sseg, iseg)
        if iseg_offset.shape[0] == 0:
            continue
        affiliations, offset_scores = model.get_affiliations(iseg_offset, centroids)
        filtered_coords, filtered_affils, *_ = model.get_filtered_result(coords_non_bg, affiliations, offset_scores)

    # print(" - ".join(f"{k}: {v:.4f}" for k, v in timer.get_results(reset=True).items()))
    # print("num pixels :", len(coord))
    # print("num objects:", len(centroid))

    non_bg = np.argmax(sseg[0].numpy(), axis=2) > 0
    vecs = iseg[0].numpy()
    coords = numpy_ops.meshgrid(vecs.shape[:2])[non_bg]
    vecs = vecs[non_bg]

    plt.imshow(image[0][..., ::-1])
    plt.quiver(coords[:, 0],
               coords[:, 1],
               vecs[:, 0],
               -vecs[:, 1],
               width=1,
               headwidth=3,
               headlength=4,
               units="x")
    # plt.plot(filtered_coords[:, 0], filtered_coords[:, 1], "ro", alpha=0.5)
    plt.plot(centroids[:, 0], centroids[:, 1], "gx")
    for cent in centroids:
        circle = patches.Circle(xy=cent, radius=model.offset_nms, facecolor="none", edgecolor="green", alpha=0.5)
        plt.gca().add_artist(circle)
    plt.show()
