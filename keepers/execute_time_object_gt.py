import numpy as np

from verres.data import cocodoom
from verres.utils import visualize

STRIDE = 8

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/map-val.json",
        images_root="/data/Datasets/cocodoom",
        stride=STRIDE
    )
)
full_loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/map-full-val.json",
        images_root="/data/Datasets/cocodoom",
        stride=STRIDE
    )
)
stream = cocodoom.COCODoomTimeSequence(
    cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.DETECTION,
                                  batch_size=1,
                                  shuffle=True,
                                  min_no_visible_objects=2),
    data_loader=loader,
    full_data_loader=full_loader
)

vis = visualize.Visualizer()
screen = visualize.CV2Screen(fps=25, scale=2)

for [data] in stream:

    past_image, past_hmap, past_locations, past_rreg, past_wh = data[:len(data) // 2]
    pres_image, pres_hmap, pres_locations, pres_rreg, pres_wh = data[len(data) // 2:]

    past_canvas = vis.overlay_heatmap(past_image, past_hmap)
    pres_canvas = vis.overlay_heatmap(pres_image, pres_hmap)

    canvas = np.concatenate([past_canvas, pres_canvas], axis=2)

    screen.write(canvas)
