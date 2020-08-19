import numpy as np

from verres.data import cocodoom
from verres.utils import visualize

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/map-full-val.json",
        images_root="/data/Datasets/cocodoom",
        stride=4
    )
)

vis = visualize.Visualizer(n_classes=loader.num_classes)
screen = visualize.CV2Screen(fps=25, scale=4)

for ID in loader.index:

    meta = loader.image_meta[ID]
    annos = loader.index[ID]

    img = loader.get_image(ID)
    print(img.shape)
    locations, bboxes = loader.get_bbox(ID, batch_idx=0)

    boxes = np.concatenate([locations[:, 1:3] * 4, bboxes * 4, locations[..., -1:]], axis=1)
    boxes = boxes[::4]

    canvas = vis.overlay_boxes(img, boxes.astype(int))
    screen.blit(canvas)
