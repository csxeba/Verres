import numpy as np
from matplotlib import pyplot as plt

from verres.data import cocodoom
from verres.utils import visualize

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/map-train.json",
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
    seg_mask, inst_mask = loader.get_panoptic_masks(ID)

    fg = np.linalg.norm(inst_mask, ord=1, axis=-1) > 0

    XY = np.argwhere(fg)
    # XY = XY[::4]
    DXDY = inst_mask[tuple(XY[..., 0]), tuple([XY[..., 1]])]
    X1Y1 = XY + DXDY

    plt.imshow(img[..., ::-1])

    plt.plot(XY[..., 1], XY[..., 0], "gx")
    plt.plot(X1Y1[..., 1], X1Y1[..., 0], "rx")
    # plt.quiver(XY[..., 1], XY[..., 0], X1Y1[..., 0], -X1Y1[..., 1],
    #            alpha=0.6, color="green", width=1, headwidth=3, headlength=4., units="x")
    plt.show()
