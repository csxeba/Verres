from matplotlib import pyplot as plt
from verres.data import cocodoom

loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/map-train.json",
        images_root="/data/Dataset/cocodoom",
        stride=4
    )
)

for ID in loader.index:
    seg_mask, inst_mask = loader.get_panoptic_masks(ID)
