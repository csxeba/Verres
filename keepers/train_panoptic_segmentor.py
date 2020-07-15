import tensorflow as tf

from verres.data import cocodoom


class Backbone(tf.keras.Model):

    def __init__(self, input_shape, end_layer):
        super().__init__()
        self.


class OffSegNet(tf.keras.Model):

    def __init__(self, input_shape):
        super().__init__()
        self.backbone =


loader = cocodoom.COCODoomLoader(
    cocodoom.COCODoomLoaderConfig(
        data_json="/data/Datasets/cocodoom/map-train.json",
        images_root="/data/Datasets/cocodoom",
        stride=4
    )
)
streamcfg = cocodoom.COCODoomStreamConfig(task=cocodoom.TASK.SEGMENTATION,
                                          batch_size=32,
                                          shuffle=True,
                                          min_no_visible_objects=2)
stream = cocodoom.COCODoomSequence(streamcfg, loader)
