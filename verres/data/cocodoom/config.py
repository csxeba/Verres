class TASK:

    SEGMENTATION = "seg"
    DEPTH = "depth"
    DETECTION_INFERENCE = "det_inf"
    DETECTION_TRAINING = "det_train"


class COCODoomLoaderConfig:

    def __init__(self,
                 data_json,
                 images_root,
                 stride=None):

        self.data_json = data_json
        self.images_root = images_root
        self.stride = stride


class COCODoomStreamConfig:

    def __init__(self,
                 task=TASK.SEGMENTATION,
                 batch_size=16,
                 shuffle=True,
                 run_number=None,
                 level_number=None):

        self.task = task
        self.batch_size = batch_size
        self.level_number = level_number
        self.run_number = run_number
        self.shuffle = shuffle

