class TASK:

    SEMSEG = "semseg"
    PANSEG = "panseg"
    DEPTH = "depth"
    DETECTION = "det_inf"


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
                 task,
                 batch_size=10,
                 shuffle=True,
                 run_number=None,
                 level_number=None,
                 min_no_visible_objects=0):

        self.task = task
        self.batch_size = batch_size
        self.level_number = level_number
        self.run_number = run_number
        self.shuffle = shuffle
        self.min_no_visible_objects = min_no_visible_objects

