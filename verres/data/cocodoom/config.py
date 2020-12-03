from typing import Tuple


class TASK:

    SEMSEG = "semseg"
    PANSEG = "panseg"
    DEPTH = "depth"
    DETECTION = "detection"
    INFERENCE = "inference"


class COCODoomLoaderConfig:

    def __init__(self,
                 data_json: str,
                 images_root: str,
                 stride: int = None,
                 input_shape: Tuple[int, int, int] = None,
                 allow_empty_ids: bool = False):

        self.data_json = data_json
        self.images_root = images_root
        self.stride = stride
        self.input_shape = input_shape
        self.allow_empty_ids = allow_empty_ids


class COCODoomStreamConfig:

    def __init__(self,
                 task: str,
                 batch_size: int = 10,
                 shuffle: bool = True,
                 run_number: int = None,
                 level_number: int = None,
                 min_no_visible_objects: int = 0):

        self.task = task
        self.batch_size = batch_size
        self.level_number = level_number
        self.run_number = run_number
        self.shuffle = shuffle
        self.min_no_visible_objects = min_no_visible_objects
