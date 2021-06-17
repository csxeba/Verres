import dataclasses
from typing import Tuple, List

import yaml


@dataclasses.dataclass
class ContextConfig:
    execution_type: str = ""
    artifactory_root: str = ""
    experiment_set: str = ""
    experiment_name: str = ""
    verbose: int = 1
    debug: bool = False
    float_precision: str = "float32"


@dataclasses.dataclass
class DatasetSpec:
    name: str = ""
    root: str = ""
    subset: str = ""
    sampling_probability: float = 1.0
    filtered_types: List[int] = "default"
    filtered_map_numbers: List[int] = "default"
    filtered_num_objects: int = 0
    transformations: List[dict] = dataclasses.field(default_factory=list)
    kwargs: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelSpec:
    input_width: int = -1
    input_height: int = -1
    input_shape: Tuple[int, int] = (0, 0)
    backbone_spec: dict = dataclasses.field(default_factory=dict)
    neck_spec: dict = dataclasses.field(default_factory=dict)
    head_spec: dict = dataclasses.field(default_factory=dict)
    output_features: Tuple[str] = ()
    output_strides: Tuple[int] = ()
    maximum_stride: int = 0
    weights: str = None


class TrainingConfig:

    def __init__(self,
                 data: List[dict] = None,
                 epochs: int = 0,
                 batch_size: int = 0,
                 steps_per_epoch: int = -1,
                 prefetch_batches: int = 5,
                 criteria_spec: dict = None,
                 optimizer_spec: dict = None,
                 lr_schedule_spec: dict = None,
                 callbacks: List[dict] = None,
                 initial_epoch: int = 0):

        self.data = [DatasetSpec(**spec) for spec in data]
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.prefetch_batches = prefetch_batches
        self.criteria_spec = criteria_spec
        self.optimizer_spec = optimizer_spec
        self.lr_schedule_spec = lr_schedule_spec
        self.callbacks = callbacks
        self.initial_epoch = initial_epoch


class EvaluationConfig:

    def __init__(self,
                 data: List[dict] = None,
                 detection_output_file: str = "temporary"):

        self.data = [DatasetSpec(**spec) for spec in data]
        self.detection_output_file = detection_output_file


class InferenceConfig:

    def __init__(self,
                 data: List[dict] = None,
                 to_screen: bool = False,
                 output_video_path: str = "",
                 fps: int = -1,
                 total_num_frames: int = -1,
                 output_upscale_factor: int = 1,
                 visualization_mode: str = ""):

        self.data = [DatasetSpec(**spec) for spec in data]
        self.to_screen = to_screen
        self.output_video_path = output_video_path
        self.fps = fps
        self.total_num_frames = total_num_frames
        self.output_upscale_factor = output_upscale_factor
        self.visualization_mode = visualization_mode


class Config:

    _instance: "Config" = None

    def __new__(cls, config_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__(config_path)
        else:
            if cls._instance.config_path != config_path:
                raise RuntimeError(f"Config is a singleton. "
                                   f"You tried to reinitialize it with config_path = {config_path}")
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("Config is not yet initialized.")
        return cls._instance

    def __init__(self, config_path: str):
        self.config_path = config_path
        config_dict = yaml.load(open(config_path))
        self.context = ContextConfig(**config_dict["context"])
        self.model = ModelSpec(**config_dict["model"])
        self.model.input_shape = self.model.input_height, self.model.input_width
        self.training = TrainingConfig(**config_dict["training"])
        self.evaluation = EvaluationConfig(**config_dict["evaluation"])
        self.inference = InferenceConfig(**config_dict["inference"])
        self.whiteboard = {}  # This is populated with runtime descriptors and info

        # noinspection PyUnresolvedReferences
        print(" [Verres] - Read config from", config_path)
