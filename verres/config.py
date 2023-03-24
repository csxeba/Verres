from typing import Tuple, List, Dict, Any, Optional

import yaml
import pydantic


class ContextConfig(pydantic.BaseModel):
    execution_type: str = ""
    artifactory_root: str = ""
    experiment_set: str = ""
    experiment_name: str = ""
    verbose: int = 1
    debug: bool = False
    float_precision: str = "float32"


class DatasetSpec(pydantic.BaseModel):
    name: str = ""
    root: str = ""
    subset: str = ""
    sampling_probability: float = 1.0
    object_level_filters: List[dict] = []
    image_level_filters: List[dict] = []
    transformations: List[dict] = []
    kwargs: dict = {}


class ClassMapping(pydantic.BaseModel):
    class_order: List[str]
    class_colors_rgb: Dict[str, Tuple[int, int, int]]
    class_mapping: Dict[str, str]

    @classmethod
    def from_path(cls, path: str) -> "ClassMapping":
        return cls(**yaml.load(open(path, "r"), Loader=yaml.FullLoader))

    def coco_name_to_verres_name(self, category_name: str) -> str:
        return self.class_mapping[category_name]

    def coco_name_to_verres_index(self, category_name: str) -> int:
        return self.class_order.index(self.coco_name_to_verres_name(category_name))

    @property
    def num_classes(self) -> int:
        return len(self.class_order)

    def __contains__(self, item):
        return item in self.class_mapping


class ModelSpec(pydantic.BaseModel):
    input_width: int = -1
    input_height: int = -1
    backbone_spec: dict = {}
    neck_spec: dict = {}
    head_spec: dict = {}
    maximum_stride: int = 0
    weights: Optional[str] = None

    @property
    def input_shape_wh(self) -> Tuple[int, int]:
        return self.input_width, self.input_height

    @property
    def input_shape_hw(self):
        return self.input_height, self.input_width


class TrainingConfig(pydantic.BaseModel):
    data: List[DatasetSpec]
    epochs: int = 0
    batch_size: int = 0
    steps_per_epoch: int = -1
    prefetch_batches: int = 5
    criteria_spec: Optional[dict] = None
    optimizer_spec: Optional[dict] = None
    lr_schedule_spec: Optional[dict] = None
    callbacks: List[dict] = []
    initial_epoch: int = 0


class EvaluationConfig(pydantic.BaseModel):
    data: List[DatasetSpec]
    detection_output_file: str = "temporary"


class InferenceConfig(pydantic.BaseModel):
    data: List[DatasetSpec]
    to_screen: bool = False
    output_video_path: str = ""
    fps: int = -1
    total_num_frames: int = -1
    output_upscale_factor: int = 1
    visualization_mode: str = ""


class Config(pydantic.BaseModel):
    config_path: str
    class_mapping_path: str
    class_mapping: ClassMapping
    context: ContextConfig
    model: ModelSpec
    training: TrainingConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    whiteboard: Dict[str, Any] = {}

    @classmethod
    def from_paths(cls, config_path: str, class_mapping_path: str):
        class_mapping = yaml.load(open(class_mapping_path), Loader=yaml.FullLoader)
        config_dict = yaml.load(open(config_path), Loader=yaml.FullLoader)
        config_dict["config_path"] = config_path
        config_dict["class_mapping_path"] = class_mapping_path
        config_dict["class_mapping"] = class_mapping
        print(f" [Verres] - Read config from {config_path} and class mapping from {class_mapping_path}")
        return cls(**config_dict)
