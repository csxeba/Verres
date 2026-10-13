import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import verres as V
from verres.architecture import VRSArchitecture
from verres.utils import keras_utils

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASS_MAPPING = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
CONFIG_PATHS = ["config/exp_od_baseline.yml", "config/exp_od_overfit.yml"]
ALL_CONFIG_PATHS = CONFIG_PATHS
OUTPUT_STRIDE = 4  # the OD head decodes on the neck's output stride
BATCH_SIZE = 2


def _load(relative_path: str) -> V.Config:
    return V.Config.from_paths(str(REPO_ROOT / relative_path), CLASS_MAPPING)


@pytest.mark.parametrize("config_path", ALL_CONFIG_PATHS)
def test_config_parsing_smokes(config_path):
    config = _load(config_path)
    total_steps = config.training.epochs * config.training.steps_per_epoch
    assert config.training.lr_schedule_spec["decay_steps"] == total_steps  # the cosine decay must span the run


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_od_architecture_forward(config_path):
    config = _load(config_path)
    config.model.backbone_spec["weights"] = None  # skip the ImageNet download, only shapes are under test
    architecture = VRSArchitecture.factory(config)

    height, width = config.model.input_shape_hw
    inputs = np.random.rand(BATCH_SIZE, height, width, 3).astype("float32") * 255.
    outputs = architecture(architecture.preprocess_input(inputs), training=False)

    assert set(outputs) == {"heatmap", "refinement", "box_wh"}
    for tensor in outputs.values():
        assert tensor.shape[0] == BATCH_SIZE
        assert tuple(tensor.shape[1:3]) == (height // OUTPUT_STRIDE, width // OUTPUT_STRIDE)
    assert outputs["heatmap"].shape[-1] == config.class_mapping.num_classes
    assert outputs["refinement"].shape[-1] == outputs["box_wh"].shape[-1] == 2
