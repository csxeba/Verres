import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import verres as V
from verres.architecture import VRSArchitecture
from verres.utils import keras_utils

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASS_MAPPING = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
CONFIG_PATHS = ["config/exp_od_small_fcnn.yml", "config/exp_od_baseline.yml"]
COLAB_CONFIG_PATHS = ["config/exp_od_small_fcnn_colab.yml", "config/exp_od_baseline_colab.yml",
                      "config/exp_od_cpu_colab.yml"]
ALL_CONFIG_PATHS = CONFIG_PATHS + COLAB_CONFIG_PATHS + ["config/exp_panoptic_baseline.yml"]
OUTPUT_STRIDE = 4  # the OD head decodes on the neck's output stride
BATCH_SIZE = 2


def _load(relative_path: str) -> V.Config:
    return V.Config.from_paths(str(REPO_ROOT / relative_path), CLASS_MAPPING)


@pytest.mark.parametrize("config_path", ALL_CONFIG_PATHS)
def test_config_parsing_smokes(config_path):
    config = _load(config_path)
    boundaries = config.training.lr_schedule_spec["boundaries"]
    values = config.training.lr_schedule_spec["values"]
    total_steps = config.training.epochs * config.training.steps_per_epoch
    assert len(values) == len(boundaries) + 1  # one LR value per interval
    assert 0.5 * total_steps < max(boundaries) < total_steps  # decay must span the run, not finish early


def test_backbone_weights_reach_keras(monkeypatch):
    config = _load("config/exp_od_baseline.yml")
    seen = {}
    original_make_model = keras_utils.ApplicationCatalogue.make_model

    def spy(self, model_name, **kwargs):
        seen["weights"] = kwargs.get("weights")
        return original_make_model(self, model_name, **{**kwargs, "weights": None})  # skip the download

    monkeypatch.setattr(keras_utils.ApplicationCatalogue, "make_model", spy)
    VRSArchitecture.factory(config)
    assert seen["weights"] == config.model.backbone_spec["weights"] == "imagenet"


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
