import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import verres as V
from verres.architecture import VRSArchitecture

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASS_MAPPING = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
CONFIG_PATH = str(REPO_ROOT / "config/exp_od_small_fcnn.yml")
BATCH_SIZE = 1


@pytest.fixture(scope="module")
def forward():
    config = V.Config.from_paths(CONFIG_PATH, CLASS_MAPPING)
    config.model.backbone_spec["weights"] = None  # skip the ImageNet download
    architecture = VRSArchitecture.factory(config)

    height, width = config.model.input_shape_hw
    inputs = np.random.rand(BATCH_SIZE, height, width, 3).astype("float32") * 255.
    outputs = architecture(architecture.preprocess_input(inputs), training=False)
    return architecture.head, outputs


def _last_bias(head) -> np.ndarray:
    """The final output Conv2D of a VRSHead stack."""
    return np.asarray(head.layer_objects[-1].layer_objects[0].bias)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def test_heatmap_prior(forward):
    head, outputs = forward
    probability = _sigmoid(outputs["heatmap"].numpy())
    assert 0.05 < probability.mean() < 0.2
    np.testing.assert_allclose(_last_bias(head.heads[0]), np.log(0.1 / 0.9), rtol=1e-4)


def test_other_heads_start_at_zero(forward):
    head, _ = forward
    for other in (head.heads[1], head.heads[2]):  # refinement, box_wh
        np.testing.assert_array_equal(_last_bias(other), np.zeros_like(_last_bias(other)))
