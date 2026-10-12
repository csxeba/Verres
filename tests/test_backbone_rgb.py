import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

import verres as V
from verres.architecture.backbone.application import ApplicationBackbone

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = str(REPO_ROOT / "config/exp_od_overfit_mnv3.yml")
CLASS_MAPPING = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
HEIGHT, WIDTH = 208, 320


@pytest.fixture(scope="module")
def backbone():
    config = V.Config.from_paths(CONFIG_PATH, CLASS_MAPPING)
    assert config.model.backbone_spec["weights"] is None  # no ImageNet download
    return ApplicationBackbone(config)


def _bgr_image() -> np.ndarray:
    """BGR image with constant channels 1, 2, 3, as cv2 would load it."""
    image = np.zeros((1, HEIGHT, WIDTH, 3), dtype="float32")
    image[..., 0], image[..., 1], image[..., 2] = 1., 2., 3.
    return image


def test_preprocess_input_swaps_bgr_to_rgb(backbone):
    inputs = _bgr_image()
    output = np.asarray(backbone.preprocess_input(inputs))[0, :HEIGHT, :WIDTH]
    expected = np.asarray(tf.keras.applications.mobilenet_v3.preprocess_input(inputs[..., ::-1]))[0]
    np.testing.assert_allclose(output, expected)

    channels = [np.unique(output[..., c]) for c in range(3)]
    assert all(len(values) == 1 for values in channels)  # constant per channel
    assert [float(values[0]) for values in channels] == pytest.approx(
        np.asarray(tf.keras.applications.mobilenet_v3.preprocess_input(
            np.array([[[[3., 2., 1.]]]], dtype="float32"))).ravel().tolist())


def test_preprocess_input_is_channel_symmetric(backbone):
    inputs = np.random.default_rng(0).random((1, HEIGHT, WIDTH, 3)).astype("float32") * 255.
    forward = np.asarray(backbone.preprocess_input(inputs))
    swapped = np.asarray(backbone.preprocess_input(inputs[..., ::-1]))
    np.testing.assert_allclose(forward, swapped[..., ::-1])
