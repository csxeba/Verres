"""SmallFCNN must pad its input to the model stride, exactly like ApplicationBackbone does."""
import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

import verres as V

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = str(REPO_ROOT / "config/exp_od_overfit.yml")
CLASS_MAPPING_PATH = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
STRIDE = 8


def _backbone():
    return V.architecture.backbone.factory(V.Config.from_paths(CONFIG_PATH, CLASS_MAPPING_PATH))


def _image(height, width):
    return np.arange(height * width * 3, dtype="float32").reshape((1, height, width, 3))


def test_configured_input_is_not_padded():
    """200x320 is a multiple of the stride: preprocessing must stay a no-op there."""
    backbone = _backbone()
    assert np.asarray(backbone.preprocess_input(_image(200, 320))).shape == (1, 200, 320, 3)


def test_input_is_padded_to_the_model_stride():
    backbone = _backbone()
    padded = np.asarray(backbone.preprocess_input(_image(205, 320)))
    assert padded.shape == (1, 208, 320, 3)
    np.testing.assert_allclose(padded[:, :205], _image(205, 320) / 255.)
    assert np.all(padded[:, 205:] == 0.)


def test_stride_is_consumed_by_the_backbone():
    """The pooled features must keep the padded grid, i.e. 26x40 for a 205x320 input."""
    backbone = _backbone()
    features = backbone(backbone.preprocess_input(_image(205, 320)), training=False)
    assert [tuple(tf.keras.backend.int_shape(f)[1:3]) for f in features][-1] == (208 // STRIDE, 320 // STRIDE)
