import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

import verres as V
from verres.architecture import VRSArchitecture
from verres.execution import export

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASS_MAPPING = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
CONFIG_PATH = str(REPO_ROOT / "config/exp_od_small_fcnn.yml")


def _build(config_path: str = CONFIG_PATH):
    config = V.Config.from_paths(config_path, CLASS_MAPPING)
    config.model.backbone_spec["weights"] = None  # random weights are enough for export round-trips
    return config, VRSArchitecture.factory(config)


@pytest.fixture(scope="module")
def model_and_input():
    config, model = _build()
    height, width = config.model.input_shape_hw
    inputs = np.random.rand(1, height, width, 3).astype("float32") * 255.
    return model, inputs


def test_saved_model_round_trip(tmp_path, model_and_input):
    model, inputs = model_and_input
    expected = model(model.preprocess_input(inputs), training=False)

    path = export.export_saved_model(model, str(tmp_path / "saved_model"))
    serving = tf.saved_model.load(path).signatures["serving_default"]
    actual = serving(tf.constant(inputs))

    assert set(actual) == set(expected)
    for name, tensor in expected.items():
        np.testing.assert_allclose(actual[name].numpy(), tensor.numpy(), rtol=1e-6, atol=1e-6)


def test_tflite_conversion(tmp_path, model_and_input):
    model, inputs = model_and_input
    try:
        path = export.export_tflite(model, str(tmp_path / "model.tflite"))
    except Exception as error:  # converter op coverage depends on the TF build
        pytest.skip(f"TFLite conversion unsupported here: {error}")

    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    assert interpreter.get_input_details()
    assert interpreter.get_output_details()


def test_load_weights_into(tmp_path, model_and_input):
    model, inputs = model_and_input
    run_dir = tmp_path / "xp_test"
    (run_dir / "checkpoints").mkdir(parents=True)
    model.save_weights(str(run_dir / "checkpoints" / "latest.h5"))
    assert np.any(model.weights[0].numpy())  # a zero-initialised model would make the check vacuous

    _, fresh = _build()
    used = export.load_weights_into(fresh, str(run_dir))

    assert used.endswith("latest.h5")
    expected = model(model.preprocess_input(inputs), training=False)
    actual = fresh(fresh.preprocess_input(inputs), training=False)
    for name, tensor in expected.items():
        np.testing.assert_allclose(actual[name].numpy(), tensor.numpy(), rtol=1e-6, atol=1e-6)


def test_load_weights_into_prefers_the_highest_map(tmp_path, model_and_input):
    """A resumed run leaves several best-epoch files behind, and they do not sort by mAP."""
    model, _ = model_and_input
    run_dir = tmp_path / "xp_resumed"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    model.save_weights(str(checkpoints / "chkp_epoch_4_map_0.4000.h5"))
    model.save_weights(str(checkpoints / "chkp_epoch_5_map_0.1000.h5"))
    model.save_weights(str(checkpoints / "chkp_epoch_12_map_0.1000.h5"))

    assert export.load_weights_into(_build()[1], str(run_dir)).endswith("chkp_epoch_4_map_0.4000.h5")


def test_load_weights_into_missing_checkpoint(tmp_path, model_and_input):
    _, model = model_and_input
    with pytest.raises(FileNotFoundError, match="No weights in"):
        export.load_weights_into(model, str(tmp_path / "empty_run"))
