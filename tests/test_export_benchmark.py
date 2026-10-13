import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow")
# The tool runs on ONNX Runtime/OpenVINO, which only the export venv has; elsewhere the module skips cleanly.
pytest.importorskip("tf2onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("openvino")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import export_benchmark as tool  # noqa: E402

CONFIG_PATH = str(REPO_ROOT / "config/exp_od_baseline.yml")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Random-weight SmallFCNN exported to SavedModel + ONNX once for the whole module."""
    directory = tmp_path_factory.mktemp("export")
    config, model, codec = tool.build(CONFIG_PATH)
    saved_model_dir = tool.export.export_saved_model(model, str(directory / "saved_model"))
    return config, model, codec, tool.to_onnx(saved_model_dir, directory / "model.onnx")


def test_timeit_percentiles():
    p50, p95 = tool.timeit(lambda: np.zeros(64) + 1., 30, 2)
    assert p50 <= p95


def test_onnx_matches_keras(exported):
    config, model, codec, onnx_path = exported
    image = tool.load_image(config)
    reference = {name: np.asarray(tensor) for name, tensor in tool.serving(model)(image).items()}
    heads = tool.ort_runner(onnx_path, 1)(image)

    assert set(tool.HEADS) <= set(heads)
    for name, tensor in reference.items():
        assert np.abs(tensor - heads[name]).max() < 1e-3

    keras_label = tool.decode_heads(codec, reference)
    assert len(tool.decode_heads(codec, heads).object_scores) == len(keras_label.object_scores)


def test_openvino_matches_keras(exported):
    config, model, _, onnx_path = exported
    image = tool.load_image(config)
    reference = {name: np.asarray(tensor) for name, tensor in tool.serving(model)(image).items()}
    heads = tool.ov_runner(onnx_path, 1)(image)

    assert set(tool.HEADS) <= set(heads)
    for name, tensor in reference.items():
        assert np.abs(tensor - heads[name]).max() < 1e-3
