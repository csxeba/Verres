import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import cv2

import verres as V
from verres.execution.inference import _apply_sequence_selector

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = str(REPO_ROOT / "config/exp_od_baseline.yml")
CLASS_MAPPING = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
DATASET_ROOT = "/data/Datasets/cocodoom"
SEQUENCE = "run1/map01"
NUM_FRAMES = 20


def _skip_without_dataset(annotation_name: str):
    if not (pathlib.Path(DATASET_ROOT) / annotation_name).exists():
        pytest.skip(f"COCODoom dataset is not available at {DATASET_ROOT}")


def _base_config() -> V.Config:
    return V.Config.from_paths(CONFIG_PATH, CLASS_MAPPING)


def _inference_spec(config: V.Config, subset: str, full: bool, **kwargs) -> V.config.DatasetSpec:
    """A single-image transformation spec: inference must not need any ground truth."""
    return V.config.DatasetSpec(
        name="cocodoom",
        root=DATASET_ROOT,
        subset=subset,
        transformations=[{"name": "image", "stride": 1}],
        kwargs={"full": full, "split": "run", **kwargs},
    )


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    _skip_without_dataset("run-train.json")
    config = _base_config()
    config.context.execution_type = V.execution.ExecutionType.INFERENCE
    config.inference.data = [_inference_spec(config, "train", False, sequence=SEQUENCE,
                                             frame_range=[0, NUM_FRAMES])]
    config.inference.to_screen = False
    config.inference.output_video_path = str(tmp_path_factory.mktemp("render") / "out.mp4")
    config.inference.output_upscale_factor = 1
    return config


@pytest.fixture(scope="module")
def executor(config):
    return V.execution.InferenceExecutor.factory(config)


def test_sequence_selector_yields_sequence_order(executor):
    dataset = executor.pipelines[0].dataset
    assert len(dataset.IDs) == NUM_FRAMES
    relative_paths = [os.path.relpath(dataset.index[ID].input.image_path, DATASET_ROOT) for ID in dataset.IDs]
    assert all(path.startswith(f"{SEQUENCE}/") for path in relative_paths)
    assert dataset.IDs == sorted(dataset.IDs)  # ids are monotonic in sequence order


def test_unknown_sequence_lists_available_ones():
    _skip_without_dataset("run-val-mini.json")
    config = _base_config()
    config.context.execution_type = V.execution.ExecutionType.INFERENCE
    spec = _inference_spec(config, "val-mini", False, sequence="run9/map99")
    pipeline = V.data.factory(config, specs=[spec])[0]

    with pytest.raises(RuntimeError, match="run2/map01"):
        _apply_sequence_selector(pipeline, spec)


def test_video_is_written_with_the_selected_frame_count(executor, config):
    executor.execute()

    output = pathlib.Path(config.inference.output_video_path)
    assert output.exists() and output.stat().st_size > 0

    capture = cv2.VideoCapture(str(output))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == NUM_FRAMES
        frames_read = 0
        while capture.read()[0]:
            frames_read += 1
    finally:
        capture.release()
    assert frames_read == NUM_FRAMES


def test_overlay_draws_boxes_labels_and_hud():
    visualizer = V.visualization.DataVisualizer(_base_config())
    frame = np.full((200, 320, 3), 60, dtype="uint8")
    corners = np.array([[0.2, 0.2, 0.6, 0.7]])

    canvas = visualizer.overlay_boxes(frame, corners, np.array([0]), np.array([0.9]), score_threshold=0.3)
    assert canvas.shape == frame.shape
    assert not np.array_equal(canvas, frame)  # box, colour and label were drawn
    assert np.array_equal(frame, np.full((200, 320, 3), 60, dtype="uint8"))  # input untouched

    # Detections below the threshold are dropped entirely.
    assert np.array_equal(visualizer.overlay_boxes(frame, corners, np.array([0]), np.array([0.1]),
                                                   score_threshold=0.3), frame)

    hud = visualizer.overlay_lines(np.full((200, 320, 3), 60, dtype="uint8"),
                                  ["compute 12.3 FPS", "model 45.6 ms", "frame 1/20"])
    assert not np.array_equal(hud, np.full((200, 320, 3), 60, dtype="uint8"))


def test_overlay_rejects_out_of_range_class():
    visualizer = V.visualization.DataVisualizer(_base_config())
    with pytest.raises(RuntimeError, match="class index"):
        visualizer.overlay_boxes(np.zeros((200, 320, 3), dtype="uint8"),
                                 np.array([[0.1, 0.1, 0.5, 0.5]]), np.array([99]))
